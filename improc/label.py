import numpy as np

from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
from scipy.ndimage import label as nd_label
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.filters import gaussian_filter

import warnings


def relabel_size_sorted(labels):
    '''Relabels image with labels sorted according to the size of the objects'''

    labels, _, _ = relabel_sequential(labels)
    _, l_count = np.unique(labels, return_counts=True)
    # ignore background
    l_count = l_count[1:]
    lut = np.concatenate([[0], np.argsort(np.argsort(l_count)[::-1]) + 1])
    return lut[labels]


def find_objects_bb(mask):
    '''Returns the bounding boxes of objects found in mask, sorted by size.'''

    labels, _ = nd_label(mask)
    labels = relabel_size_sorted(labels)

    return find_objects(labels)


def size_opening(labels, threshold):
    '''Removes labels with size less than 'threshold'
    '''

    lut, counts = np.unique(labels, return_counts=True)
    lut[counts < threshold] = 0
    labels = lut[labels]

    return labels


def smooth_label_edges(labels, sigma, area_eps=0.0001, n_iter_max=100):
    '''smooth labels one by one.
    
    Attempts to maintain the original area/volume of each mask
    '''

    locs = find_objects(labels)
    sigma = np.broadcast_to(np.asarray(sigma), labels.ndim)
    margin = np.rint(4 * sigma).astype(int)
    smoothed_labels = np.zeros_like(labels)

    for l, loc in enumerate(locs, start=1):
        if loc is None:
            continue

        loc = tuple(
            slice(max(0, s.start - m), s.stop + m)
            for s, m in zip(loc, margin))
        mask = labels[loc] == l
        blurred_mask = gaussian_filter(mask.astype(np.float32),
                                       sigma,
                                       mode='constant',
                                       cval=0.)

        # iteratively adjust threshold until the smooth mask has the same area as input
        threshold = 0.5
        threshold_best = 0.5
        mask_sum = mask.sum()
        min_n_px_diff = mask_sum
        px_eps = max(1, area_eps * mask_sum)

        for i in range(n_iter_max):
            smooth_mask = blurred_mask > threshold
            smooth_mask_sum = smooth_mask.sum()
            n_px_diff = np.abs(mask.sum() - smooth_mask.sum())

            # safeguard in case it does not converge
            if min_n_px_diff > n_px_diff:
                min_n_px_diff = min_n_px_diff
                threshold_best = threshold

            if n_px_diff < px_eps:
                break

            threshold = threshold * smooth_mask_sum / mask_sum

        smooth_mask = blurred_mask > threshold_best
        smoothed_labels[loc][smooth_mask] = l

    return smoothed_labels
