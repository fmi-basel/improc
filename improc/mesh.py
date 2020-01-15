import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import find_objects


def resample_labels(labels, factor):
    '''Resample labels one by one with a gaussian kernel'''

    # TODO check alignment for float re-scaling factors

    factor = np.broadcast_to(np.asarray(factor), labels.ndim)
    resampled_labels = np.zeros(np.round(labels.shape * factor).astype(int),
                                dtype=np.int16)

    # ignore label 0 (background)
    for l in filter(None, np.unique(labels)):

        loc = find_objects(labels == l)[0]
        loc = tuple(
            slice(max(0, s.start - f - 1), s.stop + f + 1)
            for s, f in zip(loc,
                            np.ceil(factor).astype(int)))

        mask = labels[loc] == l

        # get label coordinates and upsample
        coords = np.round(np.where(mask) * factor[:, None]).astype(int)
        coords = tuple([*coords])

        # create upsampled mask
        resampled_mask = np.zeros(np.ceil(mask.shape * factor).astype(int),
                                  dtype=np.float32)
        resampled_mask[coords] = np.prod(factor)

        # interpolate upsampled image
        resampled_mask = gaussian_filter(resampled_mask, sigma=factor.max())

        loc_resampled = tuple(
            slice(
                np.round(f * s.start).astype(int),
                np.round(f * s.start).astype(int) + s_size)
            for s, f, s_size in zip(loc, factor, resampled_mask.shape))
        resampled_labels[loc_resampled][resampled_mask > 0.5] = l

    return resampled_labels


# TODO mesh export (clean up draft from jupyter notebook)
# TODO mesh simplification/resampling (smooting currently done paraview)
