import numpy as np

from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
from scipy.ndimage import label as nd_label


def relabel_size_sorted(labels):
    '''Relabels image with labels sorted according to the size of the objects'''

    labels, _, _ = relabel_sequential(labels)
    _, l_count = np.unique(labels, return_counts=True)
    # ignore background
    l_count = l_count[1:]
    lut = np.concatenate([[0], np.argsort(np.argsort(l_count)[::-1]) + 1])
    return lut[labels]


def find_objects_center(mask, image, spacing=1):
    '''Returns the center of mass and bounding boxes sorted by size of all objects found in mask.'''

    labels, _ = nd_label(mask)
    labels = relabel_size_sorted(labels)

    properties = regionprops(labels, image)
    centers = [p.weighted_centroid for p in properties]

    def format_bb(bb):
        '''converts bbox list to tupel of slices'''
        bb = np.array(bb).reshape(2, -1).T
        return tuple(slice(start, stop) for start, stop in bb)

    bboxes = [format_bb(p.bbox) for p in properties]

    return centers, bboxes
