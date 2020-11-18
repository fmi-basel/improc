import numpy as np

from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops
from scipy.ndimage import label as nd_label
from scipy.ndimage.measurements import find_objects

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
