import numpy as np

from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from scipy.ndimage import find_objects
from math import ceil


def extend_bbox(bbox, max_shape, spacing, border):
    '''pads the bounding box to have (border / spacing) pixels more.

    '''
    def pad(bbox_range, dim_max, dim_spacing):
        '''create a padded slice while respecting the original dimension
        of the array.

        '''
        padding = round(border / dim_spacing)
        return slice(max(0, bbox_range.start - padding),
                     min(dim_max, bbox_range.stop + padding))

    return tuple(
        pad(bounding_range, dim_max, dim_spacing)
        for bounding_range, dim_max, dim_spacing in zip(
            bbox, max_shape, spacing))


def shift_roi(bb, centers, spacing=1):
    '''Moves the given bounding box to the closet center.'''

    spacing = np.broadcast_to(np.asarray(spacing), len(bb))
    bb_center = np.asarray([(s.stop + s.start) / 2 for s in bb])

    distances = cdist([bb_center * spacing], centers * spacing)
    closest_center_id = np.argmin(distances)
    closest_center = centers[closest_center_id]

    offset = tuple(
        int(round(c2 - c1)) for c1, c2 in zip(bb_center, closest_center))

    return tuple(
        slice(s.start + off, s.stop + off) for s, off in zip(bb, offset))


def padded_crop(image, roi, mode='constant', constant_values=0):
    padding = [(max(0, -s.start), max(0, s.stop - size))
               for s, size in zip(roi, image.shape)]
    roi = tuple(
        slice(max(0, s.start), min(dim, s.stop))
        for s, dim in zip(roi, image.shape))

    return np.pad(image[roi], padding, mode, constant_values=constant_values)


def crop_object(images,
                labels,
                margins=0,
                min_shape=0,
                mode='constant',
                constant_values=0):
    '''
    Crop all images based on object found in labels.

    To crop labels itself, pass it in the images list as well.
    '''

    margins = np.broadcast_to(np.asarray(margins), labels.ndim).copy()
    min_shape = np.broadcast_to(np.asarray(min_shape), labels.ndim)

    loc = find_objects(labels >= 1)[0]

    # if needed, increase margins to output object at least min_shape
    object_shape = np.asarray(tuple(sli.stop - sli.start for sli in loc))
    margins = np.maximum(margins,
                         np.ceil((min_shape - object_shape) / 2).astype(int))

    loc = tuple(
        slice(sli.start - margin, sli.stop + margin)
        for sli, margin in zip(loc, margins))

    return [padded_crop(image, loc, mode, constant_values) for image in images]
