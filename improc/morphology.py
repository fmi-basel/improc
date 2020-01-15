import numpy as np

from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt, binary_opening
from scipy.ndimage import find_objects
from scipy.ndimage import label as nd_label

from improc.label import relabel_size_sorted


def fill_holes_sliced(mask):
    '''slice-wise binary hole filling of 3D stack'''

    if mask.ndim == 2:
        return binary_fill_holes(mask)
    else:
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])

        return mask


def split_and_size_filter(mask, size_threshold=None):
    '''
    split disconnected components and keep largest or keep larger that 
    size_threshold if specified.
    '''

    labels, n_split = nd_label(mask)
    labels = relabel_size_sorted(labels)

    if size_threshold is not None:
        unique_l, count = np.unique(labels, return_counts=True)
        small_l = unique_l[count < size_threshold]

        if len(small_l) > 0:
            labels[labels >= min(small_l)] = 0

        return labels > 0
    else:
        return labels == 1


def anisotropic_sphere_struct(radius, spacing):
    '''Returns binary n-sphere with possibly anisotropic sampling'''

    spacing = np.asarray(spacing)
    shape = (radius / spacing).astype(int) * 2 + 1
    struct = np.zeros(shape, dtype=bool)
    struct[tuple(s // 2 for s in shape)] = 1
    struct = distance_transform_edt(struct == 0, sampling=spacing) < radius

    return struct


def clean_up_mask(mask, struct=None, fill_holes=True, size_threshold=None):
    '''slice-wise hole filling, binary opening to clean jagged edges 
    (optional) and removal of small disconnected components.'''

    # process only object region with one pixel padding
    loc = find_objects(mask)[0]
    loc = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in loc)

    submask = mask[loc]

    if fill_holes:
        submask = fill_holes_sliced(mask[loc])

    if struct is not None:
        submask = binary_opening(submask, struct)

    mask[loc] = split_and_size_filter(submask, size_threshold)

    return mask


def clean_up_labels(labels,
                    idxs=None,
                    struct=None,
                    fill_holes=True,
                    size_threshold=None):
    '''Cleans up labels, one at a time.
    
    Fill holes (slice wise), binary opening with struct if provided
    and removes small disconnected components.
    
    Optionaly only processes/keeps the indices passed in arguments'''

    if idxs is None:
        idxs = np.unique(labels)[1:]

    clean_labels = np.zeros_like(labels)
    for l in idxs:

        mask = clean_up_mask(labels == l, struct, fill_holes, size_threshold)
        clean_labels[mask] = l

    return clean_labels
