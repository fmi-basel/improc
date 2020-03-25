import numpy as np
from multiprocessing import Pool
import os
from functools import lru_cache

from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt, binary_opening
from scipy.ndimage import find_objects
from scipy.ndimage import label as nd_label
from tqdm import tqdm

from improc.label import relabel_size_sorted


def fill_holes_sliced(mask):
    '''slice-wise binary hole filling of 3D stack'''

    if mask.ndim == 2:
        return binary_fill_holes(mask)
    else:
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])

        return mask


def split_and_size_filter(mask, size_threshold=None, keep_largest=False):
    '''
    Returns the largest of disjoint components or union of components 
    larger than size_threshold if specified.
    '''

    if size_threshold is None and not keep_largest:
        # nothing to do
        return mask

    labels, n_split = nd_label(mask)
    labels = relabel_size_sorted(labels)

    if keep_largest:
        labels[labels != 1] = 0

    if size_threshold is not None:
        unique_l, count = np.unique(labels, return_counts=True)
        small_l = unique_l[count < size_threshold]

        if len(small_l) > 0:
            labels[labels >= min(small_l)] = 0

    return labels > 0


@lru_cache(maxsize=1)
def anisotropic_sphere_struct(radius, spacing):
    '''Returns binary n-sphere with possibly anisotropic sampling'''

    spacing = np.asarray(spacing)
    shape = (radius / spacing).astype(int) * 2 + 1
    struct = np.zeros(shape, dtype=bool)
    struct[tuple(s // 2 for s in shape)] = 1
    struct = distance_transform_edt(struct == 0, sampling=spacing) < radius

    return struct


def clean_up_mask(mask,
                  fill_holes=True,
                  radius=None,
                  size_threshold=None,
                  keep_largest=False,
                  spacing=1):
    '''slice-wise hole filling, binary opening to clean jagged edges 
    (optional) and removal of small disconnected components.'''

    if fill_holes:
        mask = fill_holes_sliced(mask)

    if radius is not None:
        spacing = tuple(np.broadcast_to(np.array(spacing), mask.ndim))
        struct = anisotropic_sphere_struct(radius, spacing)
        mask = binary_opening(mask, struct)

    mask = split_and_size_filter(mask, size_threshold, keep_largest)

    return mask


def _pooled_clean_up_mask(packed_inputs):

    if packed_inputs is None:
        return None
    else:
        return clean_up_mask(*packed_inputs)


# NOTE would be a better fit in improc.label but circular import dependency with "relabel_size_sorted"...??
def clean_up_labels(labels,
                    fill_holes=True,
                    radius=None,
                    size_threshold=None,
                    keep_largest=False,
                    spacing=1,
                    processes=None):
    '''Cleans up labels, one at a time.
    
    Fill holes (slice wise), binary opening if radius provided
    and removes small disconnected components.
    
    Optionaly only processes/keeps the indices passed in arguments'''

    locs = find_objects(labels)
    # expand boundary by one px
    locs = [
        tuple(slice(max(0, s.start - 1), s.stop + 1)
              for s in loc) if loc else None for loc in locs
    ]

    clean_labels = np.zeros_like(labels)

    # create generator for cleaning label masks and multiprocess
    cleanup_inputs = ((labels[loc] == l, fill_holes, radius, size_threshold,
                       keep_largest, spacing) if loc else None
                      for l, loc in enumerate(locs, start=1))

    # increase chunksize for large number of items
    if processes is None:
        processes = os.cpu_count(
        )  # default value in Pool, but needed for chunksize
    chunksize = max(1, len(locs) // (processes * 10))

    with Pool(processes=processes) as pool:
        # ~for l, (loc, mask) in enumerate(zip(tqdm(locs), pool.imap(_pooled_clean_up_mask, cleanup_inputs, chunksize)), start=1):
        for l, (loc, mask) in enumerate(zip(
                locs,
                pool.imap(_pooled_clean_up_mask, cleanup_inputs, chunksize)),
                                        start=1):
            if loc:
                clean_labels[loc][mask] = l

    return clean_labels
