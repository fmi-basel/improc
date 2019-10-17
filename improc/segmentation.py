import itertools
import numpy as np

from skimage.filters import threshold_otsu
from scipy.ndimage.filters import gaussian_filter


def segment_from_projections(image, spacing=1, sigma=2):
    '''Segments a 3 dimensional image from its projected planes and returns the intersection'''

    spacing = np.broadcast_to(np.asarray(spacing), image.ndim)
    mask = np.ones(image.shape, dtype=bool)
    for proj_axis, (d1, d2) in enumerate(
            reversed(list(itertools.combinations(range(image.ndim), 2)))):

        proj = image.max(axis=proj_axis)
        smoothed_proj = gaussian_filter(proj, sigma=sigma / spacing[[d1, d2]])
        mask_proj = smoothed_proj > threshold_otsu(smoothed_proj)
        mask = np.logical_and(mask, np.expand_dims(mask_proj, axis=proj_axis))

    return mask
