import itertools
import numpy as np

from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_minimum
from scipy.ndimage.filters import gaussian_filter

THRESHOLDS_METHODS = {
    'otsu': threshold_otsu,
    'li': threshold_li,
    'yen': threshold_yen,
    'triangle': threshold_triangle,
    'minimum': threshold_minimum
}


def segment_from_projections(image,
                             spacing=1,
                             sigma=2,
                             threshold_method='otsu'):
    """
    Segments a 3 dimensional image from its projected planes and returns the intersection mask
    
    Parameters
    ----------
    image : array_like
       3D image to segment.
    spacing : float or tuple of float
       voxel size of the image stack
    sigma : float
        blurring sigma for segmentation
    threshold_method : str or callable
        threshold method to use. accepts one of ['otsu', 'li', 'yen', 'triangle', 'minimum'] or a callable
        
    Returns
    -------
    mask : array_like
        3D mask generated from interesction of 2D semgentations
    
    """

    if isinstance(threshold_method, str):
        try:
            threshold_method = THRESHOLDS_METHODS[threshold_method]
        except Exception as e:
            raise NotImplementedError(
                '{} threshold method not implemented'.format(
                    'threshold_method'))

    spacing = np.broadcast_to(np.asarray(spacing), image.ndim)
    mask = np.ones(image.shape, dtype=bool)

    for proj_axis, (d1, d2) in enumerate(
            reversed(list(itertools.combinations(range(image.ndim), 2)))):

        proj = image.max(axis=proj_axis)
        proj = gaussian_filter(proj, sigma=sigma / spacing[[d1, d2]])
        mask_proj = proj > threshold_method(proj)
        mask = np.logical_and(mask, np.expand_dims(mask_proj, axis=proj_axis))

    return mask
