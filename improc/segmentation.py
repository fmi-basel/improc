import itertools
import numpy as np
import cv2 as cv

from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_minimum
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label as nd_label
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential

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

        if proj.max() <= 0:
            # if smoothed projection is empty, default to background
            # can happen even if the input image has small objects that get smoothed "away" after rounding
            mask_proj = np.zeros_like(proj, dtype=bool)
        else:
            mask_proj = proj > threshold_method(proj)
        mask = np.logical_and(mask, np.expand_dims(mask_proj, axis=proj_axis))

    return mask


def quantiles_refine_mask(img, mask, spacing, sigma=1, threshold=0.5):
    '''Refines a binary mask by computing background/foreground levels from medians over existing mask.
    
    Can be used to refine a rough 3D mask obtained from max projections or locally over a cropped region.
    
    Args:
        img: image to segment
        mask: rough mask (e.g. obtained from max projections)
        spacing: pixel/voxel size
        sigma: image smoothing sigma before thresholding
        threshold: relative threshold between median background and foreground levels
    '''

    spacing = np.broadcast_to(np.asarray(spacing), img.ndim)
    img = gaussian_filter(img, sigma=sigma / spacing)

    bg_level = np.quantile(img[~mask], 0.5)
    fg_level = np.quantile(img[mask], 0.5)

    return img > bg_level + (fg_level - bg_level) * threshold


def label_with_watershed(mask, sigma=2, spacing=1):
    '''Return labeled objects from mask while attempting to split touching objects
    
    Args:
        mask: binary mask
        sigma: distance smoothing sigma to find local maxima
        spacing: pixel/voxel size
    '''

    spacing = np.broadcast_to(np.asarray(spacing), mask.ndim)
    distance = distance_transform_edt(mask, sampling=spacing)

    smoothed_distance = gaussian_filter(distance, sigma=sigma / spacing)

    markers = peak_local_max(smoothed_distance,
                             indices=False,
                             labels=mask,
                             exclude_border=False)
    markers, _ = nd_label(markers)
    labels = watershed(-distance, markers=markers, mask=mask)

    return labels


def approx_gaussian_blur(image, ksize=1001):
    '''Fast blur approx
    
    Notes:
    ------
    fast gaussian filter approximation:
    
    KOVESI, Peter. Fast almost-gaussian filtering. In: Digital Image Computing:
    Techniques and Applications (DICTA), 2010 International Conference on. IEEE, 2010. S. 121-125.
    '''

    image = cv.boxFilter(image, -1, (ksize, ksize))
    image = cv.boxFilter(image, -1, (ksize, ksize))
    image = cv.boxFilter(image, -1, (ksize, ksize))

    return image
