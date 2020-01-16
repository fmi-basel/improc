import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import maximum_filter


def local_max(image, min_distance=1, threshold=0.5, spacing=1):
    '''Finds local maxima that are above threshold.
    
    Notes:
    unlike skimage.feature.peak_local_max, Supports anisotropic spacing'''

    if min_distance < 1:
        raise ValueError('min_distance should be > 1: {}'.format(min_distance))

    spacing = np.broadcast_to(np.asarray(spacing), image.ndim)

    # smooth details smaller than radius of interest
    blurred_image = gaussian_filter(image,
                                    sigma=np.sqrt(min_distance / spacing))
    # maintain the max value regardless of sigma to facilitate threshold selection
    blurred_image *= (image.max() / blurred_image.max())

    max_filt_size = np.maximum(1, min_distance / spacing * 2 + 1)
    max_image = maximum_filter(blurred_image, size=max_filt_size)
    markers = np.argwhere((max_image == blurred_image)
                          & (blurred_image >= threshold))
    if len(markers) == 0:
        return [], []

    intensities = blurred_image[tuple(np.transpose(markers))]

    # sort markers/intensities by decreasing intensities
    return tuple(
        list(t) for t in zip(*sorted(
            zip(markers, intensities), key=lambda x: x[1], reverse=True)))
