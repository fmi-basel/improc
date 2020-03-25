import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import find_objects
from skimage.transform import rescale


def resample_labels(labels, factor):
    '''Resample labels one by one with a gaussian kernel'''

    # TODO check alignment for float re-scaling factors

    factor = np.broadcast_to(np.asarray(factor), labels.ndim)
    resampled_labels = np.zeros(np.round(labels.shape * factor).astype(int),
                                dtype=np.int16)

    locs = find_objects(labels)
    locs = [loc for loc in locs if loc is not None]

    # ignore label 0 (background)
    for l, loc in zip(filter(None, np.unique(labels)), locs):

        margin = np.ceil(factor.max()).astype(int)
        loc = tuple(
            slice(max(0, s.start - margin), s.stop + margin) for s in loc)

        mask = labels[loc] == l

        # get label coordinates and upsample
        coords = np.round(np.where(mask) * factor[:, None]).astype(int)
        coords = tuple([*coords])

        # create upsampled mask
        resampled_mask = np.zeros(np.round(mask.shape * factor).astype(int),
                                  dtype=np.float32)
        resampled_mask[coords] = np.prod(factor)

        # interpolate upsampled image
        resampled_mask = gaussian_filter(resampled_mask, sigma=factor.max())

        loc_resampled = tuple(
            slice(
                np.round(f * s.start).astype(int),
                np.round(f * s.start).astype(int) + s_size)
            for s, f, s_size in zip(loc, factor, resampled_mask.shape))

        # TODO figure out general solution/threshold for different factors
        # 0.48 instead of 0.5 --> slightly larger label mask to prevent introduction of background between touching labels
        # not general, e.g. works for (2,0.26,26) to 1, but not from 25 to 37x
        resampled_labels[loc_resampled][resampled_mask > 0.48] = l

    return resampled_labels


def match_spacing(img,
                  src_spacing,
                  dst_spacing='isotropic',
                  image_type='greyscale'):
    '''
    TODO
    
    args:
    image_type: one of 'greyscale', 'label_nearest', 'label_interp'
    
    '''

    input_dtype = img.dtype
    src_spacing = np.asarray(src_spacing)

    if dst_spacing == 'isotropic':
        scale = src_spacing / src_spacing.min()
    else:
        dst_spacing = np.asarray(dst_spacing)
        scale = np.array(src_spacing) / np.array(dst_spacing)

    if image_type == 'greyscale':
        return rescale(img.astype(np.float32),
                       scale=scale,
                       multichannel=False,
                       anti_aliasing=True,
                       order=1).astype(input_dtype)

    elif image_type == 'label_interp':
        return resample_labels(img, scale)

    elif image_type == 'label_nearest':
        return rescale(img.astype(np.float32),
                       scale=scale,
                       multichannel=False,
                       anti_aliasing=False,
                       order=0).astype(input_dtype)

    else:
        raise ValueError(
            'image type "{}" not supported, expected "greyscale", "label_nearest" or "label_interp"'
            .format(image_type))
