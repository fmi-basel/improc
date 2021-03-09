from PIL import Image
from PIL.TiffTags import TAGS
import json
from skimage.io import imsave
import itertools
import numpy as np
import matplotlib.pyplot as plt


def read_tif_meta(path):
    with Image.open(path) as img:
        meta_dict = {TAGS[key]: val[0] for key, val in img.tag.items()}

    meta_dict['ImageDescription'] = json.loads(
        meta_dict.get('ImageDescription', '{}'))
    return meta_dict


def imagej_imsave(path, img, cmaps=None, intensity_bounds=None, spacing=None):
    '''Saves an image as a tif file compatible with imageJ.
    Colormaps and intensity bounds are embedded in metadata and 
    automatically applied when opening in imageJ
    
    Args:
        path: output path
        img: image array with channels as last dim
        cmaps: list of colormaps for each channels
        intensity_bounds: list of pairs of clipping bounds for each channel
        spacing: pixel/voxel size
    '''
    def imagej_lut(matplotlib_lut):
        val_range = np.arange(256, dtype=np.uint8)
        return (plt.get_cmap(matplotlib_lut)(val_range)[..., :-1].T *
                255).astype(np.uint8)

    ijmeta = {}
    if cmaps is not None:
        ijmeta['LUTs'] = [imagej_lut(name) for name in cmaps]
    if intensity_bounds is not None:
        ijmeta['Ranges'] = [list(itertools.chain(*intensity_bounds))]

    resolution = None if spacing is None else [1 / s for s in spacing]
    imsave(path,
           np.moveaxis(img, -1, 0),
           imagej=True,
           compress=9,
           metadata={'mode': 'composite'},
           ijmetadata=ijmeta,
           resolution=resolution)
