from PIL import Image
from PIL.TiffTags import TAGS
import json


def read_tif_meta(path):
    with Image.open(path) as img:
        meta_dict = {TAGS[key]: val[0] for key, val in img.tag.items()}

    meta_dict['ImageDescription'] = json.loads(
        meta_dict.get('ImageDescription', '{}'))
    return meta_dict
