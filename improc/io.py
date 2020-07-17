import re
import parse
from glob import glob
from skimage.io import imread, imsave
from PIL import Image
import pandas as pd
import numpy as np
import os
import warnings

# TODO add logger


def parse_collection(pattern, keys):
    '''Parse a data collection and returns a dataframe of parsed attributes with keys as index.
    
    Pattern example:
    '/dir1/dir2/dir3/{channel}/T{time:04d}.{ext}'
    '''

    pattern = os.path.normpath(pattern)
    compiled_pattern = parse.compile(pattern)
    paths = sorted(glob(re.sub('{.*?}', '*', pattern)))

    # extract basedir from pattern
    if '{basedir}' in pattern:
        raise ValueError(
            'basedir key is reserved and cannot be used in pattern: {}'.format(
                pattern))

    basedir = os.path.dirname(pattern.split('{', 1)[0])
    pattern = pattern.replace(basedir, '{basedir}')

    # apply pattern to all paths that matched
    parsed_paths = [compiled_pattern.parse(p) for p in paths]

    # build df, exclude file that matched the glob pattern bu not the formatting ( returns None)
    df = (pd.DataFrame([{
        'pattern': pattern,
        'basedir': basedir,
        **pp.named
    } for pp in parsed_paths if pp]).set_index(keys).sort_index())

    duplicated_index_mask = df.index.duplicated()
    if duplicated_index_mask.sum():

        duplicated_index = df[duplicated_index_mask].index
        example = df.loc(axis=0)[duplicated_index[0]].to_string()

        raise ValueError(
            'Selected keys {} do not form a unique index. e.g. \n{}'.format(
                keys, example))

    return df


# ~ @pd.api.extensions.register_series_accessor('lsc')
class LSCAccessor:
    '''Adds cached read/write accessors to pandas series representing lightsheet items.'''
    def __init__(self, pandas_obj):
        if isinstance(pandas_obj, pd.Series):
            self._obj = pandas_obj.to_frame().T
        else:
            self._obj = pandas_obj

        # TODO check that _obj a pattern and necessary keys

    def __getitem__(self, key):
        '''Returns a dataframe of requested index with all levels maintained, even if it has only one row'''

        # TODO fix indexing with list for fir level only: df.lsc[['toto','tata']]

        if isinstance(key, slice) or (isinstance(
                key,
            (list, tuple, np.ndarray)) and any(
                [isinstance(k, (slice, list, tuple, np.ndarray))
                 for k in key])):
            return self._obj.loc(axis=0)[key]
        else:
            # prevent from returning a pd.Series or dropping index level
            if isinstance(self._obj.index, pd.MultiIndex) and isinstance(
                    key,
                (list, tuple,
                 np.ndarray)) and len(key) != len(self._obj.index.levels):
                return self._obj.xs(key, drop_level=False)
            else:
                return self._obj.loc(axis=0)[[key]]

    @classmethod
    def register(cls):
        pd.api.extensions.register_series_accessor('lsc')(cls)
        pd.api.extensions.register_dataframe_accessor('lsc')(cls)

    def read(self):
        return [
            self.map_read_fct(p, ext)
            for p, ext in zip(self.path, self._obj.ext)
        ]

    def write(self,
              data,
              index=True,
              compressed=False,
              imagej=False,
              compress=0):

        # if data is not a list, we assume it is because the is only one item
        if not isinstance(data, list):
            data = [data]

        for p, ext, d in zip(self.path, self._obj.ext, data):

            out_dir = os.path.dirname(p)
            os.makedirs(out_dir, exist_ok=True)

            if ext == 'csv':
                d.to_csv(p, index=index)
            elif ext == 'tif' or ext == 'stk' or ext == 'png' or ext == 'bmp':

                if compressed:

                    warnings.warn(
                        'compressed flag will be removed in future. Use compress=[0,9] instead.'
                    )

                    if d.ndim == 2:  # 2D image
                        d = d[None]

                    img = Image.fromarray(d[0]).save(
                        p,
                        compression="tiff_lzw",
                        save_all=True,
                        append_images=[
                            Image.fromarray(img_slice) for img_slice in d[1:]
                        ])
                else:
                    imsave(p, d, imagej=imagej, compress=compress)
            elif ext == 'npz':
                # save binary comrpessed numpy array
                out_dir = os.path.dirname(p)
                os.makedirs(out_dir, exist_ok=True)
                np.savez_compressed(p, arr=d)

            else:
                raise NotImplementedError(
                    'writing .{} files not implemented'.format(ext))

    @property
    def path(self):
        return self._obj.reset_index().apply(
            lambda x: x.pattern.format(**x.to_dict()), axis=1)

    @staticmethod
    def read_img(path):
        return imread(path)

    @classmethod
    def map_read_fct(cls, p, ext):
        if ext == 'tif' or ext == 'stk' or ext == 'png' or ext == 'bmp':
            return cls.read_img(p)
        elif ext == 'csv':
            return pd.read_csv(p)
        elif ext == 'npz':
            return np.load(p)['arr']
        else:
            raise NotImplementedError(
                'Reading .{} files not implemented'.format(ext))


def register_lsc_accessor():
    pd.api.extensions.register_series_accessor('lsc')(LSCAccessor)
    pd.api.extensions.register_dataframe_accessor('lsc')(LSCAccessor)
