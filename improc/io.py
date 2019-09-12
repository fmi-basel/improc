import re
import parse
from glob import glob
from functools import lru_cache
from skimage.external.tifffile import imread, imsave
import pandas as pd
import numpy as np
import os

# TODO consider using xarray dataset --> build xarray + set xarray accessor instead df/series (directly usable with holoviews)
# TODO add logger


def parse_collection(pattern, keys):
    '''Parse a data collection and returns a dataframe of parsed attributes with keys as index.
    
    Pattern example:
    '/dir1/dir2/dir3/{channel}/T{time:04d}.{ext}'
    '''

    compiled_pattern = parse.compile(pattern)
    paths = sorted(glob(re.sub('{.*?}', '*', pattern)))

    # extract basedir from pattern
    if '{basedir}' in pattern:
        raise ValueError(
            'basedir key is reserved and cannot be used in pattern: {}'.format(
                pattern))
    basedir = os.path.dirname(pattern.split('{', 1)[0])
    pattern = re.sub(basedir, '{basedir}', pattern)

    # apply pattern to all paths that matched
    parsed_paths = [compiled_pattern.parse(p) for p in paths]

    # build df, exclude file that matched the glob pattern bu not the formatting ( returns None)
    df = (pd.DataFrame([{
        'pattern': pattern,
        'basedir': basedir,
        **pp.named
    } for pp in parsed_paths if pp]).set_index(keys).sort_index())

    if df.index.duplicated().sum():
        raise ValueError(
            'Selected keys {} do not form a unique index'.format(keys))

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
            if isinstance(key, (list, tuple, np.ndarray)) and len(key) != len(
                    self._obj.index.levels):
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

    def write(self, data):

        # if data is note a list, we assume it is because the is only one item
        if not isinstance(data, list):
            data = [data]

        for p, ext, d in zip(self.path, self._obj.ext, data):

            out_dir = os.path.dirname(p)
            os.makedirs(out_dir, exist_ok=True)

            if ext == 'csv':
                d.to_csv(p)
            elif ext == 'tif':
                imsave(p, d)
            else:
                raise NotImplementedError(
                    'writing .{} files not implemented'.format(ext))

    @property
    def path(self):
        return self._obj.reset_index().apply(lambda x: x.pattern.format(
            **x.to_dict()),
                                             axis=1)

    @staticmethod
    @lru_cache(maxsize=0)
    def read_img(path):
        return imread(path)

    @classmethod
    def map_read_fct(cls, p, ext):
        if ext == 'tif':
            return cls.read_img(p)
        elif ext == 'csv':
            return pd.read_csv(p)
        else:
            raise NotImplementedError(
                'Reading .{} files not implemented'.format(ext))

    @classmethod
    def set_img_cache_size(cls, cache_size):
        cls.read_img = staticmethod(
            lru_cache(maxsize=cache_size)(cls.read_img.__wrapped__))


def register_lsc_accessor():
    pd.api.extensions.register_series_accessor('lsc')(LSCAccessor)
    pd.api.extensions.register_dataframe_accessor('lsc')(LSCAccessor)
