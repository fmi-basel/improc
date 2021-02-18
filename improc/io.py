import re
import parse
from glob import glob
from skimage.io import imread, imsave
from PIL import Image
import pandas as pd
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable

# TODO add logger
# TODO add data collection:=dc alias


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
    pattern = pattern.replace(basedir + os.sep, '{basedir}' + os.sep)

    # apply pattern to all paths that matched
    parsed_paths = [compiled_pattern.parse(p) for p in paths]

    # build df, exclude file that matched the glob pattern but not the formatting ( returns None)
    parsed_paths = list(filter(None, parsed_paths))
    if len(parsed_paths) <= 0:
        pattern_expanded_basedir = pattern.replace('{basedir}', basedir)
        raise ValueError(
            'No files were found that match the pattern: {}'.format(
                pattern_expanded_basedir))

    df = (pd.DataFrame([{
        'pattern': pattern,
        'basedir': basedir,
        **pp.named
    } for pp in parsed_paths]).set_index(keys).sort_index())

    duplicated_index_mask = df.index.duplicated()
    if duplicated_index_mask.sum():

        duplicated_index = df[duplicated_index_mask].index

        example = '\n'
        for p in df.loc(axis=0)[duplicated_index[0]].dc.path:
            example += p + '\n'

        raise ValueError(
            '\nParsing error:\n\n{}\nSelected keys {} do not form a unique index. e.g. \n{}'
            .format(df.pattern.iloc[0], keys, example))

    return df


# ~ @pd.api.extensions.register_series_accessor('lsc')
class DCAccessor:
    '''Adds cached read/write accessors to pandas series representing lightsheet items.'''
    def __init__(self, pandas_obj):
        if isinstance(pandas_obj, pd.Series):
            self._obj = pandas_obj.to_frame().T
        else:
            self._obj = pandas_obj

        # TODO check that _obj a pattern and necessary keys

    def __getitem__(self, key):
        '''Returns a dataframe of requested index with all levels maintained, even if it has only one row'''

        if isinstance(key, tuple):
            key = tuple(k if isinstance(k, (
                Iterable, slice)) and not isinstance(k, str) else [k]
                        for k in key)
        elif not (isinstance(key,
                             (Iterable, slice)) and not isinstance(key, str)):
            key = [key]

        return self._obj.loc(axis=0)[key]

    @classmethod
    def register(cls):
        pd.api.extensions.register_series_accessor('dc')(cls)
        pd.api.extensions.register_dataframe_accessor('dc')(cls)

    def read(self):

        with ThreadPoolExecutor() as threads:
            return list(
                threads.map(self.map_read_fct, self.path, self._obj.ext))

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
            elif ext in ['tif', 'stk', 'png', 'bmp', 'jpg']:

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
                    if imagej:
                        imsave(p,
                               d,
                               imagej=imagej,
                               compress=compress,
                               check_contrast=False)
                    else:
                        # NOTE:
                        # passing imagej flag (even False) probably force a different backend
                        # can throw "data too large for standard TIFF file" error
                        imsave(p, d, compress=compress, check_contrast=False)

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
        if ext in ['tif', 'stk', 'png', 'bmp', 'jpg']:
            return cls.read_img(p)
        elif ext == 'csv':
            return pd.read_csv(p)
        elif ext == 'npz':
            return np.load(p)['arr']
        else:
            raise NotImplementedError(
                'Reading .{} files not implemented'.format(ext))


# lightsheet collection (LSC) accessor alias for backward compatiblity
class LSCAccessor(DCAccessor):
    @classmethod
    def register(cls):
        pd.api.extensions.register_series_accessor('lsc')(cls)
        pd.api.extensions.register_dataframe_accessor('lsc')(cls)
