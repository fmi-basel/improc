import czifile as cz
import numpy as np
import os
import pandas as pd
'''Draft utility functions to read multi-scene czi files. If working prefere using
 czifile or more actively developed aicspylibczi.
 (both attempt to create a giant image containing all the scenes...)'''


def read_czi_tiles(path):
    '''Generator returning tiles czi image'''

    czimg = cz.CziFile(path)
    n_tile = czimg.metadata(
        raw=False)['ImageDocument']['Metadata']['Information']['Image'].get(
            'SizeM', 1)

    # looks like min overlap is 1 px
    # TODO figure out how to get tile directly from czifile or account for overlap
    frame_width = czimg.metadata(
        raw=False)['ImageDocument']['Metadata']['Information']['Image'].get(
            'SizeX', 1) - 1
    frame_height = czimg.metadata(
        raw=False)['ImageDocument']['Metadata']['Information']['Image'].get(
            'SizeY', 1) - 1

    img = czimg.asarray()  #cz.imread(path)
    czimg.close()
    img = img.squeeze()
    img = np.moveaxis(img, 0, -1)  # channel last

    slices, width, height, channels = img.shape
    for tile_id in range(n_tile):
        tx = tile_id // round(width / frame_width)
        ty = tile_id % round(height / frame_height)

        yield img[:, tx * frame_width:(tx + 1) * frame_width, ty *
                  frame_height:(ty + 1) * frame_height]


################################################
# opening selected blocks from czi file
# TODO rewrite as a class (inherited from CziFile?)
# use dict and remove pandas dependency?


def parse_blocks(czimg):
    summary = []
    lut = cz.czifile.DIMENSIONS

    for s in czimg.filtered_subblock_directory:
        params = {}
        #         print(s.start)
        for dim_entry in s.dimension_entries:
            params[lut[dim_entry.dimension] + '_start'] = dim_entry.start
            params[lut[dim_entry.dimension] + '_size'] = dim_entry.size
            params[lut[dim_entry.dimension] +
                   '_start_coordinate'] = dim_entry.start_coordinate

        params['subblock_handle'] = s

        # check if it is a tile at the finest resolution or lower resolution part of a pyramid
        params['is_max_resolution'] = s.shape == s.stored_shape
        summary.append(params)

    df = pd.DataFrame(summary)

    # convert stage coordinates to image coordinates starting at zero
    for s in df.Scene_start.unique():
        df.loc[df.Scene_start ==
               s, 'Width_start'] -= df.loc[df.Scene_start ==
                                           s, 'Width_start'].min()
        df.loc[df.Scene_start ==
               s, 'Height_start'] -= df.loc[df.Scene_start ==
                                            s, 'Height_start'].min()

    # add default value if not present
    if 'Slice_start' not in df.columns:
        df['Slice_start'] = 0

    if 'Time_start' not in df.columns:
        df['Time_start'] = 0

    if 'Channel_start' not in df.columns:
        df['Channel_start'] = 0

    if 'Scene_start' not in df.columns:
        df['Scene_start'] = 0

    return df


def read_blocks(blocks):
    '''Returns a generator over image blocks'''

    for row in blocks.itertuples():
        yield {
            'block_img': row.subblock_handle.data_segment().data().squeeze(),
            'x_range': slice(row.Height_start,
                             row.Height_start + row.Height_size),
            'y_range': slice(row.Width_start,
                             row.Width_start + row.Width_size),
            'slice': row.Slice_start,
            'channel': row.Channel_start
        }


def stitched_shape(blocks):

    block_height_stop = blocks['Height_start'] + blocks['Height_size']
    block_width_stop = blocks['Width_start'] + blocks['Width_size']

    return (
        blocks['Slice_start'].max() + 1,  # slices
        block_height_stop.max(),  #height
        block_width_stop.max(),  # width
        blocks['Channel_start'].max() + 1)  #channels


def apply_shading(img, shading_map):
    return (img / shading_map).astype(img.dtype)


def stitch_scene(blocks, shading_map=None):
    '''basic stitching of tiles after applying shading correction'''

    blocks = blocks[blocks.is_max_resolution]

    img = None
    shape = stitched_shape(blocks)

    for b in read_blocks(blocks):
        if shading_map is not None:
            b['block_img'] = apply_shading(b['block_img'], shading_map)

        # check if 3 channels from color camera (counted as one channel in zen...)
        if b['block_img'].shape[-1] == 3:
            b['channel'] = slice(0, 3)
            b['block_img'] = b['block_img']  #[..., ::-1]  # BGR to RGB
            shape = shape[:-1] + (3, )

        # for the first time, use first block info to initialize stitched img
        if img is None:
            img = np.zeros(shape, b['block_img'].dtype)

        img[b['slice'], b['x_range'], b['y_range'],
            b['channel']] = b['block_img']

    return img.squeeze()


def stitch_selected_blocks(blocks, scene_id=0, time_id=0, shading_map=None):
    # TODO add filter by position, tile, channel etc.

    blocks = blocks[(blocks['Scene_start'] == scene_id)
                    & (blocks['Time_start'] == time_id)]

    return stitch_scene(blocks, shading_map=shading_map)
