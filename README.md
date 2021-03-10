# Improc

A collection of image processing routines.

## Installation

```bash
git clone https://github.com/fmi-basel/improc.git
pip install improc/
```

## Usage
This package consist of a collection of somewhat common functions that are not directly available in libraries such in scikit-image or opencv. That includes (non-exhaustive):

### Parsing file collections
The `io` sub-module provides a tool to parse a collection of files following the python string formatting syntax and saves the result in a pandas DataFrame. An accessor is also provided to expand pandas with read/write shortcuts for images and csv files.
see [parse_collection.ipynb](notebooks/parse_collection.ipynb) for more details.

### Cleaning up labels
`label` and `morphology` sub-modules contain routines to clean up segmentation labels (filter size, fill-holes, edge smoothing, etc.)

### Converting 3D anistotropic labels to smooth vtk meshes
see `mesh` sub-module

### Finding objects in 3D stacks from z, x and y projections
see `segmentation` sub-module

### Extracting features
The `regionprops` sub-module extends scikit-images' regionprop to multi-channel, multi-label data in a modular way.

### Resampling intensity images and labels to match a given voxel size
see `resample` sub-module

### Reading multi-scene czi files
The `czi_reader` sub-module provides draft utility functions to read multi-scene czi files. If working prefere using czifile or more actively developed aicspylibczi. (both attempt to create a giant image containing all the scenes and run out of memory)
