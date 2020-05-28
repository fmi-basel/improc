import abc
import itertools
import inspect
import pandas as pd
import numpy as np
from functools import partial

from skimage.measure import regionprops_table, perimeter
from scipy.ndimage.measurements import labeled_comprehension
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import find_objects


class BaseFeatureExtractor():
    '''Base class for feature extractors. Extract features from all combinations 
    of label-channel in labels,channels input dicts. Optionaly target keys can be filtered.
    
    Returns the result a pandas dataframe.
    '''
    def __init__(self,
                 label_targets='all',
                 channel_targets='all',
                 *args,
                 **kwargs):
        '''
        Args:
            label_targets: list of keys to filter label images
            channel_targets: list of keys to filter channel images

        '''

        self.label_targets = label_targets
        self.channel_targets = channel_targets

    def __call__(self, labels, channels):

        if not (isinstance(labels, dict) or labels is None):
            raise ValueError(
                'Expects labels to be a dictionnary of images. received {}'.
                format(type(labels)))

        if not (isinstance(channels, dict) or channels is None):
            raise ValueError(
                'Expects channels to be a dictionnary of images. received {}'.
                format(type(channels)))

        # filter targets
        if self.label_targets is None:
            labels = None
        elif self.label_targets != 'all':
            labels = {t: labels[t] for t in self.label_targets}

        if self.channel_targets is None:
            channels = None
        elif self.channel_targets != 'all':
            channels = {t: channels[t] for t in self.channel_targets}

        if labels is None and channels is None:
            raise ValueError(
                'At least one label image or one intensity channel must be provided'
            )

        if labels is None:
            # measure image properties --> single label covering entire image
            labels = {
                'img': np.ones_like(next(iter(channels.values())),
                                    dtype=np.uint8)
            }

        if channels is None:
            channels = {'na': None}

        all_props = pd.DataFrame(columns=[
            'channel', 'region', 'object_id', 'feature_name', 'feature_value'
        ])

        # all combination of labels and channel
        for (label_key,
             label), (ch_key, ch) in itertools.product(labels.items(),
                                                       channels.items()):

            props = self._extract_features(label, ch)

            props = pd.DataFrame(props)
            props = props.set_index(
                'label').stack().reset_index()  #.set_index('label')
            props.columns = ['object_id', 'feature_name', 'feature_value']
            props['channel'] = ch_key
            props['region'] = label_key
            all_props = all_props.append(props, sort=False)

        all_props = all_props.apply(self._dataframe_hook, axis=1)

        return all_props

    def _dataframe_hook(self, row):
        '''Function applied to each row of the final dataframe'''

        return row

    @abc.abstractmethod
    def _extract_features(self, label, intensity):
        '''Method to extract feature for the given label,intensity_image pair.
        Is expected to return a dict with features as keys and 1 dimensional 
        arrays containing feature value for each label as values. The returned 
        dict should also contains the key "label" with label ids as values.
        
        example:
        {'label':[1,2,3],
         'area':[101,45,1000],
         'mean_intensity': [10,100,25]}
        
        '''

        pass


class QuantilesFeatureExtractor(BaseFeatureExtractor):
    '''Extract quantiles intensities over each labeled region'''
    def __init__(self, quantiles=[0., 0.25, 0.5, 0.75, 1.0], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.quantiles = quantiles

    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]

        q_vals = np.stack([
            np.quantile(intensity[labels == l], self.quantiles)
            for l in unique_l
        ],
                          axis=-1)

        props = {
            'q{:.3f}'.format(q): qv
            for q, qv in zip(self.quantiles, q_vals)
        }
        props['label'] = unique_l

        return props


class IntensityFeatureExtractor(BaseFeatureExtractor):
    '''Extract mean,std,mad (median absolute deviation) intensities 
    over each labeled region'''

    _features_functions = {
        'mean': np.mean,
        'std': np.std,
        'mad': lambda x: np.median(
            np.abs(x - np.median(x))
        ),  # median absolute deviation defined as median(|xi - median(x)|)
    }
    _implemented_features = set(_features_functions.keys())

    def __init__(self, features=['mean'], *args, **kwargs):
        super().__init__(*args, **kwargs)

        for f in set(features) - self._implemented_features:
            raise NotImplementedError('feature {} not implemented'.format(f))

        self.features = features

    def _extract_features(self, labels, intensity):

        unique_l = np.unique(labels)
        unique_l = unique_l[unique_l != 0]

        props = {
            feature_name:
            labeled_comprehension(intensity,
                                  labels,
                                  unique_l,
                                  self._features_functions[feature_name],
                                  out_dtype=float,
                                  default=np.nan)
            for feature_name in self.features
        }
        props['label'] = unique_l

        return props


class DistanceTransformFeatureExtractor(BaseFeatureExtractor):
    '''Extract features based on distance transform (mean|max|median radius) 
    over each labeled region'''

    _features_functions = {
        'mean_radius': np.mean,
        'max_radius': np.max,
        'median_radius': np.median,
    }

    _implemented_features = set(_features_functions.keys())
    _require_isotropic = {'mean_radius', 'median_radius'}

    def __init__(self,
                 features=['mean_radius', 'max_radius', 'median_radius'],
                 physical_coords=False,
                 spacing=1,
                 *args,
                 **kwargs):
        '''
        Args:
            features: list of features to compute
            physical_coords: whether to convert px coordinates to physical coordinates
            spacing: voxel size to do the coordinate conversion
        '''
        super().__init__(channel_targets=None, *args, **kwargs)

        for f in set(features) - self._implemented_features:
            raise NotImplementedError('feature {} not implemented'.format(f))

        if not isinstance(
                spacing,
            (int, float)) and not np.all(np.array(spacing) == spacing[0]):
            for f in self._require_isotropic.intersection(features):
                raise ValueError(
                    '{} feature requires isotropic spacing'.format(f))

        # add compulsory 'label' needed of indexing
        self.features = set(features)
        self.spacing = spacing
        self.physical_coords = physical_coords

    def _extract_features(self, labels, intensity):

        if self.physical_coords:
            self.ndim = labels.ndim
            self.spacing = np.broadcast_to(np.array(self.spacing), self.ndim)
            sampling = self.spacing
        else:
            sampling = None

        props = {f: [] for f in self.features}
        unique_l = []

        # compute distance transform separately for each label (in case they are touching)
        locs = find_objects(labels)
        for l, loc in enumerate(locs, start=1):
            if loc:
                unique_l.append(l)
                mask = np.pad(labels[loc] > 0, 1)
                dist = distance_transform_edt(mask, sampling=sampling)
                radii = dist[mask]

                for f in self.features:
                    props[f].append(self._features_functions[f](radii))

        props['label'] = unique_l

        return props


class SKRegionPropFeatureExtractor(BaseFeatureExtractor):
    '''scikit-image regionprops wrapper.
    
    Notes:
    for details see https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    Also compute the convex_perimeter.
    '''

    # TODO complete name mapping and coord conv

    # skimage uses 2D names for 3D features (e.g. area, perimeter, etc.)
    _name_mapping_2D_3D = {'area': 'volume'}
    _name_apping_3D_2D = {val: key for key, val in _name_mapping_2D_3D.items()}
    _implemented_features = {
        'label', 'volume', 'area', 'centroid', 'weighted_centroid',
        'minor_axis_length', 'major_axis_length', 'eccentricity', 'perimeter',
        'convex_area', 'convex_perimeter', 'solidity'
    }
    _require_isotropic = {
        'minor_axis_length', 'major_axis_length', 'perimeter',
        'convex_perimeter'
    }

    _physical_coords_conversion = {
        'volume': lambda x, spacing: x * np.prod(spacing),
        'perimeter': lambda x, spacing: x * spacing[0],
        'convex_perimeter': lambda x, spacing: x * spacing[0],
        'area': lambda x, spacing: x * np.prod(spacing),
        'convex_area': lambda x, spacing: x * np.prod(spacing),
        'centroid-0': lambda c, spacing: c * spacing[0],
        'centroid-1': lambda c, spacing: c * spacing[1],
        'centroid-2': lambda c, spacing: c * spacing[2],
        'weighted_centroid-0': lambda c, spacing: c * spacing[0],
        'weighted_centroid-1': lambda c, spacing: c * spacing[1],
        'weighted_centroid-2': lambda c, spacing: c * spacing[2],
        'minor_axis_length': lambda x, spacing: x * spacing[0],
        'major_axis_length': lambda x, spacing: x * spacing[0],
    }

    def __init__(self,
                 features=['centroid'],
                 physical_coords=False,
                 spacing=1,
                 *args,
                 **kwargs):
        '''
        Args:
            features: list of features to compute
            physical_coords: whether to convert px coordinates to physical coordinates
            spacing: voxel size to do the coordinate conversion
        '''
        super().__init__(*args, **kwargs)

        for f in set(features) - self._implemented_features:
            raise NotImplementedError('feature {} not implemented'.format(f))

        # add compulsory 'label' needed of indexing
        self.features = set(features).union({'label'})
        self.spacing = spacing
        self.physical_coords = physical_coords

    def _px_to_phy(self, row):

        if self.physical_coords:

            if not self.isotropic and row.feature_name in self._require_isotropic:
                raise ValueError(
                    '{} requires isotropic spacing. spacing: {}'.format(
                        row.feature_name, self.spacing))

            convert_fun = self._physical_coords_conversion.get(
                row.feature_name)

            if convert_fun is not None:
                row.feature_value = convert_fun(row.feature_value,
                                                self.spacing)

        return row

    def _dataframe_hook(self, row):
        '''Function applied to each row of the final dataframe'''

        row = self._px_to_phy(row)

        return row

    def _extract_features(self, labels, intensity):

        self.ndim = labels.ndim
        self.spacing = np.broadcast_to(np.array(self.spacing), self.ndim)
        self.isotropic = np.all(self.spacing == self.spacing[0])

        # map 2D feature names if 3D image
        if self.ndim == 3:
            # skimage regions props uses 2D feature names (e.g. perimeter, area instead of surface, volume respectively)
            features = {
                self._name_mapping_3D_2D.get(f, f)
                for f in self.features
            }
        else:
            features = self.features

        # special case pre: extract "convex_image" to compute missing "convex_perimeter" feature
        if 'convex_perimeter' in features:
            features = [
                'convex_image' if x == 'convex_perimeter' else x
                for x in features
            ]

        # extract actual features
        props = regionprops_table(labels,
                                  intensity_image=intensity,
                                  properties=features,
                                  separator='-')

        # special case post: extract compute missing "convex_perimeter" feature
        convex_images = props.pop('convex_image', None)
        if convex_images is not None:
            props['convex_perimeter'] = [
                perimeter(hull) for hull in convex_images
            ]

        # map back 3D feature names if 3D image
        if self.ndim == 3:
            props = {
                self._name_apping_2D_3D.get(key, key): val
                for key, val in props.items
            }

        return props


class DerivedFeatureCalculator():
    '''Compute derived features from a dataframe of existing features'''

    # TODO error/warning if required based feature not found
    #   how to handle aggregated props where only some groups have the reuired features?
    # TODO handle 3D derived features (name mapping?, area-->volume, etc)
    # TODO cell profiler compactness: derived from image moments, see centrosome/cpmorphology.py

    _implemented_features = {'mass_displacement', 'convexity', 'form_factor'}

    # morphologial base features that might not be associated with a channel
    # e.g. 1 "centroid" combined with multiple "weighted_centroid" to calculate mass displacement
    _morphologial_shared_base_features = ['centroid']
    _regex_shared_base_features = '|'.join(_morphologial_shared_base_features)

    def __init__(self, features=['form_factor']):

        for f in set(features) - self._implemented_features:
            raise NotImplementedError('feature {} not implemented'.format(f))

        self.features = features

    def __call__(self, props):
        props = props.set_index(['channel', 'region',
                                 'object_id']).sort_index()

        props = props.append(
            props.groupby(['channel', 'region', 'object_id']).apply(
                partial(self._compute_subdf_features, props=props)))

        return props.reset_index()

    def _compute_subdf_features(self, subdf, props):

        # add pure morphological features without assigned channel to subdf if available and needed
        idx = subdf.index[0]
        if idx[0] != 'na':

            try:
                # channel='na', same region and object_id as subdf
                shared_based_props = props.loc(axis=0)[('na', ) + idx[1:]]
                shared_based_props = shared_based_props[
                    shared_based_props.feature_name.str.contains(
                        self._regex_shared_base_features)]
                subdf = subdf.append(shared_based_props)
            except Exception as e:
                pass

        derived_features = []

        for feature in self.features:

            # get the function computing the requested features
            fun = getattr(self, feature)

            # get a list of required base features
            fun_args = inspect.getfullargspec(fun).args

            # get required base features' value
            kwargs = {
                arg: subdf[subdf.feature_name.str.startswith(
                    arg)].feature_value.values.squeeze()
                for arg in fun_args
            }

            try:
                feature_value = fun(**kwargs)
            except Exception as e:
                feature_value = None

            if feature_value or isinstance(feature_value, (float, int)):
                derived_features.append({
                    'feature_name': feature,
                    'feature_value': feature_value
                })

        return pd.DataFrame(derived_features)

    @staticmethod
    def mass_displacement(centroid, weighted_centroid):
        '''distance between the center of mass of the binary image and of the intensity image.'''

        return np.linalg.norm(centroid - weighted_centroid)

    @staticmethod
    def convexity(perimeter, convex_perimeter):
        '''convex hull perimeter / perimeter'''

        return convex_perimeter / perimeter

    @staticmethod
    def form_factor(area, perimeter):
        '''4*π*Area/Perimeter^2
        aka area / area of disc having same perimeter'''

        return 4 * np.pi * area / perimeter**2


class GlobalFeatureExtractor():
    '''Combines several feature extractors and calculators and apply them at once
    to a pair of dictionnaries containing labels and channel images'''
    def __init__(self, extractors, calculators=[]):
        self.extractors = extractors
        self.calculators = calculators

    def __call__(self, labels, channels):

        props = self.extractors[0](labels, channels)

        for extractor in self.extractors[1:]:
            props = props.append(extractor(labels, channels))

        for calculator in self.calculators:
            props = calculator(props)

        return props
