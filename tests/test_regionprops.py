import pytest
import numpy as np
import itertools
import pandas as pd

from improc.regionprops import BaseFeatureExtractor, SKRegionPropFeatureExtractor, QuantilesFeatureExtractor, IntensityFeatureExtractor, DistanceTransformFeatureExtractor, GlobalFeatureExtractor, DerivedFeatureCalculator, HybridDerivedFeatureCalculator, RegionDerivedFeatureCalculator

LABEL_IMAGE = np.zeros((100, 100), dtype=np.uint16)
LABEL_IMAGE[15:45, 15:45] = 1
LABEL_IMAGE[40:80, 50:90] = 2
LABEL_IMAGE[80:90, 10:30] = 3

INTENSITY_IMAGE = np.zeros((100, 100), dtype=np.uint16)
INTENSITY_IMAGE[15:45, 15:45] = 45
INTENSITY_IMAGE[40:80, 50:90] = 72
INTENSITY_IMAGE[80:90, 10:30] = 223

AREAS = [30 * 30, 40 * 40, 10 * 20]
MEANS = [45, 72, 223]

# yapf: disable
PROPS = pd.DataFrame(data=
        [['ch1','obj',1,'centroid-0',20.0],
         ['ch1','obj',1,'centroid-1',20.0],
         ['ch1','obj',1,'weighted_centroid-0',10.0],
         ['ch1','obj',1,'weighted_centroid-1',20.0],
         ['ch1','obj',1,'perimeter',160.0],
         ['ch1','obj',1,'area',1681.0],
         ['ch1','obj',2,'centroid-0',80.0],
         ['ch1','obj',2,'centroid-1',80.0],
         ['ch1','obj',2,'weighted_centroid-0',80.0],
         ['ch1','obj',2,'weighted_centroid-1',90.0],
         ['ch1','obj',2,'perimeter',156.0],
         ['ch1','obj',2,'area',1600.0],
         ['ch2','obj',1,'centroid-0',20.0],
         ['ch2','obj',1,'centroid-1',20.0],
         ['ch2','obj',1,'weighted_centroid-0',10.0],
         ['ch2','obj',1,'weighted_centroid-1',20.0],
         ['ch2','obj',1,'perimeter',160.0],
         ['ch2','obj',1,'area',1681.0],
         ['ch2','obj',2,'centroid-0',80.0],
         ['ch2','obj',2,'centroid-1',80.0],
         ['ch2','obj',2,'weighted_centroid-0',80.0],
         ['ch2','obj',2,'weighted_centroid-1',90.0],
         ['ch2','obj',2,'perimeter',156.0],
         ['ch2','obj',2,'area',1600.0],],
        columns=['channel','region','object_id','feature_name', 'feature_value'])

PROPS_SHARED_MORPH = pd.DataFrame(data=
        [['na','obj',1,'centroid-0',20.0],
         ['na','obj',1,'centroid-1',20.0],
         ['ch1','obj',1,'weighted_centroid-0',10.0],
         ['ch1','obj',1,'weighted_centroid-1',20.0],
         ['na','obj',1,'perimeter',160.0],
         ['na','obj',1,'area',1681.0],
         ['na','obj',2,'centroid-0',80.0],
         ['na','obj',2,'centroid-1',80.0],
         ['ch1','obj',2,'weighted_centroid-0',80.0],
         ['ch1','obj',2,'weighted_centroid-1',90.0],
         ['na','obj',2,'perimeter',156.0],
         ['na','obj',2,'area',1600.0],
         ['ch2','obj',1,'weighted_centroid-0',10.0],
         ['ch2','obj',1,'weighted_centroid-1',20.0],
         ['ch2','obj',2,'weighted_centroid-0',80.0],
         ['ch2','obj',2,'weighted_centroid-1',90.0],],
        columns=['channel','region','object_id','feature_name', 'feature_value'])
# yapf: enable


# yapf: disable
@pytest.mark.parametrize("label_targets,channel_targets, labels,channels,expected_region_names,expected_channel_names", [
                                             ('all', 'all', {'object':LABEL_IMAGE}, None, ['object'], ['na']),
                                             ('all', 'all', {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE}, None, ['objA', 'objB','objC'], ['na']),
                                             ('all', 'all', None,{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['img'], ['ch1', 'ch2', 'ch3']),
                                             ('all', 'all', {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['objA', 'objB','objC'], ['ch1', 'ch2', 'ch3']),
                                             (None, 'all', {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['img'], ['ch1', 'ch2', 'ch3']),
                                             ('all', None, {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['objA', 'objB','objC'], ['na']),
                                             (['objA', 'objB'], 'all', {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['objA', 'objB'], ['ch1', 'ch2', 'ch3']),
                                             ('all', ['ch2'], {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['objA', 'objB','objC'], ['ch2']),
                                             (['objC'], ['ch1','ch3'], {'objA':LABEL_IMAGE, 'objB':LABEL_IMAGE, 'objC':LABEL_IMAGE},{'ch1':INTENSITY_IMAGE, 'ch2':INTENSITY_IMAGE, 'ch3':INTENSITY_IMAGE}, ['objC'], ['ch1', 'ch3']),
                                        ])
# yapf: enable
def test_base_feature_extractor(label_targets, channel_targets, labels,
                                channels, expected_region_names,
                                expected_channel_names):
    '''
    Builds a simple feature extractor (area and mean) inherited from BaseFeatureExtractor.
    Checks that output dataframe contains all expected channel+label combinations.
    '''
    class TestFeatureExtractor(BaseFeatureExtractor):

        physical_coords_conversion = {
            'area': lambda x, spacing: x * np.prod(spacing),
        }

        def _extract_features(self, label, intensity):

            unique_l = np.unique(label)
            unique_l = unique_l[unique_l != 0]
            props = {'label': unique_l}
            if intensity is not None:
                props['mean'] = [
                    intensity[label == l].mean() for l in unique_l
                ]

            props['area'] = [(label == l).sum() for l in unique_l]
            return props

    extractor = TestFeatureExtractor(label_targets, channel_targets)
    props = extractor(labels, channels)
    props = props.set_index(['channel', 'region', 'feature_name', 'object_id'])

    expected_features = ['area', 'mean']
    if channels is None or channel_targets is None:
        expected_features = ['area']

    expected_object_ids = [1, 2, 3]
    if labels is None or label_targets is None:
        expected_object_ids = [1]

    for ch, reg, f, oid in itertools.product(expected_channel_names,
                                             expected_region_names,
                                             expected_features,
                                             expected_object_ids):

        # check that all props entries exist
        assert len(props.loc(axis=0)[ch, reg, f, oid]) == 1

        # check feature values
        if reg == 'img':
            if f == 'area':
                feature_val = props.loc(axis=0)[ch, reg, f, oid].values[0]
                np.testing.assert_almost_equal(feature_val, 100**2)
            elif f == 'mean':
                feature_val = props.loc(axis=0)[ch, reg, f, oid].values[0]
                np.testing.assert_almost_equal(feature_val,
                                               INTENSITY_IMAGE.mean())

        else:
            if f == 'area':
                feature_val = props.loc(axis=0)[ch, reg, f, oid].values[0]
                np.testing.assert_almost_equal(feature_val, AREAS[oid - 1])
            elif f == 'mean':
                feature_val = props.loc(axis=0)[ch, reg, f, oid].values[0]
                np.testing.assert_almost_equal(feature_val, MEANS[oid - 1])


# yapf: disable
@pytest.mark.parametrize("labels,channels,features,spacing,physical_coords,expected_rows", [
                                         ({'object':LABEL_IMAGE}, {'ch1':INTENSITY_IMAGE},['centroid'],2,False, {('centroid-0',1):29,
                                                                                                                 ('centroid-1',1):29,
                                                                                                                 ('centroid-0',2):59,
                                                                                                                 ('centroid-1',2):69,
                                                                                                                 ('centroid-0',3):84,
                                                                                                                 ('centroid-1',3):19,}),
                                         ({'object':LABEL_IMAGE}, {'ch1':INTENSITY_IMAGE},['centroid'],2,True,  {('centroid-0',1):29*2,
                                                                                                                 ('centroid-1',1):29*2,
                                                                                                                 ('centroid-0',2):59*2,
                                                                                                                 ('centroid-1',2):69*2,
                                                                                                                 ('centroid-0',3):84*2,
                                                                                                                 ('centroid-1',3):19*2,}),
                                         ({'object':LABEL_IMAGE}, None,['convex_perimeter'],1,False,  {('convex_perimeter',1):29*4,
                                                                                                       ('convex_perimeter',2):39*4,
                                                                                                       ('convex_perimeter',3):9*2+19*2,}),
                                    ])
# yapf: enable
def test_skregionprops_feature_extractor(labels, channels, features, spacing,
                                         physical_coords, expected_rows):
    '''tests feature extractor wrapper fo sk-image regionprops with/without physical coords conversion'''

    extractor = SKRegionPropFeatureExtractor(features, physical_coords,
                                             spacing)
    props = extractor(labels, channels)
    props = props.set_index(['feature_name', 'object_id'])

    for key, val in expected_rows.items():
        np.testing.assert_almost_equal(props.loc[key, 'feature_value'], val)


# yapf: disable
@pytest.mark.parametrize("labels,channels,quantiles,expected_rows", [
                                         ({'object':LABEL_IMAGE}, {'ch1':INTENSITY_IMAGE},[0.,0.25,0.5,0.75,1.0], {('q0_000',1):45,
                                                                                                                   ('q0_250',1):45,
                                                                                                                   ('q0_500',1):45,
                                                                                                                   ('q0_750',1):45,
                                                                                                                   ('q1_000',1):45,
                                                                                                                   ('q0_000',2):72,
                                                                                                                   ('q0_250',2):72,
                                                                                                                   ('q0_500',2):72,
                                                                                                                   ('q0_750',2):72,
                                                                                                                   ('q1_000',2):72,
                                                                                                                   ('q0_000',3):223,
                                                                                                                   ('q0_250',3):223,
                                                                                                                   ('q0_500',3):223,
                                                                                                                   ('q0_750',3):223,
                                                                                                                   ('q1_000',3):223,
                                                                                                                  }),
                                         (None, {'ch1':INTENSITY_IMAGE},[0.,0.5,1.0], {('q0_000',1):INTENSITY_IMAGE.min(),
                                                                                       ('q0_500',1):np.median(INTENSITY_IMAGE),
                                                                                       ('q1_000',1):INTENSITY_IMAGE.max(),
                                                                                      }),
                                    ])
# yapf: enable
def test_quantiles_feature_extractor(labels, channels, quantiles,
                                     expected_rows):
    '''tests quantiles feature extractor'''

    extractor = QuantilesFeatureExtractor(quantiles=quantiles)
    props = extractor(labels, channels)
    props = props.set_index(['feature_name', 'object_id'])

    for key, val in expected_rows.items():
        np.testing.assert_almost_equal(props.loc[key, 'feature_value'], val)


# yapf: disable
@pytest.mark.parametrize("labels,channels,features,expected_rows", [
                                         ({'object':LABEL_IMAGE}, {'ch1':INTENSITY_IMAGE},['mean'], {('mean',1):45,
                                                                                                     ('mean',2):72,
                                                                                                     ('mean',3):223,}),
                                         ({'object':LABEL_IMAGE}, {'ch1':INTENSITY_IMAGE},['std'], {('std',1):0,
                                                                                                     ('std',2):0,
                                                                                                     ('std',3):0,}),
                                         (None, {'ch1':np.arange(101)},['mad'], {('mad',1):25,}),
                                    ])
# yapf: enable
def test_intensity_feature_extractor(labels, channels, features,
                                     expected_rows):
    '''tests intensity feature extractor'''

    extractor = IntensityFeatureExtractor(features=features)
    props = extractor(labels, channels)
    props = props.set_index(['feature_name', 'object_id'])

    for key, val in expected_rows.items():
        np.testing.assert_almost_equal(props.loc[key, 'feature_value'], val)


def test_distance_transform_feature_extractor():
    '''tests extraction features based on distance transform'''

    labels = np.zeros((100, 100), dtype=np.uint8)
    labels[0:10, 0:10] = 1
    labels[0:10, 10:20] = 2

    extractor = DistanceTransformFeatureExtractor(
        features=['mean_radius', 'max_radius', 'median_radius'])
    props = extractor({'obj': labels}, None)
    props = props.set_index(['feature_name', 'object_id'])

    np.testing.assert_almost_equal(props.loc['mean_radius', 1].feature_value,
                                   2.2)
    np.testing.assert_almost_equal(props.loc['max_radius', 1].feature_value,
                                   5.)
    np.testing.assert_almost_equal(props.loc['median_radius', 1].feature_value,
                                   2)
    np.testing.assert_almost_equal(props.loc['mean_radius', 2].feature_value,
                                   2.2)
    np.testing.assert_almost_equal(props.loc['max_radius', 2].feature_value,
                                   5.)
    np.testing.assert_almost_equal(props.loc['median_radius', 2].feature_value,
                                   2)

    extractor = DistanceTransformFeatureExtractor(
        features=['mean_radius', 'max_radius', 'median_radius'],
        spacing=(2, 2),
        physical_coords=True)
    props = extractor({'obj': labels}, None)
    props = props.set_index(['feature_name', 'object_id'])

    np.testing.assert_almost_equal(props.loc['mean_radius', 1].feature_value,
                                   2.2 * 2)
    np.testing.assert_almost_equal(props.loc['max_radius', 1].feature_value,
                                   5. * 2)
    np.testing.assert_almost_equal(props.loc['median_radius', 1].feature_value,
                                   2 * 2)
    np.testing.assert_almost_equal(props.loc['mean_radius', 2].feature_value,
                                   2.2 * 2)
    np.testing.assert_almost_equal(props.loc['max_radius', 2].feature_value,
                                   5. * 2)
    np.testing.assert_almost_equal(props.loc['median_radius', 2].feature_value,
                                   2 * 2)


def test_derived_feature_mass_displacement():

    feature_calculator = HybridDerivedFeatureCalculator(['mass_displacement'])

    derived_props = feature_calculator(PROPS)
    derived_props = derived_props.set_index(
        ['feature_name', 'object_id', 'channel'])

    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 1, 'ch1'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 2, 'ch1'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 1, 'ch2'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 2, 'ch2'), 'feature_value'],
        10)

    derived_props = feature_calculator(PROPS_SHARED_MORPH)
    derived_props = derived_props.set_index(
        ['feature_name', 'object_id', 'channel'])

    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 1, 'ch1'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 2, 'ch1'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 1, 'ch2'), 'feature_value'],
        10)
    np.testing.assert_almost_equal(
        derived_props.loc[('mass_displacement', 2, 'ch2'), 'feature_value'],
        10)


def test_derived_feature_convexity():
    ''''''

    extractor = SKRegionPropFeatureExtractor(
        features=['perimeter', 'convex_perimeter'])
    feature_calculator = DerivedFeatureCalculator(['convexity'])

    labels = LABEL_IMAGE.copy()
    # remove a quarter of the square label --> concave
    labels[40:61, 50:71] = 0

    props = extractor({'object': labels}, None)
    derived_props = feature_calculator(props)
    derived_props = derived_props.set_index(['feature_name', 'object_id'])

    np.testing.assert_almost_equal(
        derived_props.loc[('convexity', 1), 'feature_value'], 1)

    np.testing.assert_almost_equal(derived_props.loc[('convexity',
                                                      2), 'feature_value'],
                                   (156 - 40 + np.sqrt(400 + 400)) / 156,
                                   decimal=3)

    np.testing.assert_almost_equal(
        derived_props.loc[('convexity', 3), 'feature_value'], 1)


def test_derived_feature_nuclei_fraction():
    ''''''
    props = pd.DataFrame(data=[['na', 'cell', 1, 'volume', 20.0],
                               ['na', 'nuclei', 1, 'volume', 10.0],
                               ['na', 'cell', 2, 'volume', 20.0],
                               ['na', 'nuclei', 2, 'volume', 5.0]],
                         columns=[
                             'channel', 'region', 'object_id', 'feature_name',
                             'feature_value'
                         ])

    feature_calculator = RegionDerivedFeatureCalculator(['nuclei_fraction'])

    derived_props = feature_calculator(props)
    derived_props = derived_props.set_index(
        ['feature_name', 'object_id', 'channel'])

    np.testing.assert_almost_equal(
        derived_props.loc[('nuclei_fraction', 1, 'na'), 'feature_value'], 0.5)
    np.testing.assert_almost_equal(
        derived_props.loc[('nuclei_fraction', 2, 'na'), 'feature_value'], 0.25)


def test_derived_feature_form_factor():
    ''''''

    labels = np.zeros((101, 101), dtype=np.uint8)
    labels[0:40, 0:40] = 1

    extractor = SKRegionPropFeatureExtractor(features=['area', 'perimeter'])
    feature_calculator = DerivedFeatureCalculator(['form_factor'])

    props = extractor({'object': labels}, None)
    derived_props = feature_calculator(props)
    derived_props = derived_props.set_index(['feature_name', 'object_id'])

    np.testing.assert_almost_equal(
        derived_props.loc[('form_factor', 1), 'feature_value'],
        4 * np.pi * 1600 / (4 * 39)**2)


def test_global_feature_extractor():
    '''Test that global feature extractor runs and returns the expected number of rows'''

    labels = {'objA': LABEL_IMAGE, 'objB': LABEL_IMAGE, 'objC': LABEL_IMAGE}
    channels = {
        'ch1': INTENSITY_IMAGE,
        'ch2': INTENSITY_IMAGE,
        'ch3': INTENSITY_IMAGE
    }

    extractor = GlobalFeatureExtractor(
        [
            SKRegionPropFeatureExtractor(
                features=['area', 'perimeter', 'centroid', 'convex_perimeter'],
                channel_targets=None),
            SKRegionPropFeatureExtractor(features=['weighted_centroid'])
        ],
        calculators=[
            HybridDerivedFeatureCalculator(['mass_displacement']),
            DerivedFeatureCalculator(['convexity'])
        ])

    props = extractor(labels, channels)

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(props)
    print(len(props))
    print((5 * 3 * 3 + 2 * 3 * 3 * 3 + 1 * 3 * 3 * 3 + 1 * 3 * 3))

    n_morph_features = 5 * 3 * 3
    n_wg_centroid_features = 2 * 3 * 3 * 3
    n_mass_disp = 3 * 3 * 3
    n_convexity = 3 * 3
    expected_n_features = n_morph_features + n_wg_centroid_features + n_mass_disp + n_convexity

    assert len(props) == expected_n_features
