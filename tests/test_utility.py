import numpy as np
from numpy import testing

def test_random_coords_shape():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    shape = (5, 8)
    coords = get_random_coords(shape, radius)
    assert coords.shape == shape + (3,)


def test_random_coords_sphere():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    shape = (5, 8)
    coords = get_random_coords(shape, radius)

    assert np.alltrue(np.linalg.norm(coords, axis=-1) <= radius)
    assert not np.alltrue(np.linalg.norm(coords, axis=-1) <= radius * 0.5)


def test_ambiguity_strides():
    from nuc_cython import calc_ambiguity_strides

    groups = np.array([0, 0, 1, 2, 3, 3, 0, 4, 4, 5], dtype='int32')
    strides = np.array([1, 1, 1, 1, 2, 2, 1, 2, 2, 1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_strides(groups), strides)

    groups = np.array([1, 1, 1, 2, 3, 3, 0, 4, 4, 5], dtype='int32')
    strides = np.array([3, 3, 3, 1, 2, 2, 1, 2, 2, 1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_strides(groups), strides)


def test_ambiguity_strides_short():
    from nuc_cython import calc_ambiguity_strides

    groups = np.array([1], dtype='int32')
    strides = np.array([1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_strides(groups), strides)

    groups = np.array([0], dtype='int32')
    strides = np.array([1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_strides(groups), strides)

    groups = np.array([], dtype='int32')
    strides = np.array([], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_strides(groups), strides)
