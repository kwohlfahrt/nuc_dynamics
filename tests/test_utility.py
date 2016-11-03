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


def test_ambiguity_offsets():
    from nuc_dynamics import calc_ambiguity_offsets

    groups = np.array([0, 0, 1, 2, 3, 3, 0, 4, 4, 5], dtype='int32')
    offsets = np.array([0, 1, 2, 3, 4, 6, 7, 9, 10], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_offsets(groups), offsets)

    groups = np.array([1, 1, 1, 2, 3, 3, 0, 4, 4, 5], dtype='int32')
    offsets = np.array([0, 3, 4, 6, 7, 9, 10], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_offsets(groups), offsets)


def test_ambiguity_offsets_short():
    from nuc_dynamics import calc_ambiguity_offsets

    groups = np.array([1], dtype='int32')
    offsets = np.array([0, 1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_offsets(groups), offsets)

    groups = np.array([0], dtype='int32')
    offsets = np.array([0, 1], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_offsets(groups), offsets)

    groups = np.array([], dtype='int32')
    offsets = np.array([0], dtype='int32')
    testing.assert_array_equal(calc_ambiguity_offsets(groups), offsets)
