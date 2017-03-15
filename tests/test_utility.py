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

def test_flatten_dict():
    from nuc_dynamics import flatten_dict

    data = {0: {1: {2: 'foo', 3: 'foo2'}, 'bar': 'baz'}, 5: 'foo'}
    expected = {(0, 1, 2): 'foo', (0, 1, 3): 'foo2', (0, 'bar'): 'baz', (5,): 'foo'}

    assert expected == flatten_dict(data)

def test_concatenate_restraints():
    from nuc_dynamics.nuc_cython import Restraint
    from nuc_dynamics import concatenate_restraints

    seq_pos = {'a': np.arange(10), 'b': np.arange(20)}
    restraints = {'a': {'a': np.array([([ 4,  5], [0.3, 1.3], 0, 1.3),
                                       ([ 3,  2], [0.4, 1.4], 1, 1.4),],
                                      dtype=Restraint),
                        'b': np.array([([ 4,  5], [0.1, 1.1], 0, 1.1),
                                       ([ 8, 15], [0.2, 1.2], 0, 1.2),],
                                      dtype=Restraint)},
                  'b': {'b': np.array([([12, 16], [0.5, 1.5], 0, 1.5),
                                       ([ 3,  2], [0.6, 1.6], 1, 1.6),],
                                      dtype=Restraint)}
    }
    expected = (np.array([[4, 5], [3, 2], [4, 15], [8, 25], [22, 26], [13, 12]]),
                np.array([[0.3, 1.3], [0.4, 1.4], [0.1, 1.1],
                          [0.2, 1.2], [0.5, 1.5], [0.6, 1.6]]),
                np.array([1.3, 1.4, 1.1, 1.2, 1.5, 1.6]),
                np.array([0, 1, 0, 0, 0, 1]),
    )
    for a, b in zip(expected, concatenate_restraints(restraints, seq_pos)):
        np.testing.assert_array_equal(a, b)
