import numpy as np
from numpy import testing

def test_random_coords_shape():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    shape = (5, 8, 3)
    coords = get_random_coords(shape, radius)
    assert coords.shape == shape


def test_random_coords_sphere():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    shape = (5, 8, 3)
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

def test_calc_restraints():
    from nuc_dynamics.nuc_cython import Restraint
    from nuc_dynamics import flatten_dict, calc_restraints

    seq_pos = {'a': np.arange(10, 111, 10, dtype='int32'),
               'b': np.arange(40, 221, 20, dtype='int32')}
    contacts = {'a': {'a': np.array([[ 21,  80, 0],
                                     [ 20,  81, 1],
                                     [  2, 109, 1],
                                     [ 19,  81, 2],
                                     [  3, 108, 2],], dtype='int').T,
                      'b': np.array([[ 11, 120, 3],
                                     [ 40,  60, 4]], dtype='int').T},
                'b': {'b': np.array([[ 50,  50, 5]], dtype='int').T}}

    expected = {'a': {'a': np.array([([ 0, 10], [0.8, 1.2], 1, 2.0),
                                     ([ 1,  8], [0.8, 1.2], 1, 2.0),
                                     ([ 2,  7], [0.8, 1.2], 3, 1.0),], dtype=Restraint),
                      'b': np.array([([ 1,  4], [0.8, 1.2], 2, 1.0),
                                     ([ 3,  1], [0.8, 1.2], 4, 1.0)], dtype=Restraint)},
                'b': {'b': np.array([([ 1,  1], [0.8, 1.2], 5, 1.0)], dtype=Restraint)}}

    testing.assert_equal(calc_restraints(contacts, seq_pos), expected)

def test_get_interpolated_coords():
    from nuc_dynamics import get_interpolated_coords

    prev_seq_pos = np.arange(5, 15, 2, dtype='int32')
    seq_pos = np.arange(5, 15, 1, dtype='int32')
    coords = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0],
                       [4.0, 6.0, 6.0], [3.0, 7.0, 9.5],
                       [5.5, 8.0, 6.0]], dtype='float64')
    expected = np.array([[ 1.0, 2.0, 3.0], [ 1.5 , 3.0, 4.5 ],
                         [ 2.0, 4.0, 6.0], [ 3.0 , 5.0, 6.0 ],
                         [ 4.0, 6.0, 6.0], [ 3.5 , 6.5, 7.75],
                         [ 3.0, 7.0, 9.5], [ 4.25, 7.5, 7.75],
                         [ 5.5, 8.0, 6.0], [ 5.5 , 8.0, 6.0 ],])

    result = get_interpolated_coords(coords, seq_pos, prev_seq_pos)
    np.testing.assert_array_equal(result, expected)

def test_get_interpolated_coords_edge():
    from nuc_dynamics import get_interpolated_coords

    prev_seq_pos = np.arange(5, 15, 2, dtype='int32')
    seq_pos = np.arange(8, 18, 1, dtype='int32')
    coords = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0],
                       [4.0, 6.0, 6.0], [3.0, 7.0, 9.5],
                       [5.5, 8.0, 6.0]], dtype='float64')

    expected = np.array([[ 3.0 , 5.0, 6.0 ], [ 4.0, 6.0, 6.0],
                         [ 3.5 , 6.5, 7.75], [ 3.0, 7.0, 9.5],
                         [ 4.25, 7.5, 7.75], [ 5.5, 8.0, 6.0],
                         [ 5.5 , 8.0, 6.0 ], [ 5.5, 8.0, 6.0],
                         [ 5.5 , 8.0, 6.0 ], [ 5.5, 8.0, 6.0],])


    result = get_interpolated_coords(coords, seq_pos, prev_seq_pos)
    np.testing.assert_array_equal(result, expected)


def test_calc_limits():
    from nuc_dynamics import calc_limits

    contacts = {'a': {'a': np.array([[10, 50, 0],
                                     [45, 66, 0],]).T,
                      'b': np.array([[ 5, 29, 0],
                                     [86,  4, 0],]).T,},
                'b': {'a': np.array([[10,  9, 0],
                                     [ 6, 49, 0],]).T}}


    expected = {'a': (5, 86), 'b': (4, 29)}
    np.testing.assert_equal(calc_limits(contacts), expected)
