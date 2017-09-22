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

def test_unflatten_dict():
    from nuc_dynamics import unflatten_dict

    expected = {0: {1: {2: 'foo', 3: 'foo2'}, 'bar': 'baz'}, 5: 'foo'}
    data = {(0, 1, 2): 'foo', (0, 1, 3): 'foo2', (0, 'bar'): 'baz', (5,): 'foo'}

    assert expected == unflatten_dict(data)

def test_concatenate_restraints():
    from nuc_dynamics import concatenate_restraints, Restraint

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
    expected = np.array([
        (( 4,  5), (0.3, 1.3), 0, 1.3,),
        (( 3,  2), (0.4, 1.4), 1, 1.4,),
        (( 4, 15), (0.1, 1.1), 0, 1.1,),
        (( 8, 25), (0.2, 1.2), 0, 1.2,),
        ((22, 26), (0.5, 1.5), 0, 1.5,),
        ((13, 12), (0.6, 1.6), 1, 1.6,),
    ], dtype=Restraint)
    np.testing.assert_array_equal(expected, concatenate_restraints(restraints, seq_pos))

def test_calc_restraints():
    from nuc_dynamics import flatten_dict, calc_restraints, Restraint, Contact

    seq_pos = {'a': np.arange(10, 111, 10, dtype='int32'),
               'b': np.arange(40, 221, 20, dtype='int32')}
    contacts = {'a': {'a': np.array([((21,  80), 0),
                                     ((20,  81), 1),
                                     (( 2, 109), 1),
                                     ((19,  81), 2),
                                     (( 3, 108), 2),], dtype=Contact),
                      'b': np.array([((11, 120),-3),
                                     ((40,  60), 4)], dtype=Contact)},
                'b': {'b': np.array([((50,  50), 5)], dtype=Contact)}}

    expected = {'a': {'a': np.array([([ 2,  7], [0.8, 1.2], 0, 1.0),
                                     ([ 1,  8], [0.8, 1.2], 1, 1.0),
                                     ([ 0, 10], [0.8, 1.2], 1, 1.0),
                                     ([ 1,  8], [0.8, 1.2], 2, 1.0),
                                     ([ 0, 10], [0.8, 1.2], 2, 1.0),], dtype=Restraint),
                      'b': np.array([([ 1,  4], [0.8, 1.2],-3, 1.0),
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
    result = get_interpolated_coords(
        np.stack([coords, coords + 1]), seq_pos, prev_seq_pos
    )
    expected = np.stack([expected, expected + 1])
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
    from nuc_dynamics import calc_limits, Contact

    contacts = {'a': {'a': np.array([((10, 50), 0),
                                     ((45, 66), 0),], dtype=Contact),
                      'b': np.array([(( 5, 29), 0),
                                     ((86,  4), 0),], dtype=Contact),},
                'b': {'a': np.array([((10,  9), 0),
                                     (( 6, 49), 0),], dtype=Contact)}}


    expected = {'a': (5, 86), 'b': (4, 29)}
    np.testing.assert_equal(calc_limits(contacts), expected)


def test_between():
    from nuc_dynamics import between

    values = np.array([6, 4, 1, 8, 5])
    expected = np.array([True, True, False, False, True])
    np.testing.assert_equal(between(values, 2, 7), expected)


def test_remove_isolated_contacts():
    from nuc_dynamics import remove_isolated_contacts, Contact

    contact_dict = {'1': {'2': np.array([
        ((500, 800), 0), ((505, 795), 0), ((809, 509), 0),
        ((900, 200), 0), ((500, 900), 0), ((901, 201), 0),
        ((501, 801), 0),
    ], dtype=Contact)}}

    expected = {'1': {'2': np.array([
        ((500, 800), 0), ((505, 795), 0), ((809, 509), 0), ((501, 801), 0),
    ], dtype=Contact)}}
    np.testing.assert_equal(remove_isolated_contacts(contact_dict, 10, 2), expected)


def test_remove_isolated_contacts_ignore():
    # Useful for e.g. image contacts
    from nuc_dynamics import remove_isolated_contacts, Contact

    contact_dict = {'1': {'2': np.array([
            ((500, 800), 0), ((600, 200), 0),
    ], dtype=Contact)}}

    expected = {'1': {'2': np.array([
            ((500, 800), 0), ((600, 200), 0),
    ], dtype=Contact)}}
    np.testing.assert_equal(remove_isolated_contacts(contact_dict, 10, 2, {"2"}), expected)
    np.testing.assert_equal(remove_isolated_contacts(contact_dict, 10, 2, {"1"}), expected)


def test_remove_violated_contacts():
    from nuc_dynamics import remove_violated_contacts, Contact

    pos_dict = {'1': np.array([4, 6, 10]), '2': np.array([3, 4, 5, 6])}
    coords_dict = {
        '1': np.array([
            [[1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0],
             [3.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [2.0, 1.0, 0.0],
             [3.0, 1.0, 0.0]],
        ]),
        '2': np.array([
            [[1.0, 0.0, 2.0],
             [2.0, 0.0, 0.0],
             [3.0, 0.0, 0.0],
             [4.0, 0.0, 10.0]],
            [[1.0, 1.0, 0.0],
             [2.0, 1.0, 0.0],
             [3.0, 1.0, 0.0],
             [4.0, 1.0, 10.0]],
        ]),
    }

    contact_dict = {'1': {'2': np.array([
        ((4, 3), 0), ((6, 6), 0), ((5, 3.5), 0),
    ], dtype=Contact)}}

    expected = {'1': {'2': np.array([
        ((4, 3), 0), ((5, 3.5), 0),
    ], dtype=Contact)}}

    result = remove_violated_contacts(contact_dict, coords_dict, pos_dict)
    np.testing.assert_equal(result, expected)


def test_merge_restraints():
    from nuc_dynamics import merge_dicts

    # Not correct dtype, but shouldn't matter
    a = {'foo': {'bar': [0], 'foo': [1, 5]}}
    b = {'foo': {'bar': [9, 10]}, 'baz': {'bar': [6]}}

    merged = {'foo': {'bar': [0, 9, 10], 'foo': [1, 5]}, 'baz': {'bar': [6]}}
    testing.assert_equal(merge_dicts(a, b), merged)


def test_bin_restraints():
    from nuc_dynamics import bin_restraints, Restraint

    restraints = np.array([
        ((4, 5), (0.8, 1.1), 1, 1.3,),
        ((5, 4), (0.8, 1.1), 1, 1.0,),
        ((3, 2), (0.8, 1.1), 2, 1.0,),
        ((4, 5), (0.8, 1.1), 1, 0.1,),
        ((3, 5), (0.8, 1.1), 2, 1.0,),
        ((2, 6), (0.8, 1.1), 0, 1.0,),
        ((3, 2), (0.8, 1.1), 1, 1.0,),
        ((3, 2), (0.8, 1.2), 1, 1.0,),
        ((3, 2), (0.8, 1.1), 0, 1.0,),
        ((3, 2), (0.8, 1.1), 0, 1.0,),
        ((2, 3), (0.8, 1.1), 4, 1.0,),
    ], dtype=Restraint)

    expected = np.array([
        ((2, 3), (0.8, 1.1), 0, 3.0,),
        ((2, 6), (0.8, 1.1), 0, 1.0,),
        ((2, 3), (0.8, 1.2), 1, 1.0,),
        ((2, 3), (0.8, 1.1), 1, 1.0,),
        ((4, 5), (0.8, 1.1), 1, 2.4,),
        ((2, 3), (0.8, 1.1), 2, 1.0,),
        ((3, 5), (0.8, 1.1), 2, 1.0,),
    ], dtype=Restraint)
    binned = bin_restraints(restraints)
    for name in Restraint.names:
        np.testing.assert_allclose(binned[name], expected[name])


def test_merge_contacts():
    from nuc_dynamics import merge_dicts, Contact

    contacts = [
        {'a': {'a': np.array([((10, 50), 0),
                              ((45, 66), 1),], dtype=Contact),
               'b': np.array([(( 5, 29), 0),
                              ((86,  4), 0),], dtype=Contact),},
         'b': {'a': np.array([((10,  9), 4),
                              (( 6, 49), 0),], dtype=Contact)}},
        {'a': {'a': np.array([((11, 50), 0),
                              ((44, 67), 1),], dtype=Contact),
               'c': np.array([(( 5, 29), 0),
                              ((86,  8), 0),], dtype=Contact),},
         'b': {'a': np.array([((10,  9), 4),
                              (( 4, 49), 1),], dtype=Contact)}},
    ]

    expected = {
        'a': {'a': np.array([((10, 50), 0),
                             ((45, 66), 1),
                             ((11, 50), 0),
                             ((44, 67), 1),], dtype=Contact),
              'b': np.array([(( 5, 29), 0),
                             ((86,  4), 0),], dtype=Contact),
              'c': np.array([(( 5, 29), 0),
                             ((86,  8), 0),], dtype=Contact),},
        'b': {'a': np.array([((10,  9), 4),
                             (( 6, 49), 0),
                             ((10,  9), 4),
                             (( 4, 49), 1),], dtype=Contact)},
    }
    np.testing.assert_equal(merge_dicts(*contacts), expected)


def test_concatenate_into():
    from nuc_dynamics import concatenate_into

    data = [np.arange(10).reshape(2, 5), np.arange(20).reshape(4, 5)]
    expected = np.concatenate(data)

    out = np.empty_like(expected)
    concatenate_into(data, out)

    testing.assert_array_equal(expected, out)


def test_round_up():
    from nuc_dynamics import roundUp

    assert roundUp(3, 5) == 5
    assert roundUp(3, 3) == 3
    assert roundUp(3, 2) == 4
