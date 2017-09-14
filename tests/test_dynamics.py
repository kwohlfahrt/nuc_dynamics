import numpy as np

def test_get_temp():
    from nuc_dynamics.nuc_cython import getTemp

    np.random.seed(4)
    masses = np.random.uniform(0.1, 1.0, size=20)
    veloc = np.random.uniform(-1.0, 1.0, size=(20, 3))
    values = getTemp(masses, veloc, len(masses))
    expected = 76.49978143407098
    assert values == expected


def test_get_stats():
    from nuc_dynamics.nuc_cython import getStats

    np.random.seed(4)
    coords = np.random.uniform(-1.0, 1.0, size=(20, 3))
    indices = np.random.choice(len(coords), (40, 2)).astype('int32')
    limits = np.stack([np.random.uniform(0.1, 0.5, size=len(indices)),
                       np.random.uniform(0.6, 1.2, size=len(indices))], axis=-1)

    values = getStats(indices, limits, coords, len(indices))
    assert values == (31, 0.721443286736486)


def test_restraint_force():
    from nuc_dynamics.nuc_cython import getRestraintForce

    # Forces are repulsive 0-1, zero 1-2, attractive 2-3, asymptotic 3+
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 2.5],
        [0.0, 0.0, 3.5],
        [0.0, 0.0, 5.0],
    ], dtype='float64')
    indices = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    ], dtype='int32')
    weights = np.ones(len(indices), dtype='float64')
    forces = np.zeros(coords.shape, dtype='float64')
    limits = np.broadcast_to([[1.0, 2.0]], (len(indices), 2)).astype('float64')
    ambig = np.arange(len(indices) + 1, dtype='int32')

    forces = np.zeros(coords.shape, dtype='float64')
    getRestraintForce(
        forces, coords, indices, limits, weights, ambig, fConst=1.0, exponent=1.0
    )
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0,-0.5],
        [0.0, 0.0,-1.0],
        [0.0, 0.0,-1.0],
    ], dtype='float64')
    expected[0] = -expected[1:].sum(axis=0)

    print(forces)
    np.testing.assert_allclose(forces, expected)
