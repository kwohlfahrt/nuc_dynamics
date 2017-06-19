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

