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
