import numpy as np
from numpy import testing

def test_random_coords_shape():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    pos_dict = {'foo': np.arange(100), 'bar': np.arange(200)}
    coords = get_random_coords(pos_dict, pos_dict.keys(), 2, radius)
    assert coords.shape == (2, 300, 3)


def test_random_coords_sphere():
    from nuc_dynamics import get_random_coords

    np.random.seed(4)
    radius = 10.0
    pos_dict = {'foo': np.arange(100), 'bar': np.arange(200)}
    coords = get_random_coords(pos_dict, pos_dict.keys(), 2, radius)

    assert np.alltrue(np.linalg.norm(coords, axis=-1) <= radius)
    assert not np.alltrue(np.linalg.norm(coords, axis=-1) <= radius * 0.5)
