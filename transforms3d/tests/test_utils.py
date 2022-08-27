""" Test utils module
"""

import numpy as np

from transforms3d.utils import random_unit_vector, np_default_rng


def test_random_unit_vector():
    vec_sum = np.zeros(3)
    n = 1000
    for i in range(n):
        vec = random_unit_vector()
        np.all(np.abs(vec) <= 1)
        assert np.isclose(np.sqrt(np.sum(vec @ vec)), 1)
        vec_sum += vec
    assert np.all(np.abs(vec_sum / n) < 0.05)
    rng1 = np_default_rng(12)
    vec = random_unit_vector(rng1)
    rng2 = np_default_rng(12)
    vec2 = random_unit_vector(rng2)
    assert np.all(vec == vec2)
