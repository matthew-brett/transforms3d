""" Test for euler module
"""

import math

from itertools import product, permutations

import numpy as np

from .. import euler
from ..euler import euleraxes2mat, mataxes2euler
from ..taitbryan import euler2mat

from .samples import euler_tuples, euler_mats

from nose.tools import assert_true
from numpy.testing import assert_almost_equal


def test_euler_axes():
    # Test there and back with all axis specs
    aba_perms = [(v[0], v[1], v[0]) for v in permutations('xyz', 2)]
    for rs, axes in product('rs', list(permutations('xyz', 3)) + aba_perms):
        ax_spec = rs + ''.join(axes)
        for mat in euler_mats:
            a, b, c = mataxes2euler(mat, ax_spec)
            mat_back = euleraxes2mat(a, b, c, ax_spec)
            assert_almost_equal(mat, mat_back)


def test_with_euler2mat():
    # Test against Tait-Bryan implementation
    for a, b, c in euler_tuples:
        tb_mat = euler2mat(a, b, c)
        gen_mat = euleraxes2mat(a, b, c, 'szyx')
        assert_almost_equal(tb_mat, gen_mat)


def test_euleraxes2mat():
    # Test mat creation from random angles and round trip
    ai, aj, ak = (4 * math.pi) * (np.random.random(3) - 0.5)
    for axes in euler._AXES2TUPLE.keys():
       R = euleraxes2mat(ai, aj, ak, axes)
       bi, bj, bk = mataxes2euler(R, axes)
       R2 = euleraxes2mat(bi, bj, bk, axes)
       assert_almost_equal(R, R2)


def test_mataxes2euler():
    # Test mataxes2euler function
    angles = (4 * math.pi) * (np.random.random(3) - 0.5)
    for axes in euler._AXES2TUPLE.keys():
       R0 = euleraxes2mat(axes=axes, *angles)
       R1 = euleraxes2mat(axes=axes, *mataxes2euler(R0, axes))
       assert_almost_equal(R0, R1)
