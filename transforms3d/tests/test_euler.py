""" Test for euler module
"""

from itertools import product, permutations

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
