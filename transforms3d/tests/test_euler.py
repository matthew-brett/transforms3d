""" Test for euler module
"""

import math

from itertools import product, permutations

import numpy as np

from .. import euler
from ..quaternions import mat2quat, nearly_equivalent
from ..axangles import mat2axangle, axangle2mat
from ..euler import (euler2mat, mat2euler, euler2quat, quat2euler,
                     euler2axangle, axangle2euler, EulerFuncs)
from .. import taitbryan as tb

from .samples import euler_tuples, euler_mats

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_almost_equal


def test_euler_axes():
    # Test there and back with all axis specs
    aba_perms = [(v[0], v[1], v[0]) for v in permutations('xyz', 2)]
    axis_perms = list(permutations('xyz', 3)) + aba_perms
    for (a, b, c), mat in zip(euler_tuples, euler_mats):
        for rs, axes in product('rs', axis_perms):
            ax_spec = rs + ''.join(axes)
            conventioned = [EulerFuncs(ax_spec)]
            if ax_spec in euler.__dict__:
                conventioned.append(euler.__dict__[ax_spec])
            mat = euler2mat(a, b, c, ax_spec)
            a1, b1, c1 = mat2euler(mat, ax_spec)
            mat_again = euler2mat(a1, b1, c1, ax_spec)
            assert_almost_equal(mat, mat_again)
            quat = euler2quat(a, b, c, ax_spec)
            a1, b1, c1 = quat2euler(quat, ax_spec)
            mat_again = euler2mat(a1, b1, c1, ax_spec)
            assert_almost_equal(mat, mat_again)
            ax, angle = euler2axangle(a, b, c, ax_spec)
            a1, b1, c1 = axangle2euler(ax, angle, ax_spec)
            mat_again = euler2mat(a1, b1, c1, ax_spec)
            assert_almost_equal(mat, mat_again)
            for obj in conventioned:
                mat = obj.euler2mat(a, b, c)
                a1, b1, c1 = obj.mat2euler(mat)
                mat_again = obj.euler2mat(a1, b1, c1)
                assert_almost_equal(mat, mat_again)
                quat = obj.euler2quat(a, b, c)
                a1, b1, c1 = obj.quat2euler(quat)
                mat_again = obj.euler2mat(a1, b1, c1)
                assert_almost_equal(mat, mat_again)
                ax, angle = obj.euler2axangle(a, b, c)
                a1, b1, c1 = obj.axangle2euler(ax, angle)
                mat_again = obj.euler2mat(a1, b1, c1)
                assert_almost_equal(mat, mat_again)


def test_with_euler2mat():
    # Test against Tait-Bryan implementation
    for a, b, c in euler_tuples:
        tb_mat = tb.euler2mat(a, b, c)
        gen_mat = euler2mat(a, b, c, 'szyx')
        assert_almost_equal(tb_mat, gen_mat)


def test_euler2mat():
    # Test mat creation from random angles and round trip
    ai, aj, ak = (4 * math.pi) * (np.random.random(3) - 0.5)
    for ax_specs in euler._AXES2TUPLE.items():
        for ax_spec in ax_specs:
            R = euler2mat(ai, aj, ak, ax_spec)
            bi, bj, bk = mat2euler(R, ax_spec)
            R2 = euler2mat(bi, bj, bk, ax_spec)
            assert_almost_equal(R, R2)


def test_mat2euler():
    # Test mat2euler function
    angles = (4 * math.pi) * (np.random.random(3) - 0.5)
    for axes in euler._AXES2TUPLE.keys():
       R0 = euler2mat(axes=axes, *angles)
       R1 = euler2mat(axes=axes, *mat2euler(R0, axes))
       assert_almost_equal(R0, R1)


def test_names():
    assert_equal(euler.__dict__['physics'], euler.__dict__['rzxz'])
