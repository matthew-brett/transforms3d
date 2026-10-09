"""Scale-invariant axis normalization and its public consumers."""

from decimal import Decimal, localcontext
from itertools import permutations
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from transforms3d import axangles, euler, taitbryan


def _decimal_unit_axis(axis):
    # Normalize exact input values at higher precision, without the producer's
    # max-component scaling algorithm or floating-point squared norm.
    with localcontext() as context:
        context.prec = 100
        components = [Decimal.from_float(float(value)) for value in axis]
        norm = sum(value * value for value in components).sqrt()
        return np.array([float(value / norm) for value in components])


def _rodrigues_reference(axis, angle):
    unit = _decimal_unit_axis(axis)
    x, y, z = unit
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (math.cos(angle) * np.eye(3)
            + math.sin(angle) * cross
            + (1. - math.cos(angle)) * np.outer(unit, unit))


def _tolerance(dtype):
    return 2e-6 if dtype == np.float32 else 6e-15


def _assert_rotation(matrix, reference, dtype):
    tolerance = _tolerance(dtype)
    assert np.isfinite(matrix).all()
    assert_allclose(matrix, reference, rtol=tolerance, atol=tolerance)
    assert_allclose(matrix.T @ matrix, np.eye(3), rtol=0., atol=tolerance)
    assert_allclose(np.linalg.det(matrix), 1., rtol=0., atol=tolerance)


def _extreme_axis(dtype, exponent):
    return np.ldexp(np.array([1., -2., 3.], dtype=dtype), exponent)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('container', ['array', 'list', 'tuple'])
def test_axis_normalization_is_invariant_to_extreme_binary_scale(dtype,
                                                                 container):
    exponents = [-120, -60, 0, 60, 120] if dtype == np.float32 else [
        -1000, -500, 0, 500, 1000]
    for exponent in exponents:
        values = _extreme_axis(dtype, exponent)
        original = values.copy()
        if container == 'array':
            values.setflags(write=False)
            axis = values
        elif container == 'list':
            axis = values.tolist()
        else:
            axis = tuple(values.tolist())
        for angle in (0.375, math.pi / 2):
            expected = _rodrigues_reference(axis, angle)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                actual = axangles.axangle2mat(axis, angle)
            _assert_rotation(actual, expected, dtype)
        assert_array_equal(np.asarray(axis), original)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_finite_axis_whose_unscaled_norm_exceeds_float_range(dtype):
    maximum = np.finfo(dtype).max
    axis = np.array([maximum, maximum, -maximum], dtype=dtype)
    expected = _rodrigues_reference(axis, 0.375)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        actual = axangles.axangle2mat(axis, 0.375)
    _assert_rotation(actual, expected, dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_minimum_subnormal_nonzero_axis_retains_its_direction(dtype):
    minimum = np.nextafter(dtype(0), dtype(1))
    assert minimum > 0
    axis = np.array([minimum, -minimum, minimum], dtype=dtype)
    expected = _rodrigues_reference(axis, math.pi / 2)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        actual = axangles.axangle2mat(axis, math.pi / 2)
    _assert_rotation(actual, expected, dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_axis_permutations_sign_and_inverse_angle(dtype):
    extreme = 120 if dtype == np.float32 else 1000
    for exponent in (-extreme, extreme):
        for values in permutations(_extreme_axis(dtype, exponent)):
            axis = np.asarray(values, dtype=dtype)
            expected = _rodrigues_reference(axis, 0.375)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                actual = axangles.axangle2mat(axis, 0.375)
                sign_reversed = axangles.axangle2mat(-axis, -0.375)
                inverse = axangles.axangle2mat(axis, -0.375)
            _assert_rotation(actual, expected, dtype)
            assert_allclose(sign_reversed, actual, rtol=0.,
                            atol=_tolerance(dtype))
            assert_allclose(inverse, actual.T, rtol=0.,
                            atol=_tolerance(dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_affine_rotation_about_point_uses_normalized_axis(dtype):
    extreme = 120 if dtype == np.float32 else 1000
    point = np.array([1.25, -2.5, 0.75])
    displacement = np.array([0.5, -0.25, 2.])
    for exponent in (-extreme, extreme):
        axis = _extreme_axis(dtype, exponent)
        rotation = _rodrigues_reference(axis, 0.375)
        expected = np.eye(4)
        expected[:3, :3] = rotation
        expected[:3, 3] = point - rotation @ point
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            actual = axangles.axangle2aff(axis, 0.375, point)
        assert_allclose(actual, expected, rtol=_tolerance(dtype),
                        atol=_tolerance(dtype))
        assert_allclose(actual @ np.r_[point, 1.], np.r_[point, 1.],
                        rtol=0., atol=_tolerance(dtype))
        assert_allclose(actual @ np.r_[point + displacement, 1.],
                        np.r_[point + rotation @ displacement, 1.],
                        rtol=_tolerance(dtype), atol=_tolerance(dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_euler_consumers_reconstruct_scale_invariant_rotation(dtype):
    extreme = 120 if dtype == np.float32 else 1000
    for exponent in (-extreme, extreme):
        axis = _extreme_axis(dtype, exponent)
        expected = _rodrigues_reference(axis, 0.375)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for convention in euler._AXES2TUPLE:
                angles = euler.axangle2euler(axis, 0.375, axes=convention)
                actual = euler.euler2mat(*angles, axes=convention)
                _assert_rotation(actual, expected, dtype)
            angles = taitbryan.axangle2euler(axis, 0.375)
            _assert_rotation(taitbryan.euler2mat(*angles), expected, dtype)


def test_ordinary_integer_axes_and_already_normalized_control():
    expected = _rodrigues_reference([1, -2, 3], 0.375)
    for axis in ([1, -2, 3], (1, -2, 3),
                 np.array([1, -2, 3], dtype=np.int64)):
        original = np.asarray(axis).copy()
        actual = axangles.axangle2mat(axis, 0.375)
        _assert_rotation(actual, expected, np.float64)
        assert_array_equal(np.asarray(axis), original)
        assert_array_equal(axangles.axangle2mat(axis, 0.), np.eye(3))
    negative_x = np.array([np.iinfo(np.int64).min, 0, 0], dtype=np.int64)
    cosine, sine = math.cos(0.375), math.sin(0.375)
    expected_x = np.array([[1., 0., 0.], [0., cosine, sine],
                           [0., -sine, cosine]])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        _assert_rotation(axangles.axangle2mat(negative_x, 0.375), expected_x,
                         np.float64)
        large_integer = np.array([2**62, -2**61, 2**60], dtype=np.int64)
        _assert_rotation(axangles.axangle2mat(large_integer, 0.375),
                         _rodrigues_reference(large_integer, 0.375), np.float64)
    unit = np.array([2. / 3, 1. / 3, 2. / 3])
    _assert_rotation(axangles.axangle2mat(unit, 0.375, is_normalized=True),
                     _rodrigues_reference(unit, 0.375), np.float64)


def test_zero_and_nonfinite_axis_behavior_is_preserved():
    for axis in ([0., 0., 0.], (0., 0., 0.)):
        with pytest.raises(ZeroDivisionError):
            axangles.axangle2mat(axis, 0.375)
    with pytest.warns(RuntimeWarning):
        zero_array_matrix = axangles.axangle2mat(np.zeros(3), 0.375)
    assert np.isnan(zero_array_matrix).all()
    for position in range(3):
        axis = [0., 0., 0.]
        axis[position] = float('nan')
        assert np.isnan(axangles.axangle2mat(tuple(axis), 0.375)).all()
        axis[position] = float('inf')
        expected = np.full((3, 3), np.nan)
        for diagonal in range(3):
            if diagonal != position:
                expected[diagonal, diagonal] = math.cos(0.375)
        assert_allclose(axangles.axangle2mat(tuple(axis), 0.375), expected,
                        rtol=0., atol=0., equal_nan=True)
