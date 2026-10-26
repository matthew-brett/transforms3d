"""Shear decomposition must include the zero-angle (identity) transform."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from transforms3d.shears import aff2sadn, mat2sadn, sadn2aff, sadn2mat


def assert_shear_vectors(direction, normal):
    # The identity has no unique plane: require valid vectors, not chosen axes.
    assert direction.shape == normal.shape == (3,)
    assert np.isfinite(direction).all()
    assert np.isfinite(normal).all()
    assert_allclose(np.linalg.norm(direction), 1)
    assert_allclose(np.linalg.norm(normal), 1)
    assert_allclose(np.dot(direction, normal), 0, atol=1e-15)


@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_identity_shear_matrix(dtype):
    matrix = np.eye(3, dtype=dtype)
    angle, direction, normal = mat2sadn(matrix)
    assert angle == 0
    assert_shear_vectors(direction, normal)
    assert_array_equal(sadn2mat(angle, direction, normal), matrix)


@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_identity_shear_affine(dtype):
    affine = np.eye(4, dtype=dtype)
    with np.errstate(divide='raise', invalid='raise'):
        angle, direction, normal, point = aff2sadn(affine)
    assert angle == 0
    assert_shear_vectors(direction, normal)
    assert point.shape == (3,)
    assert np.isfinite(point).all()
    assert_array_equal(sadn2aff(angle, direction, normal, point), affine)


@pytest.mark.parametrize('point', [None, [2, -3, 4]])
def test_zero_angle_shear_roundtrip(point):
    direction = [1, 2, -3]
    normal = [-2, 1, 0]
    affine = sadn2aff(0, direction, normal, point)
    angle, direction, normal, point = aff2sadn(affine)
    assert_shear_vectors(direction, normal)
    assert np.isfinite(point).all()
    assert_array_equal(sadn2aff(angle, direction, normal, point), affine)
    # The same recovered vectors must also be valid for the matrix constructor.
    assert_array_equal(sadn2mat(angle, direction, normal), affine[:3, :3])


@pytest.mark.parametrize('angle', [-0.3, -1e-12, -1e-20, 1e-20, 1e-12, 0.3])
def test_nonzero_shear_roundtrip(angle):
    direction = np.array([1., 0, 0])
    normal = np.array([0., 1, 0])
    point = np.array([1., 2, 3])
    tangent = np.tan(angle)
    matrix = sadn2mat(angle, direction, normal)
    affine = sadn2aff(angle, direction, normal, point)
    # Physical oracle: x moves by tan(angle) * (y - plane_y).
    probe = np.array([4., 5, 6])
    expected = probe + [tangent * (probe[1] - point[1]), 0, 0]
    assert_allclose(affine[:3, :3].dot(probe) + affine[:3, 3], expected)
    a, d, n = mat2sadn(matrix)
    assert a != 0  # No tolerance may erase a representable nonzero shear.
    assert_shear_vectors(d, n)
    assert_allclose(sadn2mat(a, d, n), matrix, rtol=1e-14, atol=0)
    a, d, n, p = aff2sadn(affine)
    assert a != 0
    assert_shear_vectors(d, n)
    assert np.isfinite(p).all()
    assert_allclose(sadn2aff(a, d, n, p), affine, rtol=1e-14, atol=0)
