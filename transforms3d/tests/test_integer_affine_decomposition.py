import numpy as np
import pytest

from transforms3d.affines import compose, decompose, decompose44


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint16])
def test_integral_identity_affine_is_decomposed_in_float(dtype):
    affine = np.eye(4, dtype=dtype)
    affine[:3, 3] = [2, 3, 4]
    saved = affine.copy()
    t, r, z, s = decompose44(affine)
    np.testing.assert_array_equal(t, [2, 3, 4])
    np.testing.assert_allclose(r, np.eye(3), atol=0.0)
    np.testing.assert_array_equal(z, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(s, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(affine, saved)
    np.testing.assert_allclose(compose(t, r, z, s), affine, atol=0.0)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_integral_rotated_scaled_sheared_affine(dtype):
    affine = np.array([[0, -3, -6, 2], [2, 2, -2, -3], [0, 0, 4, 1], [0, 0, 0, 1]], dtype=dtype)
    saved = affine.copy()
    t, r, z, s = decompose44(affine)
    np.testing.assert_allclose(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-14)
    np.testing.assert_allclose(z, [2, 3, 4], atol=1e-14)
    np.testing.assert_allclose(s, [1, -1, 2], atol=1e-14)
    np.testing.assert_allclose(compose(t, r, z, s), affine, atol=1e-14)
    for actual, expected in zip((t, r, z, s), decompose(affine)):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(affine, saved)


def test_readonly_integer_reflection_and_boolean_identity():
    affine = np.diag([-2, 3, 4, 1])
    affine.flags.writeable = False
    t, r, z, s = decompose44(affine)
    np.testing.assert_allclose(r, np.eye(3), atol=0.0)
    np.testing.assert_array_equal(z, [-2, 3, 4])
    np.testing.assert_allclose(compose(t, r, z, s), affine, atol=0.0)
    np.testing.assert_allclose(compose(*decompose44(np.eye(4, dtype=bool))), np.eye(4), atol=0.0)
