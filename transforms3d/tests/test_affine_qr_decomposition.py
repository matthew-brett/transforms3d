import numpy as np
import pytest

from transforms3d.affines import compose, decompose


@pytest.mark.parametrize("shear", [1e6, 1e8])
def test_nonsingular_triangular_affine_retains_rotation_and_shears(shear):
    affine = np.array([[1.0, shear, 0.0, 2.0], [0.0, 1.0, shear, -3.0],
                       [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]])
    t, r, z, s = decompose(affine)
    np.testing.assert_allclose(t, [2, -3, 4], atol=0.0)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(z, [1, 1, 1], atol=1e-14)
    np.testing.assert_allclose(s, [shear, 0, shear], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(compose(t, r, z, s), affine, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("reflection", [False, True])
def test_qr_recovers_known_factors_in_arbitrary_dimensions(dimension, reflection):
    rotation = np.eye(dimension)
    rotation[:2, :2] = [[0.6, -0.8], [0.8, 0.6]]
    scale = np.arange(1, dimension + 1, dtype=float)
    if reflection:
        scale[0] *= -1
    shears = np.arange(dimension * (dimension - 1) // 2) * 0.2 - 0.3
    translation = np.arange(dimension, dtype=float) - 1
    affine = compose(translation, rotation, scale, shears)
    t, r, z, s = decompose(affine)
    for actual, expected in zip((t, r, z, s), (translation, rotation, scale, shears)):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(r.T @ r, np.eye(dimension), atol=1e-13)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-13)
    np.testing.assert_allclose(compose(t, r, z, s), affine, rtol=1e-13, atol=1e-13)


def test_exact_zero_scale_still_raises_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        decompose(np.diag([1.0, 0.0, 2.0, 1.0]))
