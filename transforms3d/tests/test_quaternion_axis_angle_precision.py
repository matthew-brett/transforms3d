import math

import numpy as np
import pytest

import transforms3d.quaternions as tq


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("angle", [1e-10, 1e-8, 1e-6, 1e-4])
def test_axis_angle_retains_small_rotation_angles(dtype, angle):
    axis = np.array([0.6, -0.8, 0.0])
    q = np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis].astype(dtype)
    actual_axis, actual_angle = tq.quat2axangle(q)
    # The expected angle uses both independently stored quaternion parts.
    imaginary_norm = math.sqrt(sum(float(value)**2 for value in q[1:]))
    expected_angle = 2 * math.atan2(imaginary_norm, float(q[0]))
    assert actual_angle > 0.0
    np.testing.assert_allclose(actual_angle, expected_angle,
                               rtol=3e-7 if dtype is np.float32 else 1e-14, atol=0.0)
    np.testing.assert_allclose(actual_axis, axis,
                               rtol=3e-7 if dtype is np.float32 else 1e-14, atol=1e-15)
    np.testing.assert_allclose(tq.axangle2quat(actual_axis, actual_angle), q,
                               rtol=3e-7 if dtype is np.float32 else 1e-14, atol=1e-15)


@pytest.mark.parametrize("angle", [0.2, math.pi, 2 * math.pi - 0.2])
@pytest.mark.parametrize("scale", [1.0, 7.0])
def test_existing_orientation_range_and_nonunit_inputs(angle, scale):
    axis = np.array([0.6, -0.8, 0.0])
    q = scale * np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]
    actual_axis, actual_angle = tq.quat2axangle(q)
    np.testing.assert_allclose(actual_angle, angle, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual_axis, axis, rtol=1e-14, atol=1e-14)


def test_identity_and_explicit_identity_threshold_controls():
    np.testing.assert_array_equal(tq.quat2axangle([1.0, 0.0, 0.0, 0.0])[0], [1.0, 0.0, 0.0])
    assert tq.quat2axangle([1.0, 0.0, 0.0, 0.0])[1] == 0.0
    assert tq.quat2axangle([1.0, 1e-8, 0.0, 0.0], identity_thresh=1e-6)[1] == 0.0
