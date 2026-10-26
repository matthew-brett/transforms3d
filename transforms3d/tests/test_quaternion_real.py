''' Real-axis limits of quaternion exponential, logarithm and power. '''

import cmath
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from transforms3d import quaternions as tq


DTYPES = (np.float32, np.float64, np.longdouble)


def tolerance(dtype):
    # The complex reference uses Python's double-precision complex arithmetic.
    return max(16 * np.finfo(dtype).eps, 2e-14)


@pytest.mark.parametrize('dtype', DTYPES)
def test_qexp_real(dtype):
    for w in (-3., -1., -0.25, 0., 0.25, 1., 3.):
        q = np.array([w, 0., 0., 0.], dtype=dtype)
        result = tq.qexp(q)
        assert_allclose(result, [math.exp(w), 0., 0., 0.],
                        rtol=tolerance(dtype), atol=0)
        assert result.dtype == dtype
        assert_array_equal(q, [w, 0., 0., 0.])


@pytest.mark.parametrize('dtype', DTYPES)
def test_qlog_real(dtype):
    for w in (-4., -1., -0.25, 0.25, 1., 4.):
        q = np.array([w, 0., 0., 0.], dtype=dtype)
        expected = cmath.log(complex(w, 0.))
        result = tq.qlog(q)
        assert_allclose(result, [expected.real, expected.imag, 0., 0.],
                        rtol=tolerance(dtype), atol=0)
        assert result.dtype == dtype
        assert_array_equal(q, [w, 0., 0., 0.])


@pytest.mark.parametrize('dtype', DTYPES)
def test_qpow_real(dtype):
    for w in (-4., -1., -0.25, 0.25, 1., 4.):
        q = np.array([w, 0., 0., 0.], dtype=dtype)
        for power in (-2, -1, -0.5, 0, 0.5, 1, 2, 2.5):
            expected = complex(w, 0.) ** power
            result = tq.qpow(q, power)
            assert_allclose(result,
                            [expected.real, expected.imag, 0., 0.],
                            rtol=tolerance(dtype), atol=tolerance(dtype))
            assert result.dtype == dtype
        assert_array_equal(q, [w, 0., 0., 0.])


@pytest.mark.parametrize('dtype', DTYPES)
def test_real_exp_log_roundtrip(dtype):
    for w in (-4., -1., -0.25, 0.25, 1., 4.):
        q = np.array([w, 0., 0., 0.], dtype=dtype)
        # Do not use nearly_equivalent: opposite signs are not equivalent
        # for general quaternion arithmetic, unlike rotation quaternions.
        assert_allclose(tq.qexp(tq.qlog(q)), q,
                        rtol=tolerance(dtype), atol=tolerance(dtype))
        assert_allclose(tq.qlog(tq.qexp(q)), q,
                        rtol=tolerance(dtype), atol=tolerance(dtype))


@pytest.mark.parametrize('dtype', DTYPES)
def test_real_integer_power_algebra(dtype):
    for w in (-4., -1., -0.25, 0.25, 1., 4.):
        q = np.array([w, 0., 0., 0.], dtype=dtype)
        product = tq.qeye(dtype)
        for power in range(5):
            assert_allclose(tq.qpow(q, power), product,
                            rtol=tolerance(dtype), atol=0)
            product = tq.qmult(product, q)
        assert_allclose(tq.qmult(q, tq.qpow(q, -1)), tq.qeye(dtype),
                        rtol=tolerance(dtype), atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
def test_real_log_extreme_magnitudes(dtype):
    info = np.finfo(dtype)
    smallest = np.nextafter(dtype(0), dtype(1))
    for magnitude in (smallest, info.tiny, info.max):
        for sign in (-1, 1):
            q = np.array([sign * magnitude, 0., 0., 0.], dtype=dtype)
            expected = np.zeros(4, dtype=dtype)
            expected[0] = np.log(magnitude)
            if sign < 0:
                expected[1] = np.pi
            with np.errstate(over='raise', under='raise', invalid='raise'):
                result = tq.qlog(q)
            assert_allclose(result, expected, rtol=tolerance(dtype), atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
def test_real_power_extreme_magnitudes(dtype):
    info = np.finfo(dtype)
    for magnitude in (info.tiny, info.max):
        for sign in (-1, 1):
            q = np.array([sign * magnitude, 0., 0., 0.], dtype=dtype)
            # Some longdouble power implementations flag overflow even for
            # finite scalar powers.  Preserve NumPy's warning and check values.
            assert_array_equal(tq.qpow(q, 0), tq.qeye(dtype))
            assert_array_equal(tq.qpow(q, 1), q)
            reciprocal = tq.qpow(q, -1)
            expected = np.array([dtype(1) / q[0], 0., 0., 0.], dtype=dtype)
            assert_array_equal(reciprocal, expected)


@pytest.mark.parametrize('factory', (list, tuple, np.array))
def test_real_arraylike_inputs(factory):
    q = factory([2., 0., 0., 0.])
    if isinstance(q, np.ndarray):
        q.flags.writeable = False
    assert_allclose(tq.qexp(q), [math.exp(2), 0., 0., 0.])
    assert_allclose(tq.qlog(q), [math.log(2), 0., 0., 0.])
    assert_array_equal(tq.qpow(q, 2), [4., 0., 0., 0.])


@pytest.mark.parametrize('dtype', DTYPES)
def test_zero_quaternion_legacy_shortcuts(dtype):
    # These existing zero-base conventions are not redesigned by this fix.
    q = np.zeros(4, dtype=dtype)
    assert_array_equal(tq.qexp(q), tq.qeye(dtype))
    assert_array_equal(tq.qlog(q), tq.qeye(dtype))
    for power in (-1, 0, 0.5, 2):
        assert_array_equal(tq.qpow(q, power), tq.qeye(dtype))


@pytest.mark.parametrize('dtype', DTYPES)
def test_qexp_nonreal_underflow_shortcut_unchanged(dtype):
    # The exact-real fix does not redesign the existing underflow policy for
    # a nonzero imaginary part whose squared norm rounds to zero.
    smallest = np.nextafter(dtype(0), dtype(1))
    q = np.array([2., smallest, 0., 0.], dtype=dtype)
    assert q[1] != 0
    assert_array_equal(tq.qexp(q), tq.qeye(dtype))


def test_longdouble_negative_real_branch():
    dtype = np.longdouble
    if np.finfo(dtype).eps >= np.finfo(np.float64).eps:
        pytest.skip('longdouble has no extra precision on this platform')
    eps = np.finfo(dtype).eps
    # Independent identity, without first rounding pi to a Python float.
    pi = dtype(4) * np.arctan(dtype(1))
    q = np.array([-1., 0., 0., 0.], dtype=dtype)
    assert_allclose(tq.qlog(q)[1], pi, rtol=8 * eps, atol=0)
    for power, sign in ((0.5, 1), (-0.5, -1), (1.5, -1), (-1.5, 1)):
        expected = np.array([0., sign, 0., 0.], dtype=dtype)
        assert_allclose(tq.qpow(q, power), expected, rtol=8 * eps, atol=8 * eps)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('exponent_type', (float, np.longdouble))
def test_negative_real_power_phase_reduction(dtype, exponent_type):
    q = np.array([-1., 0., 0., 0.], dtype=dtype)
    exponent_dtype = np.dtype(exponent_type)
    result_dtype = (np.dtype(dtype) if exponent_type is float else
                    np.result_type(dtype, exponent_dtype))
    eps = np.finfo(result_dtype).eps
    powers = [40, 50]
    if np.finfo(exponent_dtype).nmant > 60:
        powers.append(60)
    for power in powers:
        whole = exponent_type(2) ** power
        for offset, imaginary_sign in ((0.5, 1), (1.5, -1)):
            for sign in (-1, 1):
                exponent = sign * (whole + exponent_type(offset))
                assert abs(np.fmod(exponent, 1)) == 0.5
                # (-1)**2 = 1, so these representable half-integers must
                # have exactly the same phase as +/-0.5 or +/-1.5.
                expected = np.array([0., sign * imaginary_sign, 0., 0.],
                                    dtype=result_dtype)
                result = tq.qpow(q, exponent)
                assert result.dtype == result_dtype
                assert_allclose(result, expected, rtol=16 * eps, atol=16 * eps)
    phase_dtype = np.result_type(result_dtype, np.float64)
    pi = phase_dtype.type(4) * np.arctan(phase_dtype.type(1))
    for exponent in (exponent_type(-1e-20), -exponent_type(2) ** -100):
        result = tq.qpow(q, exponent)
        assert result.dtype == result_dtype
        assert result[0] == 1
        # At these magnitudes sin(pi*n) rounds to pi*n.  Zero absolute
        # tolerance detects loss of tiny negative phase under positive mod.
        assert_allclose(result[1], pi * exponent, rtol=16 * eps, atol=0)
        assert_array_equal(result[2:], [0., 0.])
    for even in (-4, -2, 2, 4):
        exponent = exponent_type(np.nextafter(exponent_type(even),
                                              exponent_type(0)))
        offset = exponent - exponent_type(even)
        result = tq.qpow(q, exponent)
        assert result[0] == 1
        # The tiny offset from the even integer determines the phase.  Center
        # reduction before pi multiplication instead of evaluating near 2*pi.
        assert_allclose(result[1], pi * offset, rtol=16 * eps, atol=0)
        assert_array_equal(result[2:], [0., 0.])
