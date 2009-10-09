""" Testing zooms and shears 

"""

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

import transforms3d.zooms_shears as tzs


def test_zdir_zmat_aff():
    # test zdir to zmat and back
    for i in range(5):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        S0 = tzs.zdir2zmat(factor, None)
        f2, d2 = tzs.zmat2zdir(S0)
        S1 = tzs.zdir2zmat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        direct = np.random.random(3) - 0.5
        S0 = tzs.zdir2zmat(factor, direct)
        f2, d2 = tzs.zmat2zdir(S0)
        S1 = tzs.zdir2zmat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        # affine versions of same
        S0 = tzs.zdir2aff(factor)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)
        S0 = tzs.zdir2aff(factor, direct)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)
        S0 = tzs.zdir2aff(factor, direct, origin)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)

    
