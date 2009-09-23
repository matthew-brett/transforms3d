''' Utilities for derivations '''
import numpy as np

import sympy


def matrices_equal(M1, M2):
    d = M1 - M2
    d.simplify()
    return np.all(np.array(d) == 0)
    
