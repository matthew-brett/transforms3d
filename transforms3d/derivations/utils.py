''' Utilities for derivations '''
import numpy as np

import sympy

from sympy import Symbol, simplify

from sympy.matrices import Matrix, zeros


def make_matrix(name_prefix, N, M):
    name_format = name_prefix + '%d%d'
    name_func = lambda i, j : Symbol(name_format % (i, j))
    return Matrix(N, M, name_func)
                  

def matrices_equal(M1, M2):
    d = M1 - M2
    d.simplify()
    return np.all(np.array(d) == 0)
    

def matrix_simplify(M):
    n, m = M.shape
    Md = zeros((n, m))
    for i in range(n):
        for j in range(m):
            Md[i,j] = simplify(M[i,j])
    return Md
