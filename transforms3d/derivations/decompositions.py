''' Derivations for extracting rotations, zooms, shears '''
import numpy as np

from sympy import Symbol, symbols
from sympy.matrices import Matrix

sx, sy, sz, sxy, sxz, syz = symbols('sx, sy, sz, sxy, sxz, syz')

R = Matrix(3, 3, lambda i, j : Symbol('R%d%d' % (i, j)))
Z = Matrix(np.diag([sx, sy, sz]))
S = Matrix([[1, sxy,sxz],
            [0,  1, syz],
            [0,  0,   1]])

# Rotations composed on zooms composed on shears
RZS = R * Z * S

# Results used in subsequent decompositions
R0_RZS1 = R[:,0].T * RZS[:,1]
R0_RZS2 = R[:,0].T * RZS[:,2]
R1_RZS2 = R[:,1].T * RZS[:,2]

