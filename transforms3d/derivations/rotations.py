''' Symbolic representations of rotation matrices '''

from sympy import Symbol

from sympy.matrices import Matrix

R = Matrix(3, 3, lambda i, j : Symbol('R%d%d' % (i, j)))
