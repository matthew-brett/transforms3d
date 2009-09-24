import numpy as np

from .utils import inique, permuted_signs, permuted_with_signs

from ..taitbryan import euler2mat

# Regular points around a sphere
_r13 = np.sqrt(1/3.0)
_r12 = np.sqrt(0.5)
sphere_points = (
        tuple(inique(permuted_with_signs([1, 0, 0]))) + 
        tuple(inique(permuted_with_signs([_r12, _r12, 0]))) + 
        tuple(inique(permuted_signs([_r13, _r13, _r13])))
    )

# Example rotations '''
euler_tuples = []
params = (-np.pi,np.pi,np.pi/2)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            euler_tuples.append((x, y, z))

euler_mats = tuple(euler2mat(*t) for t in euler_tuples)

