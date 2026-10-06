"""Independent review cases; not part of the frozen original candidate patch."""
from fractions import Fraction

import numpy as np
import pytest

from transforms3d.affines import decompose


def exact_rank(values):
    rows = [[Fraction(value) for value in row] for row in values]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        factor = rows[rank][column]
        rows[rank] = [value / factor for value in rows[rank]]
        for index in range(rank + 1, len(rows)):
            factor = rows[index][column]
            rows[index] = [value - factor * basis for value, basis in zip(rows[index], rows[rank])]
        rank += 1
    return rank


@pytest.mark.parametrize("linear", [
    [[1, 2], [2, 4]],
    [[1, 1], [1, 1]],
    [[1, 2, 3], [2, 4, 1], [3, 6, 2]],
    [[1, 1, 0], [2, 2, 1], [3, 3, 2]],
    [[0.5, 1.0], [1.25, 2.5]],
    [[0.5, 1.0, 2.0], [1.25, 2.5, 0.5], [2.25, 4.5, 1.5]],
])
def test_strictly_rank_deficient_affine_is_rejected(linear):
    dimension = len(linear)
    # These literals are integers or exactly representable binary fractions.
    # Rational elimination proves dependence without a floating rank tolerance.
    assert exact_rank(linear) < dimension
    affine = np.eye(dimension + 1)
    affine[:dimension, :dimension] = linear
    with pytest.raises(np.linalg.LinAlgError):
        decompose(affine)
