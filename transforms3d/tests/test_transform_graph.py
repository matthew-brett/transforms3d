from transforms3d.transform_graph import Transformation, create_tranformation_graph, get_tranformation_matrix
import numpy as np
import igraph as ig
import math
import pytest

from transforms3d.euler import euler2mat
from transforms3d.affines import compose


def define_ros_transform(x, y, z, roll, pitch, yaw, deg: bool = False):

    if deg:
        roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)

    R = euler2mat(roll, pitch, yaw, axes="rxyz")
    t = np.array([x, y, z])

    H = compose(t, R, np.ones((3,)))
    # equals to
    # H = np.zeros((4, 4))
    # H[:3, :3] = R
    # H[:3, 3] = t
    # H[3, 3] = 1.0
    return H


def vector_with_one(x, y, z):
    v = np.ones((4))
    v[:3] = [x, y ,z]
    return v


def test_simple_world():

    world_to_baloon = Transformation("world", "baloon",
        define_ros_transform(50, 50, 50, 0, 0, 0))
    world_to_dragon = Transformation("world", "dragon",
        define_ros_transform(-50, 0, 50, 3.0, -3.0, 0, deg=True))

    simple_world_transforms = [
        world_to_baloon,
        world_to_dragon,
    ]

    g = create_tranformation_graph(simple_world_transforms)
    assert isinstance(g, ig.Graph)

    H = get_tranformation_matrix(g, "dragon", "baloon")

    H_inv = get_tranformation_matrix(g, "baloon", "dragon")

    dragon_origin = vector_with_one(0, 0, 0)
    dragon_in_baloons_eyes = H_inv @ dragon_origin
    dist_in_baloons_eyes = math.dist(dragon_in_baloons_eyes[:3], [0, 0, 0])

    assert isinstance(H, np.ndarray)

    baloon_origin = vector_with_one(0, 0, 0)
    baloon_in_dragons_eyes = H @ baloon_origin

    dist_in_dragons_eyes = math.dist(baloon_in_dragons_eyes[:3], [0, 0, 0])
    assert dist_in_dragons_eyes > 75.0

    assert pytest.approx(0) == (dist_in_baloons_eyes - dist_in_dragons_eyes)

    x, y, z = baloon_in_dragons_eyes[:3]
    assert z < 0
    assert y > 40



def test_create_tranformation_graph():

    # T = np.zeros((4,4), dtype=float)
    T = np.identity(4)
    wizard_transforms = [
        Transformation("world", "left_hand", T),
        Transformation("world", "right_hand", T),
        Transformation("right_hand", "magic_stick", T),
        Transformation("left_hand", "book", T),
        Transformation("book", "page_corner", T)
    ]

    g = create_tranformation_graph(wizard_transforms)
    assert isinstance(g, ig.Graph)

    H = get_tranformation_matrix(g, "world", "page_corner")

    assert isinstance(H, np.ndarray)
