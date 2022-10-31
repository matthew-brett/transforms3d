from transforms3d.transform_graph import Transformation, create_tranformation_graph, get_tranformation_matrix
import numpy as np
import igraph as ig
import math
import pytest

from transforms3d.euler import euler2mat
from transforms3d.affines import compose


def test_create_tranformation_graph():

    world_to_balloon = Transformation("world", "balloon",
        define_ros_transform(50, 50, 50, 0, 0, 0))
    world_to_drone = Transformation("world", "drone",
        define_ros_transform(-50, 0, 50, 3.0, -3.0, 180, deg=True))

    simple_world = [world_to_balloon, world_to_drone]

    g = create_tranformation_graph(simple_world)
    assert isinstance(g, ig.Graph)

    H = get_tranformation_matrix(g, "world", "drone")

    assert isinstance(H, np.ndarray)


def test_simple_world():

    world_to_balloon = Transformation("world", "balloon",
        define_ros_transform(50, 50, 50, 0, 0, 0))
    world_to_drone = Transformation("world", "drone",
        define_ros_transform(-50, 0, 50, 3.0, -3.0, 180, deg=True))

    simple_world_transforms = [
        world_to_balloon,
        world_to_drone,
    ]

    g = create_tranformation_graph(simple_world_transforms)
    assert isinstance(g, ig.Graph)

    H = get_tranformation_matrix(g, "drone", "balloon")

    H_inv = get_tranformation_matrix(g, "balloon", "drone")

    drone_origin = vector4(0, 0, 0)
    drone_in_balloons_eyes = H_inv @ drone_origin
    dist_in_balloons_eyes = math.dist(drone_in_balloons_eyes[:3], [0, 0, 0])

    assert isinstance(H, np.ndarray)

    balloon_origin = vector4(0, 0, 0)
    balloon_in_drones_eyes = H @ balloon_origin

    dist_in_drones_eyes = math.dist(balloon_in_drones_eyes[:3], [0, 0, 0])
    assert dist_in_drones_eyes > 75.0

    assert pytest.approx(0) == (dist_in_balloons_eyes - dist_in_drones_eyes)

    x, y, z = balloon_in_drones_eyes[:3]
    assert z < 0
    assert y < -40
    assert x < -75.0


def define_ros_transform(x, y, z, roll, pitch, yaw, deg: bool = False):
    # helper function to create a transformation matrix from
    # - translation
    # - RPY rotation

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


def vector4(x, y, z):
    v = np.ones((4))
    v[:3] = [x, y ,z]
    return v