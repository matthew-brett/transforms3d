from dataclasses import dataclass
from typing import List
from itertools import tee

import numpy as np
import igraph as ig

@dataclass
class Transformation:
    source_frame: str
    target_frame: str
    A: np.ndarray  # 4x4 matrix for 3D affine trafo


def create_tranformation_graph(transformations: List[Transformation]):

    unique_frames = set([tf.source_frame for tf in transformations] + [tf.target_frame for tf in transformations])
    unique_frames = list(unique_frames)

    unique_frames_map = {i: frame for i, frame in enumerate(unique_frames)}
    unique_frames_map_reverse = {frame: i for i, frame in unique_frames_map.items()}

    transformations_inv = invert_transformations(transformations)
    transformations_all = transformations_inv + transformations

    tf_pairs_as_ids = [(unique_frames_map_reverse[tf.source_frame], unique_frames_map_reverse[tf.target_frame]) \
        for tf in transformations_all]

    # TODO check for duplicates

    tf_graph = ig.Graph(len(unique_frames), tf_pairs_as_ids, directed=True)
    tf_graph.vs["name"] = unique_frames

    tf_graph.es["A"] = [tf.A for tf in transformations_all]

    return tf_graph


def get_tranformation_matrix(tf_graph: ig.Graph, source: str, target: str) -> List[int]:

    vertices_on_path = tf_graph.get_shortest_paths(source, to=target, mode="out", output='vpath')[0]
    
    H = np.identity(4)
    for e0, e1 in pairwise(vertices_on_path):
        edge = tf_graph.es.find(_source=e0, _target=e1)
        H = H @ edge["A"]
    return H


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def invert_transformations(transformations: List[Transformation]) -> List[Transformation]:

    inverted_transformations = list()
    for tf in transformations:
        tf_inv = _invert_tf(tf)
        inverted_transformations.append(tf_inv)
    return inverted_transformations

def _invert_tf(tf: Transformation) -> Transformation:

    # we could use decompose (?)
    R = tf.A[:3, :3]
    p = tf.A[:3, 3]

    R_inv = np.linalg.inv(R)

    A_inv = np.identity(4, dtype=float)
    A_inv[:3, :3] = R_inv
    A_inv[:3, 3] = - R_inv @ p

    return Transformation(tf.target_frame, tf.source_frame, A_inv)
