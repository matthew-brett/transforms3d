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

    unique_frames_map = {i: f for f, i in enumerate(list(unique_frames))}
    unique_frames_map_reverse = {v: k for k, v in unique_frames_map.items()}

    tf_pairs_as_ids = set([(unique_frames_map[tf.source_frame], unique_frames_map[tf.target_frame]) \
        for tf in transformations])

    tf_graph = ig.Graph(len(unique_frames), list(tf_pairs_as_ids))
    tf_graph.vs["name"] = [unique_frames_map_reverse[i] for i in range(len(unique_frames))]

    return tf_graph


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
