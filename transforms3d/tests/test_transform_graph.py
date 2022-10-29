from transforms3d.transform_graph import Transformation, create_tranformation_graph, get_tranformation_matrix
import numpy as np
import igraph as ig


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
