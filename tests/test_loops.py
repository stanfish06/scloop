import numpy as np


def test_loop_vertices_to_edge_ids_with_signs(scloop_utils):
    num_vertices = 10
    loop = np.array([0, 1, 2, 0], dtype=np.int64)
    edge_ids, signs = scloop_utils.loop_vertices_to_edge_ids_with_signs(
        loop, num_vertices
    )
    expected_ids = np.array(
        [0 * num_vertices + 1, 1 * num_vertices + 2, 0 * num_vertices + 2],
        dtype=np.int64,
    )
    expected_signs = np.array([1, 1, -1], dtype=np.int8)
    assert np.array_equal(edge_ids, expected_ids)
    assert np.array_equal(signs, expected_signs)


def test_loops_to_coords_closure(scloop_utils):
    embedding = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    loops = [[0, 1, 2]]
    coords = scloop_utils.loops_to_coords(embedding, loops)
    assert len(coords[0]) == 4
    assert coords[0][0] == coords[0][-1]
    loops_closed = [[0, 1, 0]]
    coords_closed = scloop_utils.loops_to_coords(embedding, loops_closed)
    assert len(coords_closed[0]) == 3
    assert coords_closed[0][0] == coords_closed[0][-1]
