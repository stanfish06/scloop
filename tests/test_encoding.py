import pytest


@pytest.mark.parametrize(
    "num_vertices,i,j",
    [
        (10, 0, 1),
        (10, 3, 7),
        (10, 9, 2),
    ],
)
def test_edge_idx_roundtrip(scloop_utils, num_vertices, i, j):
    edge_id = scloop_utils.edge_idx_encode(i, j, num_vertices)
    u, v = scloop_utils.edge_idx_decode(edge_id, num_vertices)
    assert (u, v) == (min(i, j), max(i, j))


@pytest.mark.parametrize(
    "num_vertices,i,j,k",
    [
        (7, 0, 1, 2),
        (7, 5, 3, 1),
        (7, 6, 4, 2),
    ],
)
def test_triangle_idx_roundtrip(scloop_utils, num_vertices, i, j, k):
    tri_id = scloop_utils.triangle_idx_encode(i, j, k, num_vertices)
    u, v, w = scloop_utils.triangle_idx_decode(tri_id, num_vertices)
    assert (u, v, w) == tuple(sorted([i, j, k]))
