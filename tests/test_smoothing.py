import numpy as np


def test_smooth_along_loop_1d(scloop_utils):
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    smoothed = scloop_utils.smooth_along_loop_1d(values, 1)
    expected = np.array([4.0 / 3.0, 1.0, 2.0, 5.0 / 3.0], dtype=np.float64)
    assert np.allclose(smoothed, expected)


def test_smooth_along_loop_2d(scloop_utils):
    values = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    smoothed = scloop_utils.smooth_along_loop_2d(values, 1)
    expected = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    assert np.allclose(smoothed, expected)
