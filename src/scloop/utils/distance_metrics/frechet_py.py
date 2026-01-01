from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ...data.constants import DEFAULT_N_MAX_WORKERS
from .frechet import compute_loop_frechet


def compute_pairwise_loop_frechet(
    loop_set_a: list[list[list[float]]] | list[np.ndarray],
    loop_set_b: list[list[list[float]]] | list[np.ndarray],
    n_workers: int = DEFAULT_N_MAX_WORKERS,
) -> np.ndarray:
    loop_set_a_arr = [
        np.ascontiguousarray(loop, dtype=np.float64) for loop in loop_set_a
    ]
    loop_set_b_arr = [
        np.ascontiguousarray(loop, dtype=np.float64) for loop in loop_set_b
    ]

    n_a = len(loop_set_a_arr)
    n_b = len(loop_set_b_arr)
    n_total = n_a * n_b

    results = np.empty(n_total, dtype=np.float64)

    if n_workers == 1:
        idx = 0
        for a in loop_set_a_arr:
            for b in loop_set_b_arr:
                results[idx] = compute_loop_frechet(a, b)
                idx += 1
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = {}
            idx = 0
            for a in loop_set_a_arr:
                for b in loop_set_b_arr:
                    task = executor.submit(compute_loop_frechet, a, b)
                    tasks[task] = idx
                    idx += 1

            for task in as_completed(tasks):
                idx = tasks[task]
                results[idx] = task.result()

    return results
