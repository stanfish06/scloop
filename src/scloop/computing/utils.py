import multiprocessing
import queue
from typing import Literal

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from ..data.constants import (
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
)
from ..data.types import Count_t, PositiveFloat


def run_eigsh_worker(
    q: multiprocessing.Queue,
    matrix: csr_matrix,
    which: Literal["LM", "SM", "LA", "SA", "BE"],
    k: int,
    tol: float,
    maxiter: int | None,
) -> None:
    try:
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(matrix, k=k, which=which, tol=tol, maxiter=maxiter)  # type: ignore[arg-type]
        q.put(("success", (vals, vecs)))
    except Exception as e:
        q.put(("error", e))


def compute_sparse_eigendecomposition(
    matrix: csr_matrix,
    which: Literal["LM", "SM", "LA", "SA", "BE"],
    n_components: Count_t,
    timeout: PositiveFloat = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    maxiter: Count_t | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
) -> tuple[np.ndarray, np.ndarray] | None:
    tolerances = [1e-6, 1e-5, 1e-4, 1e-3]
    for tol in tolerances:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_eigsh_worker,
            args=(q, matrix, which, n_components, tol, maxiter),
        )
        p.start()
        try:
            status, result = q.get(timeout=timeout)
            p.join()
            if status == "success":
                eigenvalues, eigenvectors = result
                match which:
                    case "LM":
                        sort_idx = np.argsort(-np.abs(eigenvalues))
                    case "SM":
                        sort_idx = np.argsort(np.abs(eigenvalues))
                    case "LA":
                        sort_idx = np.argsort(-eigenvalues)
                    case "SA":
                        sort_idx = np.argsort(eigenvalues)
                    case _:
                        sort_idx = np.arange(len(eigenvalues))
                return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
            else:
                logger.warning(
                    f"Eigendecomposition failed with tol={tol}: {result}. Retrying..."
                )
                continue
        except queue.Empty:
            logger.warning(
                f"Eigendecomposition timed out ({timeout}s) with tol={tol}. "
                "Retrying with looser tolerance."
            )
            p.terminate()
            p.join()
            q.close()
            continue
        except Exception as e:
            logger.warning(
                f"Eigendecomposition failed with tol={tol}: {e}. Retrying..."
            )
            if p.is_alive():
                p.terminate()
                p.join()
            continue
    logger.error("Eigendecomposition failed after all retries.")
    return None
