# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from numba import jit
from anndata import AnnData
from ..data.types import IndexListDownSample
from pydantic import validate_call
from typing import TypeAlias
import numpy as np

@jit
def _sample_impl(data: np.ndarray, n: int) -> np.ndarray:
    return np.array([1, 1])

@validate_call(config={"arbitrary_types_allowed": True})
def sample(adata: AnnData, n: int) -> IndexListDownSample:
    return _sample_impl(np.array(), n).tolist()
