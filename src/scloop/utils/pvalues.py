# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
from scipy.stats import false_discovery_control

from ..data.types import MultipleTestCorrectionMethod

_PValuesArray = np.ndarray
_PValuesList = list[float]
PValues = _PValuesList | _PValuesArray


@overload
def correct_pvalues(
    pvalues: _PValuesList,
    method: MultipleTestCorrectionMethod | None = "bonferroni",
) -> _PValuesList: ...


@overload
def correct_pvalues(
    pvalues: _PValuesArray,
    method: MultipleTestCorrectionMethod | None = "bonferroni",
) -> _PValuesArray: ...


def correct_pvalues(
    pvalues: PValues,
    method: MultipleTestCorrectionMethod | None = "bonferroni",
) -> PValues:
    if method is None:
        return pvalues

    if isinstance(pvalues, np.ndarray):
        if pvalues.size == 0:
            return pvalues
        original_dtype = pvalues.dtype
        shape = pvalues.shape
        p_flat = np.asarray(pvalues, dtype=float).ravel()
        # NaN marks an untested entry: it stays NaN and is not counted as a test
        tested = ~np.isnan(p_flat)
        p_tested = p_flat[tested]
        corrected = np.full_like(p_flat, np.nan)
        match method:
            case "bonferroni":
                corrected[tested] = np.clip(p_tested * p_tested.size, 0.0, 1.0)
            case "benjamini-hochberg":
                if p_tested.size > 0:
                    corrected[tested] = false_discovery_control(p_tested, method="bh")
            case _:
                raise ValueError(f"{method} unsupported")
        corrected = corrected.reshape(shape)
        if np.issubdtype(original_dtype, np.floating):
            return corrected.astype(original_dtype, copy=False)
        return corrected

    if isinstance(pvalues, Sequence):
        if len(pvalues) == 0:
            return list(pvalues)
        return correct_pvalues(
            np.asarray(list(pvalues), dtype=float), method=method
        ).tolist()

    raise TypeError(f"Unsupported pvalues type: {type(pvalues)!r}")
