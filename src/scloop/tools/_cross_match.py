import pandas as pd
from anndata import AnnData

from ..data.types import Index_t
from ..matching import CrossDatasetMatcher

__all__ = ["match_loops"]


def match_loops(
    adatas: list[AnnData],
    shared_embedding_key: str,
    reference_idx: Index_t = 0,
    model_type: str = "mlp",
    distance_method: str = "hausdorff",
    n_permutations: int = 1000,
    return_matcher: bool = False,
    **model_kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, CrossDatasetMatcher]:
    return pd.DataFrame()
