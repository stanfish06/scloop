# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Literal

import anndata as ad
import pandas as pd
from anndata import AnnData
from loguru import logger

from ..data.constants import (
    CROSS_MATCH_RESULT_KEY,
    DEFAULT_LOOP_DIST_METHOD,
    DEFAULT_N_PERMUTATIONS,
)
from ..data.metadata import CrossDatasetMatchingMeta
from ..data.types import Count_t, CrossMatchModelTypes, Index_t, LoopDistMethod
from ..matching import CrossDatasetMatcher

__all__ = ["match_loops"]


def match_loops(
    adata_list: list[AnnData],
    reference_adata_index: Index_t,
    reference_embedding_key: str,
    shared_embedding_key: str,
    model_type: CrossMatchModelTypes = "nf",
    distance_method: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
    n_permutations: Count_t = DEFAULT_N_PERMUTATIONS,
    reembed_method: Literal["diffmap", "umap", "none"] = "none",
    n_comps_reembed: int = 15,
    n_neighbors_reembed: int = 15,
    kwargs_model: dict | None = None,
    kwargs_match: dict | None = None,
) -> tuple[AnnData, pd.DataFrame, pd.DataFrame]:
    if kwargs_model is None:
        kwargs_model = {}
    if kwargs_match is None:
        kwargs_match = {}

    reference_embedding_keys = [reference_embedding_key] * len(adata_list)

    meta = CrossDatasetMatchingMeta(
        shared_embedding_key=shared_embedding_key,
        reference_embedding_keys=reference_embedding_keys,
        reference_idx=reference_adata_index,
        model_type=model_type,
    )

    matcher = CrossDatasetMatcher(adata_list=adata_list, meta=meta)

    matcher._train_reference_mapping(**kwargs_model)

    matcher._transform_all_to_reference()

    if reembed_method != "none":
        matcher._compute_joint_reembedding(
            n_neighbors=n_neighbors_reembed,
            n_comps=n_comps_reembed,
            reembed_method=reembed_method,  # type: ignore[arg-type]
        )

    for i in range(len(adata_list)):
        for j in range(i + 1, len(adata_list)):
            matcher._loops_cross_match(
                n_permute=n_permutations,
                source_dataset_idx=i,
                target_dataset_idx=j,
                method=distance_method,
                **kwargs_match,
            )

    assert matcher.loop_matching_result is not None
    matcher.loop_matching_result._compute_tracks()

    tracks_df, matches_df = matcher.loop_matching_result._to_dataframe()

    gene_sets = [set(adata.var_names) for adata in adata_list]
    common_genes = set.intersection(*gene_sets)
    all_genes = set.union(*gene_sets)

    if len(common_genes) < len(all_genes):
        logger.warning(
            f"Gene name mismatch: {len(common_genes)} common genes out of {len(all_genes)} total. "
            "Concatenating with join='inner'"
        )

    concat_adata = ad.concat(
        adata_list,
        axis=0,
        join="inner",
        label="dataset",
        keys=[str(i) for i in range(len(adata_list))],
        index_unique="_",
    )

    concat_adata.uns[CROSS_MATCH_RESULT_KEY] = matcher

    return concat_adata, tracks_df, matches_df
