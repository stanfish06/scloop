# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Literal

import glasbey
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.constants import CROSS_MATCH_KEY, CROSS_MATCH_RESULT_KEY
from ..data.types import Index_t, PositiveFloat
from ..matching import CrossDatasetMatcher
from ._utils import _create_figure_standard, _get_embedding_key

__all__ = ["match_loops_overlay"]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def match_loops_overlay(
    adata_combined: AnnData,
    track_ids: list[Index_t] | Index_t | None = None,
    dataset_ids: list[Index_t] | None = None,
    embedding: Literal["joint", "separate", "both"] = "joint",
    key_matching: str = CROSS_MATCH_RESULT_KEY,
    components: tuple[Index_t, Index_t] | list[Index_t] = (0, 1),
    ax: Axes | None = None,
    *,
    include_bootstrap_loops: bool = True,
    show_loop_vertices: bool = True,
    pointsize: PositiveFloat = 1,
    figsize: tuple[PositiveFloat, PositiveFloat] = (5, 5),
    dpi: PositiveFloat = 300,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
) -> Axes:
    if embedding == "both":
        raise NotImplementedError("Thumbnail view (both) not yet implemented")

    matcher: CrossDatasetMatcher = adata_combined.uns[key_matching]

    if len(components) != 2:
        raise ValueError("components must contain exactly two entries.")

    if isinstance(track_ids, int):
        track_ids = [track_ids]
    track_ids = track_ids or []

    n_datasets = len(matcher.adata_list)
    dataset_ids = dataset_ids or list(range(n_datasets))

    n_datasets_plot = len(dataset_ids)
    if n_datasets_plot > 0:
        block_size = 5
        cmap = glasbey.create_block_palette(block_sizes=[block_size] * n_datasets_plot)
        cmap = [cmap[i : i + block_size] for i in range(0, len(cmap), block_size)]
    else:
        block_size = 0
        cmap = []

    ax = (
        _create_figure_standard(
            figsize=figsize,
            dpi=dpi,
            kwargs_figure=kwargs_figure,
            kwargs_axes=kwargs_axes,
            kwargs_layout=kwargs_layout,
        )
        if ax is None
        else ax
    )

    for ds_idx in dataset_ids:
        ds_adata = matcher.adata_list[ds_idx]
        if embedding == "joint":
            emb = ds_adata.obsm[CROSS_MATCH_KEY]
        else:
            hd = matcher.homology_data_list[ds_idx]
            if (
                hd.meta.preprocess is None
                or hd.meta.preprocess.embedding_method is None
            ):
                raise ValueError(
                    f"Dataset {ds_idx} missing preprocessing metadata. "
                    "Run scl.pp.prepare_adata() first."
                )
            emb_key = _get_embedding_key(hd.meta.preprocess.embedding_method)
            if emb_key not in ds_adata.obsm:
                raise KeyError(f"Embedding {emb_key} not found in dataset {ds_idx}")
            emb = ds_adata.obsm[emb_key]

        hd = matcher.homology_data_list[ds_idx]
        if hd.meta.preprocess is not None and hd.meta.preprocess.indices_downsample:
            vertex_indices = hd.meta.preprocess.indices_downsample
        else:
            vertex_indices = list(range(emb.shape[0]))
        emb_background = emb[vertex_indices]

        ax.scatter(
            emb_background[:, components[0]],
            emb_background[:, components[1]],
            color="lightgray",
            s=pointsize,
            **(kwargs_scatter or {}),
        )

    assert matcher.loop_matching_result is not None
    assert matcher.loop_matching_result.tracks is not None

    dataset_to_color_idx = {ds_id: i for i, ds_id in enumerate(dataset_ids)}

    for track_id in track_ids:
        if track_id >= len(matcher.loop_matching_result.tracks):
            continue

        track = matcher.loop_matching_result.tracks[track_id]

        for loop_class_idx in track:
            ds_idx = loop_class_idx.idx_dataset
            if ds_idx not in dataset_ids:
                continue

            loop_cls_idx = loop_class_idx.idx_loop_class
            ds_adata = matcher.adata_list[ds_idx]
            hd = matcher.homology_data_list[ds_idx]

            if loop_cls_idx >= len(hd.selected_loop_classes):
                continue

            if embedding == "joint":
                emb = ds_adata.obsm[CROSS_MATCH_KEY]
            else:
                if (
                    hd.meta.preprocess is None
                    or hd.meta.preprocess.embedding_method is None
                ):
                    continue
                emb_key = _get_embedding_key(hd.meta.preprocess.embedding_method)
                if emb_key not in ds_adata.obsm:
                    continue
                emb = ds_adata.obsm[emb_key]

            try:
                assert type(emb) is np.ndarray
                loops_plot = hd._get_loop_embedding(
                    selector=loop_cls_idx,
                    embedding_alt=emb,
                    include_bootstrap=include_bootstrap_loops,
                )
            except AssertionError:
                continue

            color_block_idx = dataset_to_color_idx[ds_idx]
            color_block = cmap[color_block_idx]

            for j, loop in enumerate(loops_plot):
                loop = np.array(loop)
                color = color_block[j % block_size]

                ax.plot(
                    loop[:, components[0]],
                    loop[:, components[1]],
                    color=color,
                    **(kwargs_scatter or {}),
                )

                if show_loop_vertices:
                    ax.scatter(
                        loop[:, components[0]],
                        loop[:, components[1]],
                        color=color,
                        s=pointsize,
                        **(kwargs_scatter or {}),
                    )

    return ax
