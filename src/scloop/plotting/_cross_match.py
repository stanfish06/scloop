# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Literal

import glasbey
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from pydantic import ConfigDict, validate_call

from ..data.constants import (
    CROSS_MATCH_KEY,
    CROSS_MATCH_RESULT_KEY,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
)
from ..data.types import Index_t, PositiveFloat
from ..matching import CrossDatasetMatcher
from ._utils import _create_figure_standard, _get_embedding_key

__all__ = ["match_loops_overlay"]


def _plot_thumbnail_view(
    adata_combined: AnnData,
    track_ids: list[Index_t] | Index_t | None,
    dataset_ids: list[Index_t] | None,
    key_matching: str,
    components: tuple[Index_t, Index_t] | list[Index_t],
    components_thumbnails: dict[Index_t, tuple[Index_t, Index_t]]
    | tuple[Index_t, Index_t]
    | None,
    color_by: Literal["dataset", "track"],
    include_bootstrap_loops: bool,
    show_loop_vertices: bool,
    pointsize: PositiveFloat,
    figsize: tuple[PositiveFloat, PositiveFloat],
    dpi: PositiveFloat,
    kwargs_figure: dict | None,
    kwargs_axes: dict | None,
    kwargs_layout: dict | None,
    kwargs_scatter: dict | None,
    title_main: str | None,
    title_thumbnails: dict[Index_t, str] | str | None,
    ax: Axes | None = None,
) -> Axes:
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    matcher: CrossDatasetMatcher = adata_combined.uns[key_matching]

    if len(components) != 2:
        raise ValueError("components must contain exactly two entries.")

    if isinstance(track_ids, int):
        track_ids = [track_ids]
    track_ids = track_ids or []

    n_datasets = len(matcher.adata_list)
    dataset_ids = dataset_ids or list(range(n_datasets))
    n_datasets_plot = len(dataset_ids)

    assert matcher.loop_matching_result is not None
    assert matcher.loop_matching_result.tracks is not None

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, **(kwargs_figure or {}))
        height_ratios = [3, 1]
        gs = GridSpec(
            2,
            n_datasets_plot,
            figure=fig,
            height_ratios=height_ratios,
            hspace=0.3,
            wspace=0.2,
        )
    else:
        fig = ax.figure
        gs = GridSpecFromSubplotSpec(
            2,
            n_datasets_plot,
            subplot_spec=ax.get_subplotspec(),
            height_ratios=[3, 1],
            hspace=0.3,
            wspace=0.2,
        )
        ax.remove()

    ax_main = fig.add_subplot(gs[0, :])

    if kwargs_axes:
        ax_main.set(**kwargs_axes)

    if title_main:
        ax_main.set_title(title_main)

    _plot_thumbnail_main_panel(
        ax=ax_main,
        matcher=matcher,
        track_ids=track_ids,
        dataset_ids=dataset_ids,
        components=components,
        color_by=color_by,
        pointsize=pointsize,
        include_bootstrap_loops=include_bootstrap_loops,
        show_loop_vertices=show_loop_vertices,
        kwargs_scatter=kwargs_scatter,
    )

    for idx, ds_id in enumerate(dataset_ids):
        ax_thumb = fig.add_subplot(gs[1, idx])
        if kwargs_axes:
            ax_thumb.set(**kwargs_axes)

        if title_thumbnails is not None:
            if isinstance(title_thumbnails, dict):
                thumb_title = title_thumbnails.get(ds_id, None)
            else:
                thumb_title = title_thumbnails
            if thumb_title:
                ax_thumb.set_title(thumb_title)

        if components_thumbnails is None:
            thumb_components = components
        elif isinstance(components_thumbnails, dict):
            thumb_components = components_thumbnails.get(ds_id, components)
        else:
            thumb_components = components_thumbnails

        _plot_thumbnail_dataset(
            ax=ax_thumb,
            matcher=matcher,
            dataset_idx=ds_id,
            track_ids=track_ids,
            components=thumb_components,
            color_by=color_by,
            pointsize=pointsize,
            include_bootstrap_loops=include_bootstrap_loops,
            show_loop_vertices=show_loop_vertices,
            kwargs_scatter=kwargs_scatter,
        )

    return ax_main


def _plot_thumbnail_main_panel(
    ax: Axes,
    matcher: CrossDatasetMatcher,
    track_ids: list[Index_t],
    dataset_ids: list[Index_t],
    components: tuple[Index_t, Index_t] | list[Index_t],
    color_by: Literal["dataset", "track"],
    pointsize: PositiveFloat,
    include_bootstrap_loops: bool,
    show_loop_vertices: bool,
    kwargs_scatter: dict | None,
) -> None:
    loop_matching_result = matcher.loop_matching_result
    assert loop_matching_result is not None
    tracks = loop_matching_result.tracks
    assert tracks is not None

    block_size = 5
    dataset_to_color_idx = {ds_id: i for i, ds_id in enumerate(dataset_ids)}
    track_to_color_idx = {track_id: i for i, track_id in enumerate(range(len(tracks)))}

    cmap_dataset_split = []
    cmap_track = []
    if color_by == "dataset":
        cmap_dataset = glasbey.create_block_palette(
            block_sizes=[block_size] * len(dataset_ids)
        )
        cmap_dataset_split = [
            cmap_dataset[i : i + block_size]
            for i in range(0, len(cmap_dataset), block_size)
        ]
    else:
        n_tracks = len(tracks)
        cmap_track = glasbey.create_palette(palette_size=n_tracks)

    for ds_idx in dataset_ids:
        ds_adata = matcher.adata_list[ds_idx]
        emb = ds_adata.obsm[CROSS_MATCH_KEY]
        assert emb is not None
        assert type(emb) is np.ndarray

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

    for track_id in track_ids:
        if track_id >= len(tracks):
            continue

        track = tracks[track_id]

        for loop_class_idx in track:
            ds_idx = loop_class_idx.idx_dataset
            if ds_idx not in dataset_ids:
                continue

            loop_cls_idx = loop_class_idx.idx_loop_class
            ds_adata = matcher.adata_list[ds_idx]
            hd = matcher.homology_data_list[ds_idx]

            if loop_cls_idx >= len(hd.selected_loop_classes):
                continue

            emb = ds_adata.obsm[CROSS_MATCH_KEY]
            assert emb is not None
            assert type(emb) is np.ndarray

            try:
                loops_plot = hd._get_loop_embedding(
                    selector=loop_cls_idx,
                    embedding_alt=emb,
                    include_bootstrap=include_bootstrap_loops,
                )
            except AssertionError:
                continue

            color: tuple[float, float, float] | str
            if color_by == "dataset":
                color_block_idx = dataset_to_color_idx[ds_idx]
                color_block = cmap_dataset_split[color_block_idx]
            else:
                color_block = []

            for j, loop in enumerate(loops_plot):
                loop = np.array(loop)
                if color_by == "dataset":
                    color = color_block[j % block_size]  # type: ignore[index]
                else:
                    color = cmap_track[track_to_color_idx[track_id]]

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


def _plot_thumbnail_dataset(
    ax: Axes,
    matcher: CrossDatasetMatcher,
    dataset_idx: Index_t,
    track_ids: list[Index_t],
    components: tuple[Index_t, Index_t] | list[Index_t],
    color_by: Literal["dataset", "track"],
    pointsize: PositiveFloat,
    include_bootstrap_loops: bool,
    show_loop_vertices: bool,
    kwargs_scatter: dict | None,
) -> None:
    loop_matching_result = matcher.loop_matching_result
    assert loop_matching_result is not None
    tracks = loop_matching_result.tracks
    assert tracks is not None

    ds_adata = matcher.adata_list[dataset_idx]
    hd = matcher.homology_data_list[dataset_idx]

    if hd.meta.preprocess is None or hd.meta.preprocess.embedding_method is None:
        return

    emb_key = _get_embedding_key(hd.meta.preprocess.embedding_method)
    if emb_key not in ds_adata.obsm:
        return

    emb = ds_adata.obsm[emb_key]
    assert type(emb) is np.ndarray

    if hd.meta.preprocess.indices_downsample:
        vertex_indices = hd.meta.preprocess.indices_downsample
    else:
        vertex_indices = list(range(emb.shape[0]))

    emb_background = emb[vertex_indices]
    ax.scatter(
        emb_background[:, components[0]],
        emb_background[:, components[1]],
        color="lightgray",
        s=pointsize * 0.5,
        **(kwargs_scatter or {}),
    )

    cmap_track = None
    if color_by == "track":
        n_tracks = len(tracks)
        cmap_track = glasbey.create_palette(palette_size=n_tracks)

    block_size = 5
    color_block = [
        (0.8, 0.8, 0.8),
        (0.6, 0.6, 0.6),
        (0.4, 0.4, 0.4),
        (0.2, 0.2, 0.2),
        (0.0, 0.0, 0.0),
    ]

    for track_id in track_ids:
        if track_id >= len(tracks):
            continue

        track = tracks[track_id]

        for loop_class_idx in track:
            if loop_class_idx.idx_dataset != dataset_idx:
                continue

            loop_cls_idx = loop_class_idx.idx_loop_class
            if loop_cls_idx >= len(hd.selected_loop_classes):
                continue

            try:
                loops_plot = hd._get_loop_embedding(
                    selector=loop_cls_idx,
                    embedding_alt=emb,
                    include_bootstrap=include_bootstrap_loops,
                )
            except AssertionError:
                continue

            for j, loop in enumerate(loops_plot):
                loop = np.array(loop)
                if color_by == "track":
                    assert cmap_track is not None
                    color = cmap_track[track_id]
                else:
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
                        s=pointsize * 0.5,
                        **(kwargs_scatter or {}),
                    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def match_loops_overlay(
    adata_combined: AnnData,
    track_ids: list[Index_t] | Index_t | None = None,
    dataset_ids: list[Index_t] | None = None,
    embedding: Literal["joint", "separate", "both"] = "joint",
    key_matching: str = CROSS_MATCH_RESULT_KEY,
    components: tuple[Index_t, Index_t] | list[Index_t] = (0, 1),
    components_thumbnails: dict[Index_t, tuple[Index_t, Index_t]]
    | tuple[Index_t, Index_t]
    | None = None,
    ax: Axes | None = None,
    *,
    color_by: Literal["dataset", "track"] = "dataset",
    include_bootstrap_loops: bool = True,
    show_loop_vertices: bool = True,
    pointsize: PositiveFloat = 1,
    figsize: tuple[PositiveFloat, PositiveFloat] = DEFAULT_FIGSIZE,
    dpi: PositiveFloat = DEFAULT_DPI,
    title_main: str | None = None,
    title_thumbnails: dict[Index_t, str] | str | None = None,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    kwargs_scatter: dict | None = None,
) -> Axes:
    if embedding == "both":
        return _plot_thumbnail_view(
            adata_combined=adata_combined,
            track_ids=track_ids,
            dataset_ids=dataset_ids,
            key_matching=key_matching,
            components=components,
            components_thumbnails=components_thumbnails,
            color_by=color_by,
            include_bootstrap_loops=include_bootstrap_loops,
            show_loop_vertices=show_loop_vertices,
            pointsize=pointsize,
            figsize=figsize,
            dpi=dpi,
            kwargs_figure=kwargs_figure,
            kwargs_axes=kwargs_axes,
            kwargs_layout=kwargs_layout,
            kwargs_scatter=kwargs_scatter,
            title_main=title_main,
            title_thumbnails=title_thumbnails,
            ax=ax,
        )

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
