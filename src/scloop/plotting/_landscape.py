# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import cast

import glasbey
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, Normalize, to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pydantic import ConfigDict, validate_call
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

from ..data.constants import (
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    SCLOOP_MEMBERSHIP_OBSM_KEY,
    SCLOOP_UNS_KEY,
)
from ..data.types import Index_t, Percent_t, PositiveFloat
from ._homology import _loop_plane
from ._utils import _get_homology_data, savefig_or_show

Cmap = str | list[str] | dict[str, str]


def _is_categorical(col: pd.Series) -> bool:
    return isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object


def _palette_rgb(adata: AnnData, key: str, categories: list, cmap: Cmap) -> np.ndarray:
    if isinstance(cmap, dict):
        missing = [c for c in categories if c not in cmap]
        if missing:
            raise KeyError(f"cmap has no color for categories {missing} of {key}.")
        colors = [cmap[c] for c in categories]
    elif isinstance(cmap, list):
        colors = cmap
    elif f"{key}_colors" in adata.uns:
        colors = list(adata.uns[f"{key}_colors"])
    else:
        colors = glasbey.create_palette(palette_size=len(categories))
    return np.array([to_rgb(c) for c in colors])


def _key_rows(key: list[Patch] | tuple[float, float] | None) -> float:
    # key height in text lines, used to size its row in the key column
    if key is None:
        return 0.0
    return 3.0 if isinstance(key, tuple) else 1.4 * (len(key) + 1)


def _colormap(cmap: Cmap, key: str) -> str:
    if not isinstance(cmap, str):
        raise ValueError(
            f"{key} is numeric; cmap must be a colormap name, not a list or dict."
        )
    return cmap


def _add_key(
    ax: Axes,
    key: list[Patch] | tuple[float, float],
    cmap: Cmap,
    label: str,
    slot: Axes | float,
) -> None:
    # slot is a key-column axes on figures we create, else a y anchor beside ax
    if isinstance(slot, Axes):
        slot.axis("off")
        host, anchor, loc = slot, (0.0, 1.0), "upper left"
        cax_rect = (0.3, 0.35, 0.35, 0.2)
    else:
        host, anchor, loc = ax, (1.0, slot), "center left"
        cax_rect = (1.12, slot, 0.15, 0.025)
    if isinstance(key, list):
        legend = host.legend(
            handles=key,
            loc=loc,
            bbox_to_anchor=anchor,
            fontsize=7,
            frameon=False,
            title=label,
            title_fontsize=7,
            alignment="left",
        )
        host.add_artist(legend)
    else:
        cax = host.inset_axes(cax_rect)
        mappable = plt.cm.ScalarMappable(
            cmap=_colormap(cmap, label), norm=Normalize(*key)
        )
        colorbar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
        colorbar.set_ticks([])
        # min and max beside the bar so they cannot overlap on a short bar
        cax.text(
            -0.06,
            0.5,
            f"{key[0]:.3g}",
            ha="right",
            va="center",
            fontsize=7,
            transform=cax.transAxes,
        )
        cax.text(
            1.06,
            0.5,
            f"{key[1]:.3g}",
            ha="left",
            va="center",
            fontsize=7,
            transform=cax.transAxes,
        )
        cax.set_title(label, fontsize=7)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def loop_landscape(
    adata: AnnData,
    basis: str,
    track_id: Index_t | tuple[Index_t, Index_t],
    key_homology: str = SCLOOP_UNS_KEY,
    color: str | None = None,
    cmap: Cmap = "Spectral_r",
    ax: Axes | None = None,
    *,
    show_points: bool = True,
    color_points: str | None = None,
    cmap_points: Cmap = "Spectral_r",
    size_points: PositiveFloat = 2,
    alpha_points: Percent_t = 0.8,
    add_outline: bool = True,
    outline_width: tuple[PositiveFloat, PositiveFloat] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    show_reps: bool = True,
    show_legend: bool = True,
    grid_size: int = 80,
    smooth_bins: PositiveFloat = 1.5,
    min_density: PositiveFloat = 0.05,
    vert_exag: PositiveFloat = 2.0,
    zmin: float = 0.0,
    zmax: float = 1.0,
    elev: float = 65,
    azim: float = 90,
    figsize: tuple[PositiveFloat, PositiveFloat] = DEFAULT_FIGSIZE,
    dpi: PositiveFloat = DEFAULT_DPI,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
) -> Axes | None:
    """3D landscape of a loop's membership over its loop plane.

    Height is ``sqrt(membership)`` binned on a grid over the loop plane and smoothed;
    the surface falls to 0 over regions without cells. The surface and the cells
    drawn on it are colored independently.

    Parameters
    ----------
    basis : str
        Embedding in adata.obsm the loop plane is fitted in.
    track_id : int | tuple[int, int]
        Track whose membership and plane are used. ``(track, i)`` draws only the
        i-th loop of the track, indexed over the reference and matched bootstrap
        loops as in ``loops``; an int draws the reference representatives.
    add_outline, outline_width, outline_color
        Outline points as in scanpy: a background ring of ``outline_width[0]`` and a
        gap ring of ``outline_width[1]``, relative to the point diameter.
    color : str | None
        Surface color: None for height, or an adata.obs key. Categorical keys blend
        category colors by per-bin category proportions; numeric keys use ``cmap``.
    cmap : str | list | dict
        Colormap name for height and numeric keys. For a categorical ``color``, a
        dict of category to color or a list in category order; with a name,
        categories use adata.uns[f"{color}_colors"], then glasbey.
    color_points, cmap_points
        Same as ``color`` and ``cmap`` for the cells drawn on the surface.
    grid_size : int
        Number of bins per side of the grid.
    smooth_bins : float
        Gaussian smoothing width in bins.
    show_legend : bool
        Draw the surface and point keys; False also drops the key column.
    min_density : float
        Weight of the pull towards 0, relative to the median smoothed cell count;
        larger values flatten the surface faster away from cells.
    zmin, zmax : float
        Range the surface height is clipped to; also the range of the height
        colormap and its colorbar.
    """
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin.")
    data = _get_homology_data(adata, key_homology)
    if SCLOOP_MEMBERSHIP_OBSM_KEY not in adata.obsm:
        raise KeyError(
            f"{SCLOOP_MEMBERSHIP_OBSM_KEY} not found in adata.obsm. Run find_loops first."
        )
    if basis in adata.obsm:
        emb = np.asarray(adata.obsm[basis])
    elif f"X_{basis}" in adata.obsm:
        emb = np.asarray(adata.obsm[f"X_{basis}"])
    else:
        raise KeyError(f"Embedding {basis} not found in adata.obsm.")

    track, idx_loop = track_id if isinstance(track_id, tuple) else (track_id, None)
    plane, _ = _loop_plane(data, emb, [track], 2)
    emb_mean = emb.mean(axis=0, keepdims=True)
    xy = (emb - emb_mean) @ plane
    height = np.sqrt(np.asarray(adata.obsm[SCLOOP_MEMBERSHIP_OBSM_KEY])[:, track])

    pad = 0.15 * (xy.max(axis=0) - xy.min(axis=0))
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    x_edges = np.linspace(lo[0], hi[0], grid_size + 1)
    y_edges = np.linspace(lo[1], hi[1], grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    bins = [x_edges, y_edges]

    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    median = binned_statistic_2d(
        xy[:, 0], xy[:, 1], height, statistic="median", bins=bins
    ).statistic
    num = gaussian_filter(np.nan_to_num(median) * count, smooth_bins)
    den = gaussian_filter(count, smooth_bins)
    # lam pulls the surface to 0 where there are no cells, so it extends over empty regions
    lam = min_density * np.median(den[den > 0])
    z = np.clip(num / (den + lam), zmin, zmax)
    support = (den / (den + lam))[..., None]
    floor_rgb = np.array(to_rgb("lightgray"))

    key_surface: list[Patch] | tuple[float, float]
    if color is None:
        rgb = plt.get_cmap(_colormap(cmap, "height"))((z - zmin) / (zmax - zmin))[
            ..., :3
        ]
        key_surface = (zmin, zmax)
    elif _is_categorical(adata.obs[color]):
        col = adata.obs[color].astype("category")
        categories = list(col.cat.categories)
        pal_rgb = _palette_rgb(adata, color, categories, cmap)
        codes = col.cat.codes.to_numpy()
        proportions = np.stack(
            [
                gaussian_filter(
                    np.histogram2d(xy[codes == k, 0], xy[codes == k, 1], bins=bins)[0],
                    smooth_bins,
                )
                for k in range(len(categories))
            ],
            axis=-1,
        )
        proportions /= np.maximum(proportions.sum(axis=-1, keepdims=True), 1e-12)
        rgb = support * (proportions @ pal_rgb) + (1 - support) * floor_rgb
        key_surface = [
            Patch(color=pal_rgb[k], label=c) for k, c in enumerate(categories)
        ]
    else:
        values = adata.obs[color].to_numpy(dtype=float)
        values_median = binned_statistic_2d(
            xy[:, 0], xy[:, 1], values, statistic="median", bins=bins
        ).statistic
        v_num = gaussian_filter(np.nan_to_num(values_median) * count, smooth_bins)
        v = v_num / np.where(den > 0, den, 1)
        vmin, vmax = np.nanpercentile(values, [1, 99])
        rgb = plt.get_cmap(_colormap(cmap, color))(
            np.clip((v - vmin) / (vmax - vmin), 0, 1)
        )[..., :3]
        rgb = support * rgb + (1 - support) * floor_rgb
        key_surface = (float(vmin), float(vmax))

    shaded = LightSource(azdeg=315, altdeg=45).shade_rgb(
        rgb, elevation=z, blend_mode="soft", vert_exag=vert_exag
    )
    ii, jj = np.meshgrid(
        np.arange(grid_size - 1), np.arange(grid_size - 1), indexing="ij"
    )
    ii, jj = ii.ravel(), jj.ravel()
    gx, gy = np.meshgrid(x_centers, y_centers, indexing="ij")
    corners = [(ii, jj), (ii + 1, jj), (ii + 1, jj + 1), (ii, jj + 1)]
    verts = np.stack([np.c_[gx[c], gy[c], z[c]] for c in corners], axis=1)
    faces = np.mean([shaded[c] for c in corners], axis=0)

    key_points: list[Patch] | tuple[float, float] | None = None
    if show_points:
        if color_points is None:
            colors_points = "dimgray"
        elif _is_categorical(adata.obs[color_points]):
            col = adata.obs[color_points].astype("category")
            categories = list(col.cat.categories)
            pal_rgb = _palette_rgb(adata, color_points, categories, cmap_points)
            colors_points = pal_rgb[col.cat.codes.to_numpy()]
            key_points = [
                Patch(color=pal_rgb[k], label=c) for k, c in enumerate(categories)
            ]
        else:
            values = adata.obs[color_points].to_numpy(dtype=float)
            vmin, vmax = np.nanpercentile(values, [1, 99])
            colors_points = plt.get_cmap(_colormap(cmap_points, color_points))(
                np.clip((values - vmin) / (vmax - vmin), 0, 1)
            )
            key_points = (float(vmin), float(vmax))

    slots: tuple[Axes | float, Axes | float] = (0.75, 0.25)
    if ax is None:
        # key column beside the 3d axes so the layout accounts for the keys
        fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            **{"layout": "constrained", **(kwargs_figure or {})},
        )
        grid = fig.add_gridspec(
            1, 2 if show_legend else 1, width_ratios=(1.0, 0.5) if show_legend else None
        )
        ax3d = cast(
            Axes3D,
            fig.add_subplot(grid[0, 0], projection="3d", **(kwargs_axes or {})),
        )
        if show_legend:
            # rows sized by key content, with spacers centering the block
            rows = [_key_rows(key_surface), _key_rows(key_points)]
            n_lines = figsize[1] * 72 / (7 * 1.4)
            spacer = max(0.5, (n_lines - sum(rows)) / 2)
            column = grid[0, 1].subgridspec(
                4, 1, height_ratios=(spacer, rows[0], max(rows[1], 1e-3), spacer)
            )
            slots = (fig.add_subplot(column[1]), fig.add_subplot(column[2]))
            fig.add_subplot(column[0]).axis("off")
            fig.add_subplot(column[3]).axis("off")
        if kwargs_layout:
            layout_engine = fig.get_layout_engine()
            assert layout_engine is not None
            layout_engine.set(**kwargs_layout)
    else:
        ax3d = cast(Axes3D, ax)
    # draw in zorder so points and loops always land on top of the surface
    ax3d.computed_zorder = False
    ax3d.add_collection3d(
        Poly3DCollection(
            verts, facecolors=faces, edgecolors="none", linewidths=0, zorder=1
        )
    )
    surface_at = RegularGridInterpolator(
        (x_centers, y_centers), z, bounds_error=False, fill_value=None
    )

    if show_points:
        z_points = surface_at(xy) + 0.01
        if add_outline:
            # scanpy-style outline: larger background dots, then a gap ring
            point = np.sqrt(size_points)
            gap_size = (point + point * outline_width[1] * 2) ** 2
            bg_size = (np.sqrt(gap_size) + point * outline_width[0] * 2) ** 2
            for size, outline, zorder in (
                (bg_size, outline_color[0], 4.0),
                (gap_size, outline_color[1], 4.5),
            ):
                ax3d.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    z_points,
                    c=outline,
                    s=size,
                    linewidths=0,
                    depthshade=False,
                    zorder=zorder,
                )
        ax3d.scatter(
            xy[:, 0],
            xy[:, 1],
            z_points,
            c=colors_points,
            s=size_points,
            alpha=alpha_points,
            linewidths=0,
            depthshade=False,
            zorder=5,
        )

    if show_reps:
        loops_coords = data._get_loop_embedding(
            selector=track,
            idx_loop=idx_loop,
            embedding_alt=emb,
            include_bootstrap=idx_loop is not None,
            use_refined=True,
        )
        for loop_coords in loops_coords:
            path = (np.asarray(loop_coords) - emb_mean) @ plane
            ax3d.plot(
                path[:, 0],
                path[:, 1],
                surface_at(path) + 0.02,
                color="white",
                lw=1.2,
                zorder=10,
                path_effects=[
                    pe.Stroke(linewidth=2.4, foreground="black"),
                    pe.Normal(),
                ],
            )

    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(zmin, zmax + 0.05 * (zmax - zmin))
    ax3d.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 0.45 * max(hi - lo)))
    ax3d.view_init(elev=elev, azim=azim)
    # 2d-style axes: x/y labels only, no z axis, panes or grid
    ax3d.grid(False)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_visible(False)
    ax3d.zaxis.set_visible(False)
    ax3d.set_zticks([])
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    # without tick labels the default 3d labelpad leaves labels far outside the plot
    ax3d.set_xlabel("loop plane 1", labelpad=-12)
    ax3d.set_ylabel("loop plane 2", labelpad=-12)

    if show_legend:
        _add_key(
            ax3d, key_surface, cmap, f"surface: {color or 'sqrt(membership)'}", slots[0]
        )
        if key_points is not None:
            _add_key(ax3d, key_points, cmap_points, f"points: {color_points}", slots[1])
        elif isinstance(slots[1], Axes):
            slots[1].axis("off")

    savefig_or_show(name="loop_landscape", show=show, save=save)
    if show is False:
        return ax3d
    return None
