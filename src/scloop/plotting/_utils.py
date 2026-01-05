# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes
from pydantic import ConfigDict, validate_call

from ..data.constants import DEFAULT_DPI, DEFAULT_FIGSIZE
from ..data.containers import HomologyData
from ..data.types import PositiveFloat


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def savefig_or_show(
    name: str,
    show: bool | None = None,
    save: str | bool | None = None,
) -> None:
    if save is not None:
        if isinstance(save, str):
            filename = save
        else:
            filename = f"scloop_{name}.png"
        plt.savefig(filename, bbox_inches="tight")

    if show is None:
        show = save is None

    if show:
        plt.show()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _create_figure_standard(
    figsize: tuple[PositiveFloat, PositiveFloat] = DEFAULT_FIGSIZE,
    dpi: PositiveFloat = DEFAULT_DPI,
    kwargs_figure: dict | None = None,
    kwargs_axes: dict | None = None,
    kwargs_layout: dict | None = None,
) -> Axes:
    kwargs_axes_local = dict(kwargs_axes or {})
    rect = kwargs_axes_local.pop("rect", None)
    fig = plt.figure(figsize=figsize, dpi=dpi, **(kwargs_figure or {}))
    if rect is not None:
        ax: Axes = fig.add_axes(rect, **kwargs_axes_local)
    else:
        ax = fig.add_subplot(111, **kwargs_axes_local)
        fig.tight_layout(**(kwargs_layout or {}))
    return ax


def _get_homology_data(adata: AnnData, key_homology: str) -> HomologyData:
    assert adata.uns[key_homology] is not None
    assert type(adata.uns[key_homology]) is HomologyData
    return adata.uns[key_homology]


def _get_embedding_key(embedding_method: str) -> str:
    if embedding_method in ("pca", "diffmap", "scvi"):
        return f"X_{embedding_method}"
    return embedding_method
