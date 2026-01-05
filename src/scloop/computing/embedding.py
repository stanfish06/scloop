# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData


def compute_diffmap(
    adata: AnnData,
    n_comps: int = 15,
    n_neighbors: int = 15,
    use_rep: str | None = None,
    key_added_neighbors: str = "neighbors_diffmap",
    random_state: int = 0,
) -> np.ndarray:
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        method="gauss",
        random_state=random_state,
        key_added=key_added_neighbors,
    )
    sc.tl.diffmap(
        adata,
        n_comps=n_comps,
        neighbors_key=key_added_neighbors,
    )
    return np.array(adata.obsm["X_diffmap"])
