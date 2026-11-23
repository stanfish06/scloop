# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from contextlib import nullcontext
from typing import Annotated, Any

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import Field

from ..computing.membership import compute_loop_membership
from ..computing.threshold import choose_downsample_and_thresholds
from ..data.constants import (
    DEFAULT_CANDIDATE_METHOD,
    DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    DEFAULT_MAX_ROWS_BOUNDARY_MATRIX,
    DEFAULT_MAXITER_EIGENDECOMPOSITION,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_HODGE_COMPONENTS,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_PRESENCE_METHOD,
    DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    SCLOOP_MEMBERSHIP_OBSM_KEY,
    SCLOOP_META_UNS_KEY,
    SCLOOP_UNS_KEY,
)
from ..data.containers import HomologyData
from ..data.metadata import ScloopMeta
from ..data.types import (
    CandidateMethod,
    Index_t,
    NonZeroCount_t,
    Percent_t,
    PositiveFloat,
    PresenceTestMethod,
    Size_t,
)
from ..utils.logging import (
    LogDisplay,
    create_console,
    create_progress,
    ensure_logging,
)

__all__ = ["find_loops", "analyze_loops"]


def _get_scloop_meta(adata: AnnData) -> ScloopMeta:
    if SCLOOP_META_UNS_KEY not in adata.uns:
        raise ValueError("scloop_meta not found in adata.uns. Run prepare_adata first.")
    meta = adata.uns[SCLOOP_META_UNS_KEY]
    if isinstance(meta, dict):
        meta = ScloopMeta(**meta)
    return meta


def find_loops(
    adata: AnnData,
    *,
    threshold_homology: PositiveFloat | None = None,
    threshold_boundary: PositiveFloat | None = None,
    tightness_loops: Percent_t = 0,
    n_candidates: NonZeroCount_t = 1,
    n_bootstrap: Size_t = DEFAULT_N_BOOTSTRAP,
    bootstrap_sampling: str = "downsample",
    bootstrap_downsample_fraction: Percent_t = 2 / 3,
    bootstrap_fps_top_k: int = 5,
    bootstrap_fps_alpha: float = 1.0,
    bootstrap_herding_n_features: int = 1000,
    bootstrap_herding_seed: int | None = None,
    bootstrap_candidate_method: CandidateMethod = DEFAULT_CANDIDATE_METHOD,
    bootstrap_presence_test_method: PresenceTestMethod = DEFAULT_PRESENCE_METHOD,
    require_bootstrap_homological_equivalence: bool = True,
    n_check_per_candidate: NonZeroCount_t = 1,
    max_rows_boundary_matrix: NonZeroCount_t = DEFAULT_MAX_ROWS_BOUNDARY_MATRIX,
    auto_shrink_boundary_matrix: bool = True,
    n_max_workers: NonZeroCount_t = DEFAULT_N_MAX_WORKERS,
    use_parallel: bool = True,
    verbose: bool = False,
    max_log_messages: int | None = None,
    kwargs_bootstrap: dict[str, Any] | None = None,
    kwargs_loop_test: dict[str, Any] | None = None,
    kwargs_loop_representatives: dict[str, Any] | None = None,
    kwargs_loop_refinement: dict[str, Any] | None = None,
    kwargs_loop_membership: dict[str, Any] | None = None,
    kwargs_threshold_search: dict[str, Any] | None = None,
) -> None:
    use_log_display = verbose and max_log_messages is not None
    if verbose:
        ensure_logging()

    display_console = create_console() if use_log_display else None
    progress_main = (
        create_progress(console=display_console) if use_log_display else None
    )
    log_context = (
        LogDisplay(
            maxlen=max_log_messages,
            progress=progress_main,
            console=display_console,
        )
        if use_log_display
        else nullcontext()
    )

    with log_context:
        meta = _get_scloop_meta(adata)
        if meta.bootstrap is None:
            from ..data.metadata import BootstrapMeta

            meta.bootstrap = BootstrapMeta()
        meta.bootstrap.life_pct = tightness_loops
        hd: HomologyData = HomologyData(meta=meta)
        """
        ========= Downsample size and PH thresholds =========
        - grow the point set until the boundary matrix
          no longer fits the budget
        - calibrate a threshold that is not given on pilot resamples of the chosen set
        =====================================================
        """
        assert meta.preprocess is not None
        choice, indices_chosen = choose_downsample_and_thresholds(
            adata=adata,
            meta=meta,
            max_edges=max_rows_boundary_matrix,
            n_top=n_candidates,
            threshold_homology=threshold_homology,
            threshold_boundary=threshold_boundary,
            auto_size=auto_shrink_boundary_matrix,
            verbose=verbose,
            sampling_kwargs={
                "bootstrap_sampling": bootstrap_sampling,
                "bootstrap_downsample_fraction": bootstrap_downsample_fraction,
                "bootstrap_fps_top_k": bootstrap_fps_top_k,
                "bootstrap_fps_alpha": bootstrap_fps_alpha,
                "bootstrap_herding_n_features": bootstrap_herding_n_features,
                "bootstrap_herding_seed": bootstrap_herding_seed,
            },
            **(kwargs_threshold_search or {}),
        )
        if indices_chosen != meta.preprocess.indices_downsample:
            meta.preprocess.indices_downsample = indices_chosen
            meta.preprocess.kwargs_downsample = {
                **(meta.preprocess.kwargs_downsample or {}),
                "n": choice.n,
            }
        threshold_homology = choice.threshold_homology
        boundary_thresh = choice.threshold_boundary
        meta.bootstrap.threshold_homology = threshold_homology
        meta.bootstrap.threshold_boundary = boundary_thresh
        if verbose:
            logger.info(
                f"Using {choice.n} cells, threshold_homology={threshold_homology:.4f}, "
                f"threshold_boundary={boundary_thresh:.4f}"
            )
            logger.info("Computing persistence homology")
        sparse_dist_mat = hd._compute_homology(adata=adata, thresh=threshold_homology)
        hd._compute_boundary_matrix_d1(
            adata=adata, thresh=boundary_thresh, verbose=verbose
        )
        if verbose:
            logger.info(f"Boundary matrix computed with threshold {boundary_thresh}")
        assert hd.boundary_matrix_d1 is not None

        assert meta.preprocess.embedding_method is not None

        kwargs_loop_representatives = kwargs_loop_representatives or {}
        embedding = np.array(adata.obsm[f"X_{meta.preprocess.embedding_method}"])
        if verbose:
            logger.info("Reconstructing loop representatives")
        hd._compute_loop_representatives(
            embedding=embedding,
            pairwise_distance_matrix=sparse_dist_mat,
            top_k=n_candidates,
            life_pct=tightness_loops,
            **kwargs_loop_representatives,
        )
        if verbose:
            n_h1 = len(hd.persistence_diagram[1][0])
            n_loops = sum(lc is not None for lc in hd.selected_loop_classes)
            logger.info(
                f"Found {n_h1} H1 feature(s); reconstructed {n_loops} loop candidate(s)"
            )
        """
        ========= bootstrap =========
        - resample data
        - find loops in resamples
        - map loops to original loops
        =============================
        """
        bootstrap_kwargs = dict(kwargs_bootstrap or {})
        bootstrap_candidate_method = bootstrap_kwargs.pop(
            "candidate_method", bootstrap_candidate_method
        )
        require_bootstrap_homological_equivalence = bootstrap_kwargs.pop(
            "require_homological_equivalence",
            require_bootstrap_homological_equivalence,
        )
        loop_test_kwargs = dict(kwargs_loop_test or {})
        presence_test_method = loop_test_kwargs.pop(
            "presence_test_method", bootstrap_presence_test_method
        )
        # only the wilcoxon presence test needs coherence score
        compute_homotopy_coherence = bootstrap_kwargs.pop(
            "compute_homotopy_coherence", presence_test_method == "wilcoxon"
        )
        if verbose:
            logger.info(f"Bootstrap validation: {n_bootstrap} resamples")
        hd._bootstrap(
            adata=adata,
            n_bootstrap=n_bootstrap,
            thresh=threshold_homology,
            top_k=n_candidates * n_check_per_candidate,
            k_neighbors_check_equivalence=DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
            n_max_workers=n_max_workers,
            life_pct=tightness_loops,
            reconstruct_on_full_data=False,  # this is not working, revisit in the future
            bootstrap_sampling=bootstrap_sampling,
            bootstrap_downsample_fraction=bootstrap_downsample_fraction,
            bootstrap_fps_top_k=bootstrap_fps_top_k,
            bootstrap_fps_alpha=bootstrap_fps_alpha,
            bootstrap_herding_n_features=bootstrap_herding_n_features,
            bootstrap_herding_seed=bootstrap_herding_seed,
            candidate_method=bootstrap_candidate_method,
            require_homological_equivalence=(require_bootstrap_homological_equivalence),
            compute_homotopy_coherence=compute_homotopy_coherence,
            verbose=verbose,
            progress_main=progress_main,
            use_log_display=use_log_display,
            use_parallel=use_parallel,
            **bootstrap_kwargs,
        )
        """
        ========= loop refinement =========
        - refine loops against full data
        ===================================
        """
        if verbose:
            logger.info("Refining loop representatives on full data")
        hd._refine_loop_representatives(
            embedding=embedding,
            include_bootstrap=True,
            **(kwargs_loop_refinement or {}),
        )

        """
        ========= loop membership =========
        - personalized pagerank to reps
        ===================================
        """
        if verbose:
            logger.info("Computing loop membership on full data")
        adata.obsm[SCLOOP_MEMBERSHIP_OBSM_KEY] = compute_loop_membership(
            embedding=embedding,
            representatives=[
                None
                if loop_class is None
                else loop_class.filter_valid(
                    loop_class.representatives_refined
                    or loop_class.representatives
                    or []
                )
                for loop_class in hd.selected_loop_classes
            ],
            scales=[
                0.0 if loop_class is None else loop_class.death
                for loop_class in hd.selected_loop_classes
            ],
            **(kwargs_loop_membership or {}),
        )

        """
        ========= statistcal tests =========
        - fisher exact test loop presence
        - gamma test of persistence
        ====================================
        """
        assert hd.bootstrap_data is not None
        if verbose:
            logger.info("Statistical testing (presence + persistence)")
        hd._test_loops(presence_test_method=presence_test_method, **loop_test_kwargs)
        if verbose:
            presence = hd.bootstrap_data.presence_test_result
            if presence is not None:
                q = presence.pvalues_corrected
                logger.info(
                    f"{sum(p < 0.05 for p in q)}/{len(q)} significant loop(s) (presence q<0.05)"
                )
            logger.success("Loop detection complete")
        adata.uns[SCLOOP_UNS_KEY] = hd


def analyze_loops(
    adata: AnnData,
    track_ids: list[Index_t] | None = None,
    *,
    key_values: str = "dpt_pseudotime",
    n_hodge_components: Annotated[int, Field(ge=1)] = DEFAULT_N_HODGE_COMPONENTS,
    normalized: bool = True,
    n_neighbors_edge_embedding: Annotated[
        int, Field(ge=1)
    ] = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    weight_hodge: float = 0.5,
    half_window: int = 2,
    compute_gene_trends: bool = True,
    gene_trend_genes: list[str] | None = None,
    gene_trend_confidence_level: float = 0.95,
    gene_trend_bandwidth_scale: float = 1.0,
    trajectory_padding_pct: float = 0.2,
    trajectory_smoothing: float = 0.1,
    verbose: bool = False,
    max_log_messages: int | None = None,
    timeout_eigendecomposition: float = DEFAULT_TIMEOUT_EIGENDECOMPOSITION,
    maxiter_eigendecomposition: int | None = DEFAULT_MAXITER_EIGENDECOMPOSITION,
    kwargs_edge_embedding: dict[str, Any] | None = None,
    kwargs_trajectory: dict[str, Any] | None = None,
    kwargs_gene_trends: dict[str, Any] | None = None,
) -> None:
    """Analyze loops using Hodge decomposition and edge embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with scloop results from find_loops().
    track_ids : list[Index_t] | None
        List of track IDs to analyze. If None, analyze all tracks.
    key_values : str
        Key in adata.obs for vertex values (e.g., pseudotime).
    n_hodge_components : int
        Number of Hodge eigenvector components to compute.
    n_neighbors_edge_embedding : int
        Number of neighbors for KNN smoothing of edge embedding.
    normalized : bool
        Whether to use normalized Hodge Laplacian.
    verbose : bool
        Whether to print progress messages.
    max_log_messages : int | None
        If set, use live log display showing last N messages with progress bar.
    **kwargs_edge_embedding
        Additional keyword arguments for edge embedding:
        - weight_hodge : float (default 0.5)
            Weight for Hodge embedding vs gradient (0-1). Higher = more Hodge.
        - half_window : int (default 2)
            Half window size for along-loop smoothing. 0 disables smoothing.
    """
    kwargs_edge_embedding = kwargs_edge_embedding or {}
    kwargs_trajectory = kwargs_trajectory or {}
    kwargs_gene_trends = kwargs_gene_trends or {}

    kwargs_edge_embedding.setdefault("weight_hodge", weight_hodge)
    kwargs_edge_embedding.setdefault("half_window", half_window)
    kwargs_trajectory.setdefault("padding_pct", trajectory_padding_pct)
    kwargs_trajectory.setdefault("s", trajectory_smoothing)
    kwargs_gene_trends.setdefault("confidence_level", gene_trend_confidence_level)
    kwargs_gene_trends.setdefault("bandwidth_scale", gene_trend_bandwidth_scale)

    use_log_display = verbose and max_log_messages is not None
    if verbose:
        ensure_logging()

    display_console = create_console() if use_log_display else None
    progress = create_progress(console=display_console) if use_log_display else None
    log_context = (
        LogDisplay(
            maxlen=max_log_messages,
            progress=progress,
            console=display_console,
        )
        if use_log_display
        else nullcontext()
    )

    with log_context:
        if SCLOOP_UNS_KEY not in adata.uns:
            raise ValueError("Run find_loops() first")

        if key_values not in adata.obs.columns:
            raise ValueError(f"{key_values} not found in adata.obs")

        values_vertices = np.array(adata.obs[key_values])

        assert adata.uns[SCLOOP_META_UNS_KEY].preprocess is not None
        coordinates_vertices = np.array(
            adata.obsm[
                f"X_{adata.uns[SCLOOP_META_UNS_KEY].preprocess.embedding_method}"
            ]
        )

        gene_expression_matrix = None
        gene_names = None
        if compute_gene_trends:
            if adata.X is None:
                raise ValueError(
                    "compute_gene_trends=True requires expression data, but adata.X "
                    "is None. This adata was loaded from a minified scloop file; "
                    "re-run prepare_adata to restore expression, or set "
                    "compute_gene_trends=False."
                )
            if gene_trend_genes is not None:
                adata_genes = adata[:, gene_trend_genes]
                gene_names = gene_trend_genes
            else:
                adata_genes = adata
                gene_names = adata.var_names.tolist()

            X = adata_genes.X
            gene_expression_matrix = X.toarray() if hasattr(X, "toarray") else X  # type: ignore[union-attr]

        hd: HomologyData = adata.uns[SCLOOP_UNS_KEY]

        if hd.boundary_matrix_d0 is None:
            hd._compute_boundary_matrix_d0(verbose=verbose)

        assert hd.bootstrap_data is not None
        track_ids_avail = list(hd.bootstrap_data.loop_tracks.keys())
        if track_ids is None:
            track_ids = track_ids_avail

        if not use_log_display:
            progress = create_progress()

        assert progress is not None
        with nullcontext() if use_log_display else progress:
            task_main = progress.add_task("Analyzing loops...", total=len(track_ids))

            for track_id in track_ids:
                if track_id not in track_ids_avail:
                    progress.advance(task_main)
                    continue

                progress.update(task_main, description=f"Analyzing track {track_id}...")

                hd._compute_hodge_analysis_for_track(
                    idx_track=track_id,
                    values_vertices=values_vertices,
                    coordinates_vertices=coordinates_vertices,
                    n_hodge_components=n_hodge_components,
                    normalized=normalized,
                    n_neighbors_edge_embedding=n_neighbors_edge_embedding,
                    verbose=verbose,
                    progress=progress,
                    timeout_eigendecomposition=timeout_eigendecomposition,
                    maxiter_eigendecomposition=maxiter_eigendecomposition,
                    compute_gene_trends=compute_gene_trends,
                    gene_expression_matrix=gene_expression_matrix,
                    gene_names=gene_names,
                    kwargs_edge_embedding=kwargs_edge_embedding,
                    kwargs_trajectory=kwargs_trajectory,
                    kwargs_gene_trends=kwargs_gene_trends,
                )
                progress.advance(task_main)
