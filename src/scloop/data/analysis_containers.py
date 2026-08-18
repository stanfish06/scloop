# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent
from scipy.stats import chi2_contingency, fisher_exact, gamma, wilcoxon
from scipy.stats.contingency import odds_ratio

from ..computing import compute_weighted_hodge_embedding
from ..utils.pvalues import correct_pvalues
from .base_components import (
    ImagePairRecord,
    LoopClass,
    LoopClassEquivalence,
    PersistenceTestResult,
    PresenceTestResult,
)
from .constants import (
    DEFAULT_HALF_WINDOW,
    DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING,
    DEFAULT_WEIGHT_HODGE,
    DEFAULT_WITH_RELAXATION_EQUIVALENCE,
    NUMERIC_EPSILON,
)
from .types import (
    Count_t,
    Index_t,
    MultipleTestCorrectionMethod,
    Percent_t,
    PositiveFloat,
    Size_t,
)
from .utils import (
    loops_to_coords,
    signed_area_2d,
    smooth_along_loop_1d,
    smooth_along_loop_2d,
)

if TYPE_CHECKING:
    import h5py


def _write_opt_array(parent: h5py.Group, name: str, value, kw: dict) -> None:
    """Write an optional array; absence encodes None."""
    if value is not None:
        parent.create_dataset(name, data=np.asarray(value), **kw)


def _read_opt_array(parent: h5py.Group, name: str):
    return np.asarray(parent[name]) if name in parent else None


def _write_opt_list_of_arrays(parent: h5py.Group, name: str, value, kw: dict) -> None:
    """Serialize a ``list[np.ndarray] | None`` preserving the None-vs-[] split."""
    g = parent.create_group(name)
    if value is None:
        g.attrs["_is_none"] = True
        return
    g.attrs["_is_none"] = False
    g.attrs["_count"] = len(value)
    for i, arr in enumerate(value):
        g.create_dataset(str(i), data=np.asarray(arr), **kw)


def _read_opt_list_of_arrays(parent: h5py.Group, name: str):
    if name not in parent:
        return None
    g = parent[name]
    if g.attrs.get("_is_none", False):
        return None
    return [np.asarray(g[str(i)]) for i in range(int(g.attrs["_count"]))]


def _write_diagram(parent: h5py.Group, name: str, diagram, kw: dict) -> None:
    """Serialize a persistence diagram: list-per-dim of ``[births, deaths]``."""
    g = parent.create_group(name)
    if diagram is None:
        g.attrs["_is_none"] = True
        return
    g.attrs["_is_none"] = False
    g.attrs["_n_dims"] = len(diagram)
    for d, dim_pd in enumerate(diagram):
        dg = g.create_group(str(d))
        if dim_pd is not None and len(dim_pd) >= 2:
            dg.create_dataset(
                "births", data=np.asarray(dim_pd[0], dtype=np.float64), **kw
            )
            dg.create_dataset(
                "deaths", data=np.asarray(dim_pd[1], dtype=np.float64), **kw
            )


def _read_diagram(parent: h5py.Group, name: str):
    if name not in parent:
        return None
    g = parent[name]
    if g.attrs.get("_is_none", False):
        return None
    out = []
    for d in range(int(g.attrs["_n_dims"])):
        dg = g[str(d)]
        if "births" in dg and "deaths" in dg:
            out.append(
                [np.asarray(dg["births"]).tolist(), np.asarray(dg["deaths"]).tolist()]
            )
        else:
            out.append(None)
    return out


def _write_pair_simplices(parent: h5py.Group, name: str, pps, kw: dict) -> None:
    """Serialize births/deaths critical simplices: list-per-dim of ``[births, deaths]``.

    Each simplex is a variable-length list of vertex ids; within a (dim, slot)
    """
    g = parent.create_group(name)
    if pps is None:
        g.attrs["_is_none"] = True
        return
    g.attrs["_is_none"] = False
    g.attrs["_n_dims"] = len(pps)
    for d, dim_pps in enumerate(pps):
        dg = g.create_group(str(d))
        if dim_pps is None or len(dim_pps) < 2:
            dg.attrs["_empty"] = True
            continue
        dg.attrs["_empty"] = False
        for slot, simplices in (("births", dim_pps[0]), ("deaths", dim_pps[1])):
            simplices = [list(s) for s in simplices]
            dg.attrs[slot + "_count"] = len(simplices)
            if not simplices:
                continue
            max_len = max((len(s) for s in simplices), default=0)
            arr = np.full((len(simplices), max_len), -1, dtype=np.int64)
            for i, s in enumerate(simplices):
                arr[i, : len(s)] = s
            dg.create_dataset(slot, data=arr, **kw)


def _read_pair_simplices(parent: h5py.Group, name: str):
    if name not in parent:
        return None
    g = parent[name]
    if g.attrs.get("_is_none", False):
        return None
    out = []
    for d in range(int(g.attrs["_n_dims"])):
        dg = g[str(d)]
        if dg.attrs.get("_empty", False):
            # keep the [births, deaths] shape so downstream `births, deaths =
            # pps[dim]` unpacking never breaks on an empty dimension
            out.append([[], []])
            continue
        slots = []
        for slot in ("births", "deaths"):
            count = int(dg.attrs.get(slot + "_count", 0))
            if count == 0 or slot not in dg:
                slots.append([])
                continue
            arr = np.asarray(dg[slot])
            slots.append([[int(v) for v in row if v >= 0] for row in arr])
        out.append(slots)
    return out


def _write_cocycles(parent: h5py.Group, name: str, cocycles, kw: dict) -> None:
    """Serialize cocycles: list-per-dim of list of ``(vertices, coefficient)``."""
    g = parent.create_group(name)
    if cocycles is None:
        g.attrs["_is_none"] = True
        return
    g.attrs["_is_none"] = False
    g.attrs["_n_dims"] = len(cocycles)
    for d, dim_cocycles in enumerate(cocycles):
        dg = g.create_group(str(d))
        if dim_cocycles is None:
            dg.attrs["_is_none"] = True
            continue
        dg.attrs["_is_none"] = False
        dg.attrs["_n_cocycles"] = len(dim_cocycles)
        for ci, cocycle in enumerate(dim_cocycles):
            cg = dg.create_group(str(ci))
            verts_list: list[list[int]] = []
            coeffs_list: list[int] = []
            for simplex in cocycle or []:
                try:
                    verts, coeff = simplex
                    verts_list.append(list(verts))
                    coeffs_list.append(int(coeff))
                except (ValueError, TypeError):
                    continue
            cg.attrs["_count"] = len(verts_list)
            if verts_list:
                max_len = max(len(v) for v in verts_list)
                verts_arr = np.full((len(verts_list), max_len), -1, dtype=np.int64)
                for i, v in enumerate(verts_list):
                    verts_arr[i, : len(v)] = v
                cg.create_dataset("vertices", data=verts_arr, **kw)
                cg.create_dataset(
                    "coefficients", data=np.array(coeffs_list, dtype=np.int32), **kw
                )


def _read_cocycles(parent: h5py.Group, name: str):
    if name not in parent:
        return None
    g = parent[name]
    if g.attrs.get("_is_none", False):
        return None
    out = []
    for d in range(int(g.attrs["_n_dims"])):
        dg = g[str(d)]
        if dg.attrs.get("_is_none", False):
            out.append(None)
            continue
        dim_cocycles = []
        for ci in range(int(dg.attrs.get("_n_cocycles", 0))):
            cg = dg[str(ci)]
            cocycle = []
            if "vertices" in cg and "coefficients" in cg:
                verts_arr = np.asarray(cg["vertices"])
                coeffs_arr = np.asarray(cg["coefficients"])
                for i in range(len(coeffs_arr)):
                    verts = [int(v) for v in verts_arr[i] if v >= 0]
                    cocycle.append((verts, int(coeffs_arr[i])))
            dim_cocycles.append(cocycle)
        out.append(dim_cocycles)
    return out


def _write_bootstrap_list(
    parent: h5py.Group, name: str, items, per_item_writer, kw: dict
) -> None:
    g = parent.create_group(name)
    g.attrs["_count"] = len(items)
    for i, item in enumerate(items):
        per_item_writer(g, str(i), item, kw)


def _read_bootstrap_list(parent: h5py.Group, name: str, per_item_reader) -> list:
    if name not in parent:
        return []
    g = parent[name]
    return [per_item_reader(g, str(i)) for i in range(int(g.attrs["_count"]))]


_LOOP_CLASS_ANALYSIS_ARRAY_LIST_FIELDS = (
    "coordinates_edges",
    "edge_values_raw",
    "edge_gradient_raw",
    "edge_embedding_raw",
    "edge_embedding_smooth",
    "edge_involvement_raw",
    "edge_involvement_smooth",
)

_TRAJECTORY_OPT_ARRAY_FIELDS = (
    "weights_vertices",
    "indices_vertices",
    "values_vertices",
    "distances_vertices",
    "mean_expression",
    "se_expression",
    "ci_lower",
    "ci_upper",
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoopMatch:
    idx_bootstrap: int
    target_class_idx: int
    candidate_method: Literal["geometric", "image"] = "geometric"
    topological_equivalence: Optional[LoopClassEquivalence] = None
    geometric_distance: Optional[float] = None
    neighbor_rank: Optional[int] = None
    image_death_simplex: Optional[list[int]] = None
    boundary_checked: bool = False

    def summarize_homotopy_coherence(
        self,
        mode: Literal["max", "mean", "median"] = "max",
    ) -> Percent_t | None:
        if self.topological_equivalence is None:
            return None
        values = (
            self.topological_equivalence.homotopy_coherence_matched
            + self.topological_equivalence.homotopy_coherence_matched_relax
        )
        values = [value for value in values if value is not None]
        if not values:
            return None
        match mode:
            case "max":
                return max(values)
            case "mean":
                return float(np.mean(values))
            case "median":
                return float(np.median(values))


def _serialize_loop_matches(
    matches: list[LoopMatch], group: h5py.Group, compress: bool = True
) -> None:
    if not matches:
        group.attrs["_count"] = 0
        return

    n = len(matches)
    group.attrs["_count"] = n
    kw = {"compression": "gzip"} if compress else {}
    group.create_dataset(
        "idx_bootstrap",
        data=np.array([m.idx_bootstrap for m in matches], dtype=np.int64),
        **kw,
    )
    group.create_dataset(
        "target_class_idx",
        data=np.array([m.target_class_idx for m in matches], dtype=np.int64),
        **kw,
    )
    group.create_dataset(
        "candidate_method",
        data=np.array(
            [0 if m.candidate_method == "geometric" else 1 for m in matches],
            dtype=np.int8,
        ),
        **kw,
    )
    group.create_dataset(
        "boundary_checked",
        data=np.array([m.boundary_checked for m in matches], dtype=bool),
        **kw,
    )

    # geometric_distance: store the raw value (NaN is a legitimate value) plus a
    # parallel present-mask so None does not collide with NaN.
    group.create_dataset(
        "geometric_distance",
        data=np.array(
            [
                m.geometric_distance if m.geometric_distance is not None else np.nan
                for m in matches
            ],
            dtype=np.float64,
        ),
        **kw,
    )
    group.create_dataset(
        "geometric_distance_present",
        data=np.array([m.geometric_distance is not None for m in matches], dtype=bool),
        **kw,
    )

    # neighbor_rank: -1 is no longer a None sentinel; use a present-mask.
    group.create_dataset(
        "neighbor_rank",
        data=np.array(
            [m.neighbor_rank if m.neighbor_rank is not None else -1 for m in matches],
            dtype=np.int64,
        ),
        **kw,
    )
    group.create_dataset(
        "neighbor_rank_present",
        data=np.array([m.neighbor_rank is not None for m in matches], dtype=bool),
        **kw,
    )

    # image_death_simplex: ragged (may have >3 vertices) with a None-vs-[] split.
    group.create_dataset(
        "image_death_simplex_present",
        data=np.array([m.image_death_simplex is not None for m in matches], dtype=bool),
        **kw,
    )
    lengths = np.array(
        [len(m.image_death_simplex or []) for m in matches], dtype=np.int64
    )
    group.create_dataset("image_death_simplex_lengths", data=lengths, **kw)
    max_len = int(lengths.max()) if n > 0 else 0
    values = np.full((n, max_len), -1, dtype=np.int64)
    for i, match in enumerate(matches):
        simplex = match.image_death_simplex or []
        values[i, : len(simplex)] = simplex
    group.create_dataset("image_death_simplex_values", data=values, **kw)


def _deserialize_loop_matches(group: h5py.Group) -> list[LoopMatch]:
    count = int(group.attrs.get("_count", 0))
    if count == 0:
        return []

    idx_bootstraps = np.asarray(group["idx_bootstrap"])
    target_class_idxs = np.asarray(group["target_class_idx"])
    geo_dists = np.asarray(group["geometric_distance"])
    neighbor_ranks = np.asarray(group["neighbor_rank"])
    candidate_methods = (
        np.asarray(group["candidate_method"])
        if "candidate_method" in group
        else np.zeros(count, dtype=np.int8)
    )
    boundary_checked = (
        np.asarray(group["boundary_checked"])
        if "boundary_checked" in group
        else np.zeros(count, dtype=bool)
    )

    geo_present = (
        np.asarray(group["geometric_distance_present"])
        if "geometric_distance_present" in group
        else ~np.isnan(geo_dists)
    )
    rank_present = (
        np.asarray(group["neighbor_rank_present"])
        if "neighbor_rank_present" in group
        else neighbor_ranks >= 0
    )

    ids_present = (
        np.asarray(group["image_death_simplex_present"])
        if "image_death_simplex_present" in group
        else None
    )
    ids_lengths = (
        np.asarray(group["image_death_simplex_lengths"])
        if "image_death_simplex_lengths" in group
        else None
    )
    ids_values = (
        np.asarray(group["image_death_simplex_values"])
        if "image_death_simplex_values" in group
        else None
    )

    matches = []
    for i in range(count):
        geo_dist = float(geo_dists[i]) if geo_present[i] else None
        rank = int(neighbor_ranks[i]) if rank_present[i] else None
        if ids_present is not None and ids_values is not None:
            if ids_present[i]:
                length = int(ids_lengths[i]) if ids_lengths is not None else 0
                image_death_simplex = [int(v) for v in ids_values[i, :length]]
            else:
                image_death_simplex = None
        else:
            image_death_simplex = None
        matches.append(
            LoopMatch(
                idx_bootstrap=int(idx_bootstraps[i]),
                target_class_idx=int(target_class_idxs[i]),
                candidate_method=(
                    "geometric" if candidate_methods[i] == 0 else "image"
                ),
                geometric_distance=geo_dist,
                neighbor_rank=rank,
                image_death_simplex=image_death_simplex,
                boundary_checked=bool(boundary_checked[i]),
            )
        )
    return matches


@dataclass
class LoopTrack:
    source_class_idx: int
    matches: list[LoopMatch] = Field(default_factory=list)
    hodge_analysis: HodgeAnalysis | None = None

    def filter_matches(
        self,
        keep: Literal["all", "equivalent"] = "all",
        relax: bool = DEFAULT_WITH_RELAXATION_EQUIVALENCE,
    ) -> list[LoopMatch]:
        if keep == "all":
            return list(self.matches)
        return [
            m
            for m in self.matches
            if m.topological_equivalence is None
            or m.topological_equivalence.is_equivalent(relax=relax)
        ]

    @property
    # it is possible to have one-to-many matches (TODO: need a way to select best match)
    def n_matches(self) -> Count_t:
        return len({m.idx_bootstrap for m in self.filter_matches(keep="equivalent")})

    @property
    def track_ipairs(self) -> list[tuple[Index_t, Index_t]]:
        return [
            (m.idx_bootstrap, m.target_class_idx)
            for m in self.filter_matches(keep="equivalent")
        ]

    def summarize_homotopy_coherence(
        self,
        mode: Literal["full", "mean", "median"] = "full",
        mode_match: Literal["max", "mean", "median"] = "max",
    ) -> list[tuple] | Percent_t | None:
        values = [
            (match.idx_bootstrap, match.summarize_homotopy_coherence(mode=mode_match))
            for match in self.filter_matches(keep="equivalent")
        ]
        if mode == "full":
            return values

        valid_values = [value[1] for value in values if value[1] is not None]
        if not valid_values:
            return None
        match mode:
            case "mean":
                return float(np.mean(valid_values))
            case "median":
                return float(np.median(valid_values))

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "LoopTrack"
        group.attrs["source_class_idx"] = self.source_class_idx

        matches_grp = group.create_group("matches")
        _serialize_loop_matches(self.matches, matches_grp, compress=compress)

        if self.hodge_analysis is not None:
            self.hodge_analysis.to_hdf5_group(
                group.create_group("hodge_analysis"), compress=compress
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> LoopTrack:
        source_class_idx = int(group.attrs["source_class_idx"])  # type: ignore[arg-type]
        matches_grp: h5py.Group = group["matches"]  # type: ignore[assignment]
        matches = _deserialize_loop_matches(matches_grp)
        hodge_analysis = None
        if "hodge_analysis" in group:
            hodge_analysis = HodgeAnalysis.from_hdf5_group(group["hodge_analysis"])
        return cls(
            source_class_idx=source_class_idx,
            matches=matches,
            hodge_analysis=hodge_analysis,
        )


@dataclass
class BootstrapAnalysis:
    num_bootstraps: Size_t = 0
    persistence_diagrams: list[list] = Field(default_factory=list)
    persistence_pair_simplices: list[list] = Field(default_factory=list)
    cocycles: list[list] = Field(default_factory=list)
    reference_image_pairs: list[list[ImagePairRecord]] = Field(default_factory=list)
    bootstrap_image_pairs: list[list[ImagePairRecord]] = Field(default_factory=list)
    selected_loop_classes: list[list[LoopClass | None]] = Field(default_factory=list)
    loop_tracks: dict[Index_t, LoopTrack] = Field(default_factory=dict)
    presence_test_result: PresenceTestResult | None = None
    persistence_test_result: PersistenceTestResult | None = None

    def _get_track_embedding(
        self,
        idx_track: Index_t,
        embedding_alt: np.ndarray | None = None,
        keep_matches: str = "equivalent",
        use_refined: bool = False,
    ) -> list[np.ndarray]:
        assert idx_track in self.loop_tracks
        loops = []
        for match in self.loop_tracks[idx_track].filter_matches(keep=keep_matches):
            boot_id, loop_id = match.idx_bootstrap, match.target_class_idx
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class = self.selected_loop_classes[boot_id][loop_id]
                if loop_class is not None:
                    if embedding_alt is None:
                        if loop_class.coordinates_vertices_representatives is not None:
                            loops.extend(
                                loop_class.coordinates_vertices_representatives
                            )
                    else:
                        reps = loop_class.representatives
                        if (
                            use_refined
                            and loop_class.representatives_refined is not None
                        ):
                            reps = loop_class.representatives_refined
                        if reps is not None:
                            loops.extend(
                                loops_to_coords(
                                    embedding=embedding_alt,
                                    loops_vertices=reps,
                                )
                            )
        return loops

    def _get_loop_embedding(
        self,
        idx_bootstrap: Index_t,
        idx_loop_class: Index_t,
        idx_loop: Index_t | None = None,
        embedding_alt: np.ndarray | None = None,
        use_refined: bool = False,
    ) -> list[list[list[float]]]:
        if idx_bootstrap < len(self.selected_loop_classes) and idx_loop_class < len(
            self.selected_loop_classes[idx_bootstrap]
        ):
            loop_class = self.selected_loop_classes[idx_bootstrap][idx_loop_class]
            if loop_class is not None:
                if embedding_alt is None:
                    if loop_class.coordinates_vertices_representatives is not None:
                        if idx_loop is None:
                            return loop_class.coordinates_vertices_representatives
                        else:
                            assert idx_loop < len(
                                loop_class.coordinates_vertices_representatives
                            )
                            return [
                                loop_class.coordinates_vertices_representatives[
                                    idx_loop
                                ]
                            ]
                else:
                    reps = loop_class.representatives
                    if use_refined and loop_class.representatives_refined is not None:
                        reps = loop_class.representatives_refined
                    if reps is not None:
                        if idx_loop is None:
                            return loops_to_coords(
                                embedding=embedding_alt,
                                loops_vertices=reps,
                            )
                        else:
                            assert idx_loop < len(reps)
                            return loops_to_coords(
                                embedding=embedding_alt,
                                loops_vertices=[reps[idx_loop]],
                            )

        return []

    def _analyze_track_loop_classes(
        self,
        idx_track: Index_t,
        source_loop_class: LoopClass,
        values_vertices: np.ndarray,
    ):
        hodge_analysis = self.loop_tracks[idx_track].hodge_analysis
        assert hodge_analysis is not None
        assert idx_track in self.loop_tracks

        loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
            super_obj=source_loop_class,
            values_vertices=values_vertices,
        )
        hodge_analysis.selected_loop_classes.append(loop_class)

        assert loop_class.coordinates_vertices_representatives is not None
        ref_coords = np.array(loop_class.coordinates_vertices_representatives[0])
        ref_area = signed_area_2d(ref_coords)

        for boot_id, loop_id in self.loop_tracks[idx_track].track_ipairs:
            if boot_id < len(self.selected_loop_classes) and loop_id < len(
                self.selected_loop_classes[boot_id]
            ):
                loop_class_base = self.selected_loop_classes[boot_id][loop_id]
                assert loop_class_base is not None
                loop_class: LoopClassAnalysis = LoopClassAnalysis.from_super(
                    super_obj=loop_class_base,
                    values_vertices=values_vertices,
                    ref_area=ref_area,
                )
                hodge_analysis.selected_loop_classes.append(loop_class)

    @property
    def _n_total_matches(self) -> Count_t:
        return sum([tk.n_matches for tk in self.loop_tracks.values()])

    def _contingency_table_track_to_rest(
        self, tid: int
    ) -> tuple[tuple[Count_t, Count_t], tuple[Count_t, Count_t]]:
        assert tid in self.loop_tracks
        n_matches_track = self.loop_tracks[tid].n_matches
        n_total_matches = self._n_total_matches
        return (
            (n_matches_track, n_total_matches - n_matches_track),
            (
                self.num_bootstraps - n_matches_track,
                self.num_bootstraps * (len(self.loop_tracks) - 1)
                - (n_total_matches - n_matches_track),
            ),
        )

    def fisher_test_presence(
        self, method_pval_correction: MultipleTestCorrectionMethod
    ) -> PresenceTestResult:
        assert self.num_bootstraps > 0
        probs_presence = []
        odds_ratio_presence = []
        pvalues_raw_presence = []
        for tid in self.loop_tracks.keys():
            tbl = self._contingency_table_track_to_rest(tid)
            probs_presence.append(
                float(tbl[0][0]) / (float(tbl[0][0]) + float(tbl[1][0]))
            )
            or_val = odds_ratio(np.array(tbl)).statistic
            odds_ratio_presence.append(or_val if np.isfinite(or_val) else 0.0)
            res = fisher_exact(table=tbl, alternative="greater")
            pvalues_raw_presence.append(res.pvalue)  # type: ignore[attr-defined]
        pvalues_corrected_presence = correct_pvalues(
            pvalues_raw_presence, method=method_pval_correction
        )

        return PresenceTestResult(
            probabilities=probs_presence,
            odds_ratios=odds_ratio_presence,
            pvalues_raw=pvalues_raw_presence,
            pvalues_corrected=pvalues_corrected_presence,
        )

    def chi2_test_presence(
        self, method_pval_correction: MultipleTestCorrectionMethod
    ) -> PresenceTestResult:
        assert self.num_bootstraps > 0
        probs_presence = []
        odds_ratio_presence = []
        pvalues_raw_presence = []
        for tid in self.loop_tracks.keys():
            tbl = self._contingency_table_track_to_rest(tid)
            probs_presence.append(
                float(tbl[0][0]) / (float(tbl[0][0]) + float(tbl[1][0]))
            )
            or_val = odds_ratio(np.array(tbl)).statistic
            odds_ratio_presence.append(or_val if np.isfinite(or_val) else 0.0)

            arr = np.array(tbl, dtype=np.float64)
            total = arr.sum()
            if (
                total <= 0
                or np.any(arr.sum(axis=0) == 0)
                or np.any(arr.sum(axis=1) == 0)
            ):
                pvalues_raw_presence.append(1.0)
                continue

            expected_00 = arr[0].sum() * arr[:, 0].sum() / total
            res = chi2_contingency(arr, correction=False)
            p_two = float(res.pvalue)  # type: ignore[attr-defined]
            if arr[0, 0] >= expected_00:
                p_one = p_two / 2.0
            else:
                p_one = 1.0 - p_two / 2.0
            pvalues_raw_presence.append(p_one)
        pvalues_corrected_presence = correct_pvalues(
            pvalues_raw_presence, method=method_pval_correction
        )

        return PresenceTestResult(
            probabilities=probs_presence,
            odds_ratios=odds_ratio_presence,
            pvalues_raw=pvalues_raw_presence,
            pvalues_corrected=pvalues_corrected_presence,
        )

    def wilcoxon_test_presence(
        self, method_pval_correction: MultipleTestCorrectionMethod
    ):
        track_ids = sorted(self.loop_tracks)
        coherence_per_track: dict[Index_t, list[tuple]] = {  # type: ignore[misc]
            tid: self.loop_tracks[tid].summarize_homotopy_coherence(
                mode="full",
                mode_match="mean",
            )
            for tid in track_ids
        }
        rows_bootstrap = {
            bi: ri
            for ri, bi in enumerate(
                sorted(
                    {bi for values in coherence_per_track.values() for bi, _ in values}
                )
            )
        }
        presence_matrix = np.zeros([len(rows_bootstrap), len(track_ids)])
        for ti, tid in enumerate(track_ids):
            for bi, cv in coherence_per_track[tid]:
                presence_matrix[rows_bootstrap[bi], ti] = cv if cv else 0
        presence_global = np.mean(presence_matrix, axis=1)
        pvalues_raw_presence = []
        for i in range(len(track_ids)):
            column = presence_matrix[:, i]
            if np.allclose(column, presence_global):
                pvalues_raw_presence.append(1.0)
                continue
            result = wilcoxon(x=column, y=presence_global, alternative="greater")
            pvalue = float(result.pvalue)  # type: ignore[attr-defined]
            pvalues_raw_presence.append(pvalue if np.isfinite(pvalue) else 1.0)
        pvalues_corrected_presence = correct_pvalues(
            pvalues_raw_presence, method=method_pval_correction
        )
        return PresenceTestResult(
            probabilities=[],
            odds_ratios=[],
            pvalues_raw=pvalues_raw_presence,
            pvalues_corrected=pvalues_corrected_presence,
        )

    def gamma_test_persistence(
        self,
        selected_loop_classes: list,
        method_pval_correction: MultipleTestCorrectionMethod,
    ) -> PersistenceTestResult:
        if len(self.persistence_diagrams) == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        lifetimes_bootstrap = []
        for diag in self.persistence_diagrams:
            if len(diag) <= 1:
                continue
            births = np.asarray(diag[1][0])
            deaths = np.asarray(diag[1][1])
            lifetimes_bootstrap.append(deaths - births)

        if len(lifetimes_bootstrap) == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        lifetimes_bootstrap_arr = np.concatenate(lifetimes_bootstrap)
        lifetimes_bootstrap_arr = lifetimes_bootstrap_arr[
            np.isfinite(lifetimes_bootstrap_arr) & (lifetimes_bootstrap_arr > 0)
        ]
        if lifetimes_bootstrap_arr.size == 0:
            return PersistenceTestResult(
                pvalues_raw=[], pvalues_corrected=[], gamma_null_params=None
            )

        params = gamma.fit(lifetimes_bootstrap_arr, floc=0)

        pvalues_raw_persistence: list[PositiveFloat] = []
        for loop_track in self.loop_tracks.values():
            source_idx = loop_track.source_class_idx
            if source_idx < len(selected_loop_classes):
                loop_class = selected_loop_classes[source_idx]
                if loop_class is not None:
                    lifetime = float(loop_class.lifetime)
                    p_val = float(
                        1
                        - gamma.cdf(
                            lifetime, a=params[0], loc=params[1], scale=params[2]
                        )
                    )
                    pvalues_raw_persistence.append(p_val)

        pvalues_corrected_persistence = correct_pvalues(
            pvalues_raw_persistence, method=method_pval_correction
        )

        self.gamma_null_params = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
        )
        self.persistence_test_result = PersistenceTestResult(
            pvalues_raw=pvalues_raw_persistence,
            pvalues_corrected=pvalues_corrected_persistence,
            gamma_null_params=self.gamma_null_params,
        )
        return self.persistence_test_result

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "BootstrapAnalysis"
        group.attrs["num_bootstraps"] = self.num_bootstraps

        kw = {"compression": "gzip"} if compress else {}

        # raw per-round homology: needed to recompute loop reps / rerun tests
        _write_bootstrap_list(
            group, "persistence_diagrams", self.persistence_diagrams, _write_diagram, kw
        )
        _write_bootstrap_list(
            group,
            "persistence_pair_simplices",
            self.persistence_pair_simplices,
            _write_pair_simplices,
            kw,
        )
        _write_bootstrap_list(group, "cocycles", self.cocycles, _write_cocycles, kw)

        slc_grp = group.create_group("selected_loop_classes")
        slc_grp.attrs["_count"] = len(self.selected_loop_classes)
        for boot_idx, loop_classes in enumerate(self.selected_loop_classes):
            boot_grp = slc_grp.create_group(str(boot_idx))
            boot_grp.attrs["_count"] = len(loop_classes)
            for lc_idx, lc in enumerate(loop_classes):
                lc_grp = boot_grp.create_group(str(lc_idx))
                if lc is None:
                    lc_grp.attrs["_is_none"] = True
                else:
                    lc_grp.attrs["_is_none"] = False
                    lc.to_hdf5_group(lc_grp, compress=compress)

        for field_name, records_by_bootstrap in (
            ("reference_image_pairs", self.reference_image_pairs),
            ("bootstrap_image_pairs", self.bootstrap_image_pairs),
        ):
            records_grp = group.create_group(field_name)
            records_grp.attrs["_count"] = len(records_by_bootstrap)
            for boot_idx, records in enumerate(records_by_bootstrap):
                boot_grp = records_grp.create_group(str(boot_idx))
                boot_grp.attrs["_count"] = len(records)
                for record_idx, record in enumerate(records):
                    record.to_hdf5_group(
                        boot_grp.create_group(str(record_idx)), compress=compress
                    )

        # loop_tracks: dict[int, LoopTrack]
        tracks_grp = group.create_group("loop_tracks")
        for track_id, track in self.loop_tracks.items():
            track_grp = tracks_grp.create_group(str(track_id))
            track.to_hdf5_group(track_grp, compress=compress)

        # test results
        if self.presence_test_result is not None:
            presence_grp = group.create_group("presence_test_result")
            self.presence_test_result.to_hdf5_group(presence_grp, compress=compress)

        if self.persistence_test_result is not None:
            persistence_grp = group.create_group("persistence_test_result")
            self.persistence_test_result.to_hdf5_group(
                persistence_grp, compress=compress
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> BootstrapAnalysis:
        num_bootstraps = int(group.attrs["num_bootstraps"])  # type: ignore[arg-type]

        # selected_loop_classes
        selected_loop_classes: list[list[LoopClass | None]] = []
        slc_grp: h5py.Group = group["selected_loop_classes"]  # type: ignore[assignment]
        n_boots = int(slc_grp.attrs["_count"])  # type: ignore[arg-type]
        for boot_idx in range(n_boots):
            boot_grp: h5py.Group = slc_grp[str(boot_idx)]  # type: ignore[assignment]
            n_lcs = int(boot_grp.attrs["_count"])  # type: ignore[arg-type]
            loop_classes: list[LoopClass | None] = []
            for lc_idx in range(n_lcs):
                lc_grp: h5py.Group = boot_grp[str(lc_idx)]  # type: ignore[assignment]
                if lc_grp.attrs.get("_is_none", False):
                    loop_classes.append(None)
                else:
                    loop_classes.append(LoopClass.from_hdf5_group(lc_grp))
            selected_loop_classes.append(loop_classes)

        image_records: dict[str, list[list[ImagePairRecord]]] = {}
        for field_name in ("reference_image_pairs", "bootstrap_image_pairs"):
            records_by_bootstrap: list[list[ImagePairRecord]] = []
            if field_name in group:
                records_grp: h5py.Group = group[field_name]  # type: ignore[assignment]
                n_record_bootstraps = int(records_grp.attrs["_count"])
                for boot_idx in range(n_record_bootstraps):
                    boot_grp: h5py.Group = records_grp[str(boot_idx)]  # type: ignore[assignment]
                    records_by_bootstrap.append(
                        [
                            ImagePairRecord.from_hdf5_group(boot_grp[str(i)])
                            for i in range(int(boot_grp.attrs["_count"]))
                        ]
                    )
            image_records[field_name] = records_by_bootstrap

        # loop_tracks
        loop_tracks: dict[int, LoopTrack] = {}
        tracks_grp: h5py.Group = group["loop_tracks"]  # type: ignore[assignment]
        for track_id_str in tracks_grp.keys():
            track_grp: h5py.Group = tracks_grp[track_id_str]  # type: ignore[assignment]
            loop_tracks[int(track_id_str)] = LoopTrack.from_hdf5_group(track_grp)

        # test results
        presence_test_result = None
        if "presence_test_result" in group:
            presence_grp: h5py.Group = group["presence_test_result"]  # type: ignore[assignment]
            presence_test_result = PresenceTestResult.from_hdf5_group(presence_grp)

        persistence_test_result = None
        if "persistence_test_result" in group:
            persistence_grp: h5py.Group = group["persistence_test_result"]  # type: ignore[assignment]
            persistence_test_result = PersistenceTestResult.from_hdf5_group(
                persistence_grp
            )

        persistence_diagrams = _read_bootstrap_list(
            group, "persistence_diagrams", _read_diagram
        )
        persistence_pair_simplices = _read_bootstrap_list(
            group, "persistence_pair_simplices", _read_pair_simplices
        )
        cocycles = _read_bootstrap_list(group, "cocycles", _read_cocycles)

        return cls(
            num_bootstraps=num_bootstraps,
            persistence_diagrams=persistence_diagrams,
            persistence_pair_simplices=persistence_pair_simplices,
            cocycles=cocycles,
            reference_image_pairs=image_records["reference_image_pairs"],
            bootstrap_image_pairs=image_records["bootstrap_image_pairs"],
            selected_loop_classes=selected_loop_classes,
            loop_tracks=loop_tracks,
            presence_test_result=presence_test_result,
            persistence_test_result=persistence_test_result,
        )


class LoopClassAnalysis(LoopClass):
    coordinates_edges: list[np.ndarray] | None = None
    edge_values_raw: list[np.ndarray] | None = None
    edge_gradient_raw: list[np.ndarray] | None = None
    edge_embedding_raw: list[np.ndarray] | None = None
    edge_embedding_smooth: list[np.ndarray] | None = None
    edge_involvement_raw: list[np.ndarray] | None = None
    edge_involvement_smooth: list[np.ndarray] | None = None
    valid_edge_indices_per_rep: list[list[int]] = Field(default_factory=list)
    edge_signs_per_rep: list[np.ndarray] = Field(default_factory=list)
    vertex_divergence_raw: np.ndarray | None = None
    vertex_divergence_smooth: np.ndarray | None = None
    vertex_ids_divergence: list[int] | None = None

    def _concat_property(
        self, attr_name: str, apply_filter: bool = False
    ) -> np.ndarray:
        attr = getattr(self, attr_name)
        if attr is None:
            return np.array([])

        if apply_filter and self.valid_edge_indices_per_rep:
            filtered_parts = []
            for part, indices in zip(attr, self.valid_edge_indices_per_rep):
                if len(indices) > 0:
                    filtered_parts.append(part[indices])
            if not filtered_parts:
                return (
                    np.array([]).reshape(0, *attr[0].shape[1:])
                    if attr
                    else np.array([])
                )
            return np.concatenate(filtered_parts)

        return np.concatenate(attr)

    @property
    def coordinates_edges_all(self):
        return self._concat_property("coordinates_edges", apply_filter=True)

    @property
    def edge_values_raw_all(self):
        return self._concat_property("edge_values_raw", apply_filter=True)

    @property
    def edge_gradient_raw_all(self):
        return self._concat_property("edge_gradient_raw", apply_filter=True)

    @property
    def edge_embedding_raw_all(self):
        return self._concat_property("edge_embedding_raw", apply_filter=False)

    @property
    def edge_embedding_smooth_all(self):
        return self._concat_property("edge_embedding_smooth", apply_filter=False)

    @property
    def edge_involvement_raw_all(self):
        return self._concat_property("edge_involvement_raw", apply_filter=False)

    @property
    def edge_involvement_smooth_all(self):
        return self._concat_property("edge_involvement_smooth", apply_filter=False)

    @classmethod
    def from_super(
        cls,
        super_obj: LoopClass,
        values_vertices: np.ndarray,
        ref_area: float | None = None,
    ):
        assert super_obj.representatives is not None
        assert super_obj.coordinates_vertices_representatives is not None

        coordinates_vertices = [
            np.array(coords)
            for coords in super_obj.coordinates_vertices_representatives
        ]
        representatives = [list(rep) for rep in super_obj.representatives]
        representatives_refined = (
            [list(rep) for rep in super_obj.representatives_refined]
            if super_obj.representatives_refined is not None
            else None
        )

        if len(coordinates_vertices) > 0:
            if ref_area is None:
                ref_area = signed_area_2d(coordinates_vertices[0])
            if abs(ref_area) > NUMERIC_EPSILON:
                for i in range(len(coordinates_vertices)):
                    if ref_area * signed_area_2d(coordinates_vertices[i]) < 0:
                        coordinates_vertices[i] = coordinates_vertices[i][::-1]
                        representatives[i] = representatives[i][::-1]
                        if representatives_refined is not None and i < len(
                            representatives_refined
                        ):
                            representatives_refined[i] = representatives_refined[i][
                                ::-1
                            ]

        coordinates_edges = [
            (emb[0:-1, :] + emb[1:, :]) / 2 for emb in coordinates_vertices
        ]

        if values_vertices.ndim == 1:
            values_vertices = values_vertices.reshape(-1, 1)

        edge_values_raw = loops_to_coords(
            embedding=values_vertices, loops_vertices=representatives
        )
        edge_values_raw = [
            (np.array(vals)[0:-1, :] + np.array(vals)[1:, :]) / 2
            for vals in edge_values_raw
        ]

        edge_gradient_raw = loops_to_coords(
            embedding=values_vertices, loops_vertices=representatives
        )
        edge_gradient_raw = [
            np.diff(np.array(vals), axis=0) for vals in edge_gradient_raw
        ]

        return cls(
            rank=super_obj.rank,
            persistence_index=super_obj.persistence_index,
            birth=super_obj.birth,
            death=super_obj.death,
            birth_simplex=super_obj.birth_simplex,
            death_simplex=super_obj.death_simplex,
            cocycles=super_obj.cocycles,
            representatives=representatives,
            representatives_refined=representatives_refined,
            coordinates_vertices_representatives=[
                c.tolist() for c in coordinates_vertices
            ],
            coordinates_edges=coordinates_edges,
            edge_values_raw=edge_values_raw,
            edge_gradient_raw=edge_gradient_raw,
        )

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        # base fields first, then override the stamped _type
        LoopClass.to_hdf5_group(self, group, compress=compress)
        group.attrs["_type"] = "LoopClassAnalysis"

        kw = {"compression": "gzip"} if compress else {}
        for field_name in _LOOP_CLASS_ANALYSIS_ARRAY_LIST_FIELDS:
            _write_opt_list_of_arrays(group, field_name, getattr(self, field_name), kw)

        _write_opt_list_of_arrays(
            group,
            "valid_edge_indices_per_rep",
            [np.asarray(v, dtype=np.int64) for v in self.valid_edge_indices_per_rep],
            kw,
        )
        _write_opt_list_of_arrays(
            group, "edge_signs_per_rep", self.edge_signs_per_rep, kw
        )
        _write_opt_array(group, "vertex_divergence_raw", self.vertex_divergence_raw, kw)
        _write_opt_array(
            group, "vertex_divergence_smooth", self.vertex_divergence_smooth, kw
        )
        if self.vertex_ids_divergence is not None:
            group.create_dataset(
                "vertex_ids_divergence",
                data=np.asarray(self.vertex_ids_divergence, dtype=np.int64),
                **kw,
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> LoopClassAnalysis:
        base = LoopClass._read_base_fields(group)

        extra: dict = {
            field_name: _read_opt_list_of_arrays(group, field_name)
            for field_name in _LOOP_CLASS_ANALYSIS_ARRAY_LIST_FIELDS
        }

        valid_idx = _read_opt_list_of_arrays(group, "valid_edge_indices_per_rep")
        extra["valid_edge_indices_per_rep"] = (
            [np.asarray(a).astype(int).tolist() for a in valid_idx]
            if valid_idx is not None
            else []
        )
        edge_signs = _read_opt_list_of_arrays(group, "edge_signs_per_rep")
        extra["edge_signs_per_rep"] = edge_signs if edge_signs is not None else []
        extra["vertex_divergence_raw"] = _read_opt_array(group, "vertex_divergence_raw")
        extra["vertex_divergence_smooth"] = _read_opt_array(
            group, "vertex_divergence_smooth"
        )
        extra["vertex_ids_divergence"] = (
            np.asarray(group["vertex_ids_divergence"]).astype(int).tolist()
            if "vertex_ids_divergence" in group
            else None
        )
        return cls(**base, **extra)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrajectoryAnalysis:
    trajectory_coordinates: np.ndarray
    trajectory_pseudotime_range: tuple[float, float] | None = None
    n_bins: int = 20

    weights_vertices: np.ndarray | None = None
    indices_vertices: np.ndarray | None = None
    values_vertices: np.ndarray | None = None
    bandwidth_vertices: float | None = None
    distances_vertices: np.ndarray | None = None

    gene_names: list[str] | None = None
    mean_expression: np.ndarray | None = None
    se_expression: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None

    gam_n_splines: int = 10

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        import h5py

        group.attrs["_type"] = "TrajectoryAnalysis"
        kw = {"compression": "gzip"} if compress else {}

        group.create_dataset(
            "trajectory_coordinates",
            data=np.asarray(self.trajectory_coordinates),
            **kw,
        )
        group.attrs["n_bins"] = self.n_bins
        group.attrs["gam_n_splines"] = self.gam_n_splines

        if self.trajectory_pseudotime_range is not None:
            group.create_dataset(
                "trajectory_pseudotime_range",
                data=np.asarray(self.trajectory_pseudotime_range, dtype=np.float64),
            )
        if self.bandwidth_vertices is not None:
            group.attrs["bandwidth_vertices"] = float(self.bandwidth_vertices)

        for field_name in _TRAJECTORY_OPT_ARRAY_FIELDS:
            _write_opt_array(group, field_name, getattr(self, field_name), kw)

        if self.gene_names is not None:
            group.create_dataset(
                "gene_names",
                data=np.array(self.gene_names, dtype=object),
                dtype=h5py.string_dtype(),
                **kw,
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> TrajectoryAnalysis:
        ptr = None
        if "trajectory_pseudotime_range" in group:
            arr = np.asarray(group["trajectory_pseudotime_range"])
            ptr = (float(arr[0]), float(arr[1]))

        gene_names = None
        if "gene_names" in group:
            gene_names = [
                g.decode() if isinstance(g, bytes) else str(g)
                for g in np.asarray(group["gene_names"])
            ]

        return cls(
            trajectory_coordinates=np.asarray(group["trajectory_coordinates"]),
            trajectory_pseudotime_range=ptr,
            n_bins=int(group.attrs.get("n_bins", 20)),
            weights_vertices=_read_opt_array(group, "weights_vertices"),
            indices_vertices=_read_opt_array(group, "indices_vertices"),
            values_vertices=_read_opt_array(group, "values_vertices"),
            bandwidth_vertices=(
                float(group.attrs["bandwidth_vertices"])
                if "bandwidth_vertices" in group.attrs
                else None
            ),
            distances_vertices=_read_opt_array(group, "distances_vertices"),
            gene_names=gene_names,
            mean_expression=_read_opt_array(group, "mean_expression"),
            se_expression=_read_opt_array(group, "se_expression"),
            ci_lower=_read_opt_array(group, "ci_lower"),
            ci_upper=_read_opt_array(group, "ci_upper"),
            gam_n_splines=int(group.attrs.get("gam_n_splines", 10)),
        )


class HodgeAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hodge_eigenvalues: list | None = None
    hodge_eigenvectors: list | None = None
    edges_masks_loop_classes: list[list[np.ndarray]] = Field(default_factory=list)
    selected_loop_classes: list[LoopClassAnalysis] = Field(default_factory=list)
    trajectory_analyses: list[TrajectoryAnalysis] = Field(default_factory=list)

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "HodgeAnalysis"
        kw = {"compression": "gzip"} if compress else {}

        if self.hodge_eigenvalues is not None:
            group.create_dataset(
                "hodge_eigenvalues",
                data=np.asarray(self.hodge_eigenvalues, dtype=np.float64),
                **kw,
            )
        if self.hodge_eigenvectors is not None:
            group.create_dataset(
                "hodge_eigenvectors",
                data=np.asarray(self.hodge_eigenvectors, dtype=np.float64),
                **kw,
            )

        em_grp = group.create_group("edges_masks_loop_classes")
        em_grp.attrs["_count"] = len(self.edges_masks_loop_classes)
        for i, masks in enumerate(self.edges_masks_loop_classes):
            sub = em_grp.create_group(str(i))
            sub.attrs["_count"] = len(masks)
            for j, mask in enumerate(masks):
                sub.create_dataset(str(j), data=np.asarray(mask), **kw)

        slc_grp = group.create_group("selected_loop_classes")
        slc_grp.attrs["_count"] = len(self.selected_loop_classes)
        for i, lc in enumerate(self.selected_loop_classes):
            lc.to_hdf5_group(slc_grp.create_group(str(i)), compress=compress)

        ta_grp = group.create_group("trajectory_analyses")
        ta_grp.attrs["_count"] = len(self.trajectory_analyses)
        for i, ta in enumerate(self.trajectory_analyses):
            ta.to_hdf5_group(ta_grp.create_group(str(i)), compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> HodgeAnalysis:
        hodge_eigenvalues = (
            np.asarray(group["hodge_eigenvalues"]).tolist()
            if "hodge_eigenvalues" in group
            else None
        )
        hodge_eigenvectors = (
            np.asarray(group["hodge_eigenvectors"]).tolist()
            if "hodge_eigenvectors" in group
            else None
        )

        edges_masks_loop_classes: list[list[np.ndarray]] = []
        if "edges_masks_loop_classes" in group:
            em_grp = group["edges_masks_loop_classes"]
            for i in range(int(em_grp.attrs["_count"])):
                sub = em_grp[str(i)]
                edges_masks_loop_classes.append(
                    [np.asarray(sub[str(j)]) for j in range(int(sub.attrs["_count"]))]
                )

        selected_loop_classes: list[LoopClassAnalysis] = []
        if "selected_loop_classes" in group:
            slc_grp = group["selected_loop_classes"]
            for i in range(int(slc_grp.attrs["_count"])):
                selected_loop_classes.append(
                    LoopClassAnalysis.from_hdf5_group(slc_grp[str(i)])
                )

        trajectory_analyses: list[TrajectoryAnalysis] = []
        if "trajectory_analyses" in group:
            ta_grp = group["trajectory_analyses"]
            for i in range(int(ta_grp.attrs["_count"])):
                trajectory_analyses.append(
                    TrajectoryAnalysis.from_hdf5_group(ta_grp[str(i)])
                )

        return cls(
            hodge_eigenvalues=hodge_eigenvalues,
            hodge_eigenvectors=hodge_eigenvectors,
            edges_masks_loop_classes=edges_masks_loop_classes,
            selected_loop_classes=selected_loop_classes,
            trajectory_analyses=trajectory_analyses,
        )

    def _embed_edges(
        self,
        weight_hodge: Percent_t = DEFAULT_WEIGHT_HODGE,
        half_window: int = DEFAULT_HALF_WINDOW,
    ):
        if self.hodge_eigenvectors is None:
            return

        hodge_evecs = np.array(self.hodge_eigenvectors)

        for loop_idx, loop in enumerate(self.selected_loop_classes):
            if loop.edge_gradient_raw is None or loop.coordinates_edges is None:
                continue

            edge_masks = self.edges_masks_loop_classes[loop_idx]
            loop.edge_embedding_raw = []
            loop.edge_involvement_raw = []

            for rep_idx, edge_mask in enumerate(edge_masks):
                valid_indices = loop.valid_edge_indices_per_rep[rep_idx]
                if not valid_indices:
                    continue

                edge_gradients = loop.edge_gradient_raw[rep_idx][valid_indices]
                edge_evec_values = edge_mask.astype(np.float64) @ hodge_evecs.T

                if loop.edge_signs_per_rep and rep_idx < len(loop.edge_signs_per_rep):
                    edge_signs = loop.edge_signs_per_rep[rep_idx][:, None]
                    edge_evec_values = edge_evec_values * edge_signs

                if half_window > 0:
                    edge_gradients_smooth = smooth_along_loop_1d(
                        edge_gradients.flatten().astype(np.float64), half_window
                    ).reshape(-1, 1)
                    edge_evec_smooth = smooth_along_loop_2d(
                        edge_evec_values.astype(np.float64), half_window
                    )
                else:
                    edge_gradients_smooth = edge_gradients
                    edge_evec_smooth = edge_evec_values

                weighted_edge_hodge, involvement_hodge = (
                    compute_weighted_hodge_embedding(
                        edge_evecs=edge_evec_smooth,
                        eigenvalues=np.array(self.hodge_eigenvalues),
                        edge_gradients=edge_gradients_smooth,
                    )
                )
                grad_1d = edge_gradients_smooth.flatten()
                edge_embedding = weighted_edge_hodge * weight_hodge + grad_1d * (
                    1 - weight_hodge
                )
                loop.edge_embedding_raw.append(edge_embedding)
                loop.edge_involvement_raw.append(involvement_hodge)

    def _smoothening_edge_embedding(
        self, n_neighbors: Count_t = DEFAULT_N_NEIGHBORS_EDGE_EMBEDDING
    ):
        coordinates_edges_all = np.concatenate(
            [loop.coordinates_edges_all for loop in self.selected_loop_classes], axis=0
        )
        edge_embedding_raw_all = np.concatenate(
            [loop.edge_embedding_raw_all for loop in self.selected_loop_classes], axis=0
        )
        edge_involvement_raw_all = np.concatenate(
            [loop.edge_involvement_raw_all for loop in self.selected_loop_classes],
            axis=0,
        )

        search_index = NNDescent(coordinates_edges_all)
        for loops in self.selected_loop_classes:
            loops.edge_embedding_smooth = []
            loops.edge_involvement_smooth = []
            assert loops.edge_embedding_raw is not None
            assert loops.edge_involvement_raw is not None
            assert loops.coordinates_edges is not None

            for rep_idx, coords_raw in enumerate(loops.coordinates_edges):
                valid_idx = loops.valid_edge_indices_per_rep[rep_idx]
                if not valid_idx:
                    continue

                # ISSUE: this looks inefficient, you can directly extract neighbor graph from NNDescent
                # also here it does not remove self match
                coords = coords_raw[valid_idx]
                nn_indices, nn_distances = search_index.query(
                    query_data=coords, k=n_neighbors
                )
                length_scale = np.median(nn_distances, axis=1, keepdims=True) + 1e-8
                nn_similarities = np.exp(-nn_distances / length_scale)
                nn_weights = nn_similarities / nn_similarities.sum(
                    axis=1, keepdims=True
                )

                neighbor_embeddings = edge_embedding_raw_all[nn_indices]
                if neighbor_embeddings.ndim == 2:
                    smoothed_emb = (neighbor_embeddings * nn_weights).sum(axis=1)
                else:
                    smoothed_emb = (neighbor_embeddings * nn_weights[:, :, None]).sum(
                        axis=1
                    )
                loops.edge_embedding_smooth.append(smoothed_emb)

                neighbor_involvements = edge_involvement_raw_all[nn_indices]
                if neighbor_involvements.ndim == 2:
                    smoothed_inv = (neighbor_involvements * nn_weights).sum(axis=1)
                else:
                    smoothed_inv = (neighbor_involvements * nn_weights[:, :, None]).sum(
                        axis=1
                    )
                loops.edge_involvement_smooth.append(smoothed_inv)

    def _trajectory_identification(
        self,
        coordinates_vertices: np.ndarray,
        values_vertices: np.ndarray,
        use_smooth: bool = True,
        percentile_threshold_involvement: Percent_t = 0,
        n_bins: int = 20,
        min_n_bins: int = 4,
        s: float = 0.1,
        padding_pct: float = 0.2,
        split_threshold: float = 0.0,
    ):
        from scipy.interpolate import splev, splprep

        coords_all = np.concatenate(
            [lc.coordinates_edges_all for lc in self.selected_loop_classes]
        )
        vals_all = np.concatenate(
            [lc.edge_values_raw_all for lc in self.selected_loop_classes]
        ).flatten()

        emb_all = np.concatenate(
            [
                lc.edge_embedding_smooth_all
                if use_smooth
                else lc.edge_embedding_raw_all
                for lc in self.selected_loop_classes
            ]
        )
        inv_all = np.concatenate(
            [
                lc.edge_involvement_smooth_all
                if use_smooth
                else lc.edge_involvement_raw_all
                for lc in self.selected_loop_classes
            ]
        )

        mask_edge_involved = inv_all > np.percentile(
            a=inv_all, q=percentile_threshold_involvement * 100
        )
        vals_involved = vals_all[mask_edge_involved]

        if len(vals_involved) == 0:
            self.trajectory_analyses = []
            return

        t_loop_start = float(np.percentile(vals_involved, 5))
        t_loop_end = float(np.percentile(vals_involved, 95))
        loop_span = t_loop_end - t_loop_start

        t_global_min = float(np.min(values_vertices))
        t_global_max = float(np.max(values_vertices))

        t_entry_min = max(t_global_min, t_loop_start - (loop_span * padding_pct))
        t_exit_max = min(t_global_max, t_loop_end + (loop_span * padding_pct))

        n_bins_stem = max(2, int(n_bins * padding_pct))

        def get_stem_points(t_start, t_end, n_bins_s):
            if t_end <= t_start:
                return [], [], []

            bins = np.linspace(t_start, t_end, n_bins_s + 1)
            stem_centers = []
            stem_weights = []
            stem_t = []

            for i in range(n_bins_s):
                m_bin = (values_vertices >= bins[i]) & (values_vertices < bins[i + 1])
                if np.any(m_bin):
                    center = np.mean(coordinates_vertices[m_bin], axis=0)
                    weight = np.sum(m_bin)
                    stem_centers.append(center)
                    stem_weights.append(weight)
                    stem_t.append((bins[i] + bins[i + 1]) / 2)

            return stem_centers, stem_weights, stem_t

        entry_centers, entry_weights, entry_t = get_stem_points(
            t_entry_min, t_loop_start, n_bins_stem
        )
        exit_centers, exit_weights, exit_t = get_stem_points(
            t_loop_end, t_exit_max, n_bins_stem
        )

        trajs = []
        traj_analyses = []
        for sign in [1, -1]:
            if sign == 1:
                mask_sign = emb_all > split_threshold
            else:
                mask_sign = emb_all < -split_threshold

            mask_arm = mask_edge_involved & mask_sign
            if not np.any(mask_arm):
                continue

            c_arm = coords_all[mask_arm]
            v_arm = vals_all[mask_arm]
            w_arm = inv_all[mask_arm]

            bins = np.linspace(v_arm.min(), v_arm.max(), n_bins + 1)
            arm_centers = []
            arm_weights = []
            arm_t = []

            for i in range(n_bins):
                m_bin = (v_arm >= bins[i]) & (v_arm < bins[i + 1])
                if np.any(m_bin):
                    weights_bin = w_arm[m_bin]
                    total_weight = np.sum(weights_bin) + NUMERIC_EPSILON
                    center = np.average(c_arm[m_bin], axis=0, weights=weights_bin)

                    arm_centers.append(center)
                    arm_weights.append(total_weight)
                    arm_t.append((bins[i] + bins[i + 1]) / 2)

            full_centers = entry_centers + arm_centers + exit_centers
            full_weights = entry_weights + arm_weights + exit_weights
            full_t = entry_t + arm_t + exit_t

            if len(full_centers) < min_n_bins:
                continue

            pts = np.array(full_centers).T
            w_pts = np.array(full_weights)

            idx_sort = np.argsort(full_t)
            pts = pts[:, idx_sort]
            w_pts = w_pts[idx_sort]
            w_pts = w_pts / np.max(w_pts)

            try:
                # TODO: how to make it more robust? Weight centers by involvement sometims make trajectories overally biased at narrow region
                # tck, u = splprep(pts, w=w_pts, s=s)
                tck, u = splprep(pts)
                u_fine = np.linspace(0, 1, n_bins * 10)
                traj_fine = np.array(splev(u_fine, tck)).T
                trajs.append(traj_fine)

                from scipy.spatial.distance import cdist

                edge_distances = cdist(c_arm, traj_fine).min(axis=1)
                bandwidth = float(np.percentile(edge_distances, 75))

                traj_min_t = full_t[idx_sort[0]]
                traj_max_t = full_t[idx_sort[-1]]

                traj_analysis = TrajectoryAnalysis(
                    trajectory_coordinates=traj_fine,
                    trajectory_pseudotime_range=(
                        float(traj_min_t),
                        float(traj_max_t),
                    ),
                    n_bins=n_bins,
                    bandwidth_vertices=bandwidth,
                )
                traj_analyses.append(traj_analysis)
            except Exception:
                continue

        self.trajectory_analyses = traj_analyses

    def _compute_gene_trends(
        self,
        coordinates_vertices: np.ndarray,
        gene_expression_matrix: np.ndarray,
        gene_names: list[str],
        values_vertices: np.ndarray,
        confidence_level: float = 0.95,
        bandwidth_scale: float = 1.0,
        verbose: bool = False,
    ):
        from ..analyzing.gene_trend import compute_gene_trends_for_trajectories

        if len(self.trajectory_analyses) == 0:
            if verbose:
                from loguru import logger

                logger.warning("No trajectories available for gene trend analysis")
            return

        compute_gene_trends_for_trajectories(
            trajectory_analyses=self.trajectory_analyses,
            coordinates_vertices=coordinates_vertices,
            gene_expression_matrix=gene_expression_matrix,
            gene_names=gene_names,
            values_vertices=values_vertices,
            confidence_level=confidence_level,
            bandwidth_scale=bandwidth_scale,
            verbose=verbose,
        )

    def _compute_divergence(
        self,
        boundary_matrix_d0,
        *,
        edge_field_source: str = "edge_embedding_smooth",
        negate_for_source_positive: bool = True,
        smooth_half_window: int = DEFAULT_HALF_WINDOW,
    ) -> None:
        from scipy.sparse import csr_matrix

        from ..computing.divergence import (
            compute_divergence_from_edge_field,
            scatter_loop_edge_field_to_global,
        )

        rows, cols, vals = boundary_matrix_d0.data
        bd0 = csr_matrix((vals, (rows, cols)), shape=boundary_matrix_d0.shape)

        vertex_id_by_row = boundary_matrix_d0.row_simplex_ids
        vertex_row_lookup = {v: i for i, v in enumerate(vertex_id_by_row)}

        for loop_idx, loop in enumerate(self.selected_loop_classes):
            if loop.representatives is None or len(loop.representatives) == 0:
                continue

            if edge_field_source == "edge_gradient_raw":
                edge_values_per_rep = []
                for rep_idx, grad in enumerate(loop.edge_gradient_raw or []):
                    valid_idx = loop.valid_edge_indices_per_rep[rep_idx]
                    if valid_idx:
                        edge_values_per_rep.append(grad[valid_idx].flatten())
                    else:
                        edge_values_per_rep.append(np.array([]))
            else:
                edge_values_per_rep = getattr(loop, edge_field_source) or []

            if not edge_values_per_rep:
                continue

            edge_field_global = scatter_loop_edge_field_to_global(
                edge_values_per_rep=edge_values_per_rep,
                edge_masks_per_rep=self.edges_masks_loop_classes[loop_idx],
                edge_signs_per_rep=loop.edge_signs_per_rep,
                n_global_edges=boundary_matrix_d0.shape[1],
            )

            div_global = compute_divergence_from_edge_field(
                bd0, edge_field_global, negate_for_source_positive
            )

            rep_vertices = list(loop.representatives[0])
            if len(rep_vertices) > 1 and rep_vertices[0] == rep_vertices[-1]:
                rep_vertices = rep_vertices[:-1]

            div_loop = np.array(
                [
                    div_global[vertex_row_lookup[v]]
                    for v in rep_vertices
                    if v in vertex_row_lookup
                ]
            )

            if smooth_half_window > 0 and len(div_loop) > 0:
                div_smooth = smooth_along_loop_1d(
                    div_loop.astype(np.float64), smooth_half_window
                )
            else:
                div_smooth = div_loop.copy()

            loop.vertex_ids_divergence = rep_vertices
            loop.vertex_divergence_raw = div_loop
            loop.vertex_divergence_smooth = div_smooth
