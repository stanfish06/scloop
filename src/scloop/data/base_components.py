# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from .types import Diameter_t, Index_t, Percent_t, PositiveFloat

if TYPE_CHECKING:
    import h5py


class PersistencePair(BaseModel):
    birth: Diameter_t
    death: Diameter_t
    birth_simplex: list[Index_t]
    death_simplex: list[Index_t]

    @model_validator(mode="after")
    def check_birth_death(self) -> Self:
        if self.birth > self.death:
            raise ValueError("loop dies before its birth")
        return self

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["birth"] = self.birth
        group.attrs["death"] = self.death
        kw = {"compression": "gzip"} if compress else {}
        group.create_dataset(
            "birth_simplex", data=np.asarray(self.birth_simplex, dtype=np.int64), **kw
        )
        group.create_dataset(
            "death_simplex", data=np.asarray(self.death_simplex, dtype=np.int64), **kw
        )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> PersistencePair:
        return cls(
            birth=float(group.attrs["birth"]),
            death=float(group.attrs["death"]),
            birth_simplex=np.asarray(group["birth_simplex"]).tolist(),
            death_simplex=np.asarray(group["death_simplex"]).tolist(),
        )


class ImagePairRecord(BaseModel):
    source_class_idx: Index_t
    source_pair: PersistencePair
    image_pair: PersistencePair | None = None

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["source_class_idx"] = self.source_class_idx
        self.source_pair.to_hdf5_group(
            group.create_group("source_pair"), compress=compress
        )
        if self.image_pair is not None:
            self.image_pair.to_hdf5_group(
                group.create_group("image_pair"), compress=compress
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> ImagePairRecord:
        image_pair = (
            PersistencePair.from_hdf5_group(group["image_pair"])
            if "image_pair" in group
            else None
        )
        return cls(
            source_class_idx=int(group.attrs["source_class_idx"]),
            source_pair=PersistencePair.from_hdf5_group(group["source_pair"]),
            image_pair=image_pair,
        )


class LoopClass(BaseModel):
    rank: Index_t
    persistence_index: Index_t | None = None
    birth: Diameter_t = 0.0
    death: Diameter_t = 0.0
    birth_simplex: list[Index_t] = Field(default_factory=list)
    death_simplex: list[Index_t] = Field(default_factory=list)
    cocycles: list | None = None
    representatives: list[list[Index_t]] | None = None
    coordinates_vertices_representatives: list[list[list[float]]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_birth_death(self) -> Self:
        if self.birth > self.death:
            raise ValueError("loop dies before its birth")
        return self

    @property
    def lifetime(self):
        return self.death - self.birth

    @property
    def persistence_pair(self) -> PersistencePair:
        return PersistencePair(
            birth=self.birth,
            death=self.death,
            birth_simplex=self.birth_simplex,
            death_simplex=self.death_simplex,
        )

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "LoopClass"
        group.attrs["rank"] = self.rank
        group.attrs["persistence_index"] = (
            self.rank if self.persistence_index is None else self.persistence_index
        )
        group.attrs["birth"] = self.birth
        group.attrs["death"] = self.death

        kw = {"compression": "gzip"} if compress else {}
        group.create_dataset(
            "birth_simplex", data=np.asarray(self.birth_simplex, dtype=np.int64), **kw
        )
        group.create_dataset(
            "death_simplex", data=np.asarray(self.death_simplex, dtype=np.int64), **kw
        )

        if self.cocycles is not None and len(self.cocycles) == 0:
            group.attrs["_cocycles_empty"] = True
        if self.cocycles is not None and len(self.cocycles) > 0:
            verts_list = []
            coeffs_list = []
            for simplex in self.cocycles:
                try:
                    verts, coeff = simplex
                    verts_list.append(list(verts))
                    coeffs_list.append(int(coeff))
                except (ValueError, TypeError):
                    continue
            if verts_list:
                cc_grp = group.create_group("cocycles")
                max_len = max(len(v) for v in verts_list)
                verts_arr = np.full((len(verts_list), max_len), -1, dtype=np.int64)
                for i, v in enumerate(verts_list):
                    verts_arr[i, : len(v)] = v
                cc_grp.create_dataset("vertices", data=verts_arr, **kw)
                cc_grp.create_dataset(
                    "coefficients", data=np.array(coeffs_list, dtype=np.int32), **kw
                )

        if self.representatives is not None:
            reps_grp = group.create_group("representatives")
            for i, rep in enumerate(self.representatives):
                reps_grp.create_dataset(
                    str(i), data=np.array(rep, dtype=np.int64), **kw
                )

        if self.coordinates_vertices_representatives is not None:
            coords_grp = group.create_group("coordinates_vertices_representatives")
            for i, coords in enumerate(self.coordinates_vertices_representatives):
                coords_grp.create_dataset(
                    str(i), data=np.array(coords, dtype=np.float64), **kw
                )

    @staticmethod
    def _read_base_fields(group: h5py.Group) -> dict:
        """Parse the fields common to every LoopClass into a kwargs dict."""
        rank = int(group.attrs["rank"])  # type: ignore[arg-type]
        persistence_index = int(group.attrs.get("persistence_index", rank))
        birth = float(group.attrs["birth"])  # type: ignore[arg-type]
        death = float(group.attrs["death"])  # type: ignore[arg-type]
        birth_simplex = (
            np.asarray(group["birth_simplex"], dtype=np.int64).tolist()
            if "birth_simplex" in group
            else []
        )
        death_simplex = (
            np.asarray(group["death_simplex"], dtype=np.int64).tolist()
            if "death_simplex" in group
            else []
        )

        # None vs empty-list distinction is preserved via the "_cocycles_empty"
        # marker written by to_hdf5_group.
        cocycles = None
        if group.attrs.get("_cocycles_empty", False):
            cocycles = []
        elif "cocycles" in group:
            cc_grp: h5py.Group = group["cocycles"]  # type: ignore[assignment]
            if "vertices" in cc_grp and "coefficients" in cc_grp:
                verts_arr = np.asarray(cc_grp["vertices"])
                coeffs_arr = np.asarray(cc_grp["coefficients"])
                cocycles = []
                for i in range(len(coeffs_arr)):
                    verts = [int(v) for v in verts_arr[i] if v >= 0]
                    cocycles.append((verts, int(coeffs_arr[i])))

        representatives = None
        if "representatives" in group:
            reps_grp: h5py.Group = group["representatives"]  # type: ignore[assignment]
            representatives = []
            for i in range(len(reps_grp)):
                representatives.append(np.asarray(reps_grp[str(i)]).tolist())

        coordinates_vertices_representatives = None
        if "coordinates_vertices_representatives" in group:
            coords_grp: h5py.Group = group["coordinates_vertices_representatives"]  # type: ignore[assignment]
            coordinates_vertices_representatives = []
            for i in range(len(coords_grp)):
                coordinates_vertices_representatives.append(
                    np.asarray(coords_grp[str(i)]).tolist()
                )

        return dict(
            rank=rank,
            persistence_index=persistence_index,
            birth=birth,
            death=death,
            birth_simplex=birth_simplex,
            death_simplex=death_simplex,
            cocycles=cocycles,
            representatives=representatives,
            coordinates_vertices_representatives=coordinates_vertices_representatives,
        )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> LoopClass:
        type_name = group.attrs.get("_type", "LoopClass")
        if isinstance(type_name, bytes):
            type_name = type_name.decode()

        if cls is LoopClass and type_name == "LoopClassAnalysis":
            from .analysis_containers import LoopClassAnalysis

            return LoopClassAnalysis.from_hdf5_group(group)

        return cls(**cls._read_base_fields(group))


# TODO: could consider define a class for a single loop


@dataclass
class PresenceTestResult:
    probabilities: list[PositiveFloat]
    odds_ratios: list[PositiveFloat]
    pvalues_raw: list[PositiveFloat]
    pvalues_corrected: list[PositiveFloat]

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "PresenceTestResult"
        kw = {"compression": "gzip"} if compress else {}
        group.create_dataset(
            "probabilities", data=np.array(self.probabilities, dtype=np.float64), **kw
        )
        group.create_dataset(
            "odds_ratios", data=np.array(self.odds_ratios, dtype=np.float64), **kw
        )
        group.create_dataset(
            "pvalues_raw", data=np.array(self.pvalues_raw, dtype=np.float64), **kw
        )
        group.create_dataset(
            "pvalues_corrected",
            data=np.array(self.pvalues_corrected, dtype=np.float64),
            **kw,
        )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> PresenceTestResult:
        return cls(
            probabilities=np.asarray(group["probabilities"]).tolist(),
            odds_ratios=np.asarray(group["odds_ratios"]).tolist(),
            pvalues_raw=np.asarray(group["pvalues_raw"]).tolist(),
            pvalues_corrected=np.asarray(group["pvalues_corrected"]).tolist(),
        )


@dataclass
class PersistenceTestResult:
    pvalues_raw: list[PositiveFloat]
    pvalues_corrected: list[PositiveFloat]
    gamma_null_params: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "PersistenceTestResult"
        kw = {"compression": "gzip"} if compress else {}
        group.create_dataset(
            "pvalues_raw", data=np.array(self.pvalues_raw, dtype=np.float64), **kw
        )
        group.create_dataset(
            "pvalues_corrected",
            data=np.array(self.pvalues_corrected, dtype=np.float64),
            **kw,
        )
        if self.gamma_null_params is not None:
            group.create_dataset(
                "gamma_null_params",
                data=np.array(self.gamma_null_params, dtype=np.float64),
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> PersistenceTestResult:
        gamma_null_params = None
        if "gamma_null_params" in group:
            params = np.asarray(group["gamma_null_params"])
            gamma_null_params = (float(params[0]), float(params[1]), float(params[2]))
        return cls(
            pvalues_raw=np.asarray(group["pvalues_raw"]).tolist(),
            pvalues_corrected=np.asarray(group["pvalues_corrected"]).tolist(),
            gamma_null_params=gamma_null_params,
        )


@dataclass
class LoopClassEquivalence:
    n_loop_pairs_checked: int = 0
    loop_pairs_matched: list[tuple] = Field(default_factory=list)
    loop_pairs_matched_relax: list[tuple] = Field(default_factory=list)
    homotopy_coherence_matched: list[Percent_t | None] = Field(default_factory=list)
    homotopy_coherence_matched_relax: list[Percent_t | None] = Field(
        default_factory=list
    )
    mapping_deformation_matched: list = Field(default_factory=list)
    mapping_deformation_matched_relax: list = Field(default_factory=list)

    def is_equivalent(self, relax=False):
        if relax:
            return (
                len(self.loop_pairs_matched) > 0
                or len(self.loop_pairs_matched_relax) > 0
            )
        else:
            return len(self.loop_pairs_matched) > 0
