# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from .types import Diameter_t, Index_t, PositiveFloat

if TYPE_CHECKING:
    import h5py


class LoopClass(BaseModel):
    rank: Index_t
    birth: Diameter_t = 0.0
    death: Diameter_t = 0.0
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

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "LoopClass"
        group.attrs["rank"] = self.rank
        group.attrs["birth"] = self.birth
        group.attrs["death"] = self.death

        kw = {"compression": "gzip"} if compress else {}

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

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> LoopClass:
        rank = int(group.attrs["rank"])  # type: ignore[arg-type]
        birth = float(group.attrs["birth"])  # type: ignore[arg-type]
        death = float(group.attrs["death"])  # type: ignore[arg-type]

        cocycles = None
        if "cocycles" in group:
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

        return cls(
            rank=rank,
            birth=birth,
            death=death,
            cocycles=cocycles,
            representatives=representatives,
            coordinates_vertices_representatives=coordinates_vertices_representatives,
        )


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
