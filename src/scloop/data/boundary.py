# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator

from .types import Diameter_t, Index_t, Size_t
from .utils import decode_edges, decode_triangles

if TYPE_CHECKING:
    import h5py


class BoundaryMatrix(BaseModel, ABC):
    num_vertices: Size_t
    data: tuple[list[Index_t], list[Index_t], list[int]]
    shape: tuple[Size_t, Size_t]
    row_simplex_ids: list[Index_t]
    col_simplex_ids: list[Index_t]
    row_simplex_diams: list[Diameter_t]
    col_simplex_diams: list[Diameter_t]

    @field_validator(
        "row_simplex_ids",
        "row_simplex_diams",
        "col_simplex_ids",
        "col_simplex_diams",
        mode="before",
    )
    @classmethod
    def validate_fields(cls, v: list[Index_t], info: ValidationInfo):
        shape = info.data.get("shape")
        assert shape
        if info.field_name in ["row_simplex_ids", "row_simplex_diams"]:
            if len(v) != shape[0]:
                raise ValueError(
                    f"Length of {info.field_name} does not match the number of rows of the matrix"
                )
        elif info.field_name in ["col_simplex_ids", "col_simplex_diams"]:
            if len(v) != shape[1]:
                raise ValueError(
                    f"Length of {info.field_name} does not match the number of columns of the matrix"
                )
        return v

    @property
    @abstractmethod
    def row_simplex_decode(self) -> list:
        pass

    @property
    @abstractmethod
    def col_simplex_decode(self) -> list:
        pass

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["num_vertices"] = self.num_vertices
        group.attrs["shape"] = self.shape

        kw = {"compression": "gzip"} if compress else {}
        rows, cols, vals = self.data
        data_grp = group.create_group("data")
        data_grp.create_dataset("rows", data=np.array(rows, dtype=np.int64), **kw)
        data_grp.create_dataset("cols", data=np.array(cols, dtype=np.int64), **kw)
        data_grp.create_dataset("vals", data=np.array(vals, dtype=np.int8), **kw)

        group.create_dataset(
            "row_simplex_ids", data=np.array(self.row_simplex_ids, dtype=np.int64), **kw
        )
        group.create_dataset(
            "col_simplex_ids", data=np.array(self.col_simplex_ids, dtype=np.int64), **kw
        )
        group.create_dataset(
            "row_simplex_diams",
            data=np.array(self.row_simplex_diams, dtype=np.float64),
            **kw,
        )
        group.create_dataset(
            "col_simplex_diams",
            data=np.array(self.col_simplex_diams, dtype=np.float64),
            **kw,
        )

    @classmethod
    def _from_hdf5_group_data(cls, group: h5py.Group) -> dict:
        data_grp: h5py.Group = group["data"]  # type: ignore[assignment]
        rows = np.asarray(data_grp["rows"]).tolist()
        cols = np.asarray(data_grp["cols"]).tolist()
        vals = np.asarray(data_grp["vals"]).tolist()
        return {
            "num_vertices": int(group.attrs["num_vertices"]),  # type: ignore[arg-type]
            "data": (rows, cols, vals),
            "shape": tuple(group.attrs["shape"]),  # type: ignore[arg-type]
            "row_simplex_ids": np.asarray(group["row_simplex_ids"]).tolist(),
            "col_simplex_ids": np.asarray(group["col_simplex_ids"]).tolist(),
            "row_simplex_diams": np.asarray(group["row_simplex_diams"]).tolist(),
            "col_simplex_diams": np.asarray(group["col_simplex_diams"]).tolist(),
        }


class BoundaryMatrixD1(BoundaryMatrix):
    _cached_edge_set: set[tuple[Index_t, Index_t]] | None = None

    @property
    def row_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.row_simplex_ids), self.num_vertices)

    @property
    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t, Index_t]]:
        return decode_triangles(np.array(self.col_simplex_ids), self.num_vertices)

    @property
    def edge_set(self) -> set[tuple[Index_t, Index_t]]:
        if self._cached_edge_set is None:
            self._cached_edge_set = set(self.row_simplex_decode)
        return self._cached_edge_set

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "BoundaryMatrixD1"
        super().to_hdf5_group(group, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> BoundaryMatrixD1:
        data = cls._from_hdf5_group_data(group)
        return cls(**data)


class BoundaryMatrixD0(BoundaryMatrix):
    @property
    def row_simplex_decode(self) -> list[Index_t]:
        return self.row_simplex_ids

    @property
    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.col_simplex_ids), self.num_vertices)

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "BoundaryMatrixD0"
        super().to_hdf5_group(group, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> BoundaryMatrixD0:
        data = cls._from_hdf5_group_data(group)
        return cls(**data)
