# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator

from .types import Diameter_t, Index_t, Size_t
from .utils import decode_edges, decode_triangles


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


class BoundaryMatrixD0(BoundaryMatrix):
    @property
    def row_simplex_decode(self) -> list[Index_t]:
        return self.row_simplex_ids

    @property
    def col_simplex_decode(self) -> list[tuple[Index_t, Index_t]]:
        return decode_edges(np.array(self.col_simplex_ids), self.num_vertices)
