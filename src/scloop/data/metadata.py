# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from .types import (
    CrossMatchModelTypes,
    EmbeddingMethod,
    EmbeddingNeighbors,
    FeatureSelectionMethod,
    IndexListDownSample,
    Percent_t,
    Size_t,
)

if TYPE_CHECKING:
    import h5py


class PreprocessMeta(BaseModel):
    library_normalized: bool
    target_sum: float
    feature_selection_method: FeatureSelectionMethod
    batch_key: str | None = None
    n_top_genes: int | None = None
    embedding_method: EmbeddingMethod | None = None
    embedding_neighbors: EmbeddingNeighbors | None = None
    scale_before_pca: bool
    n_pca_comps: int | None = None
    n_neighbors: int
    n_diffusion_comps: int | None = None
    scvi_key: str | None = None
    indices_downsample: IndexListDownSample | None = None
    kwargs_downsample: dict | None = None
    num_vertices: Size_t | None = None

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "PreprocessMeta"
        group.attrs["library_normalized"] = self.library_normalized
        group.attrs["target_sum"] = self.target_sum
        group.attrs["feature_selection_method"] = self.feature_selection_method
        group.attrs["scale_before_pca"] = self.scale_before_pca
        group.attrs["n_neighbors"] = self.n_neighbors

        if self.batch_key is not None:
            group.attrs["batch_key"] = self.batch_key
        if self.n_top_genes is not None:
            group.attrs["n_top_genes"] = self.n_top_genes
        if self.embedding_method is not None:
            group.attrs["embedding_method"] = self.embedding_method
        if self.embedding_neighbors is not None:
            group.attrs["embedding_neighbors"] = self.embedding_neighbors
        if self.n_pca_comps is not None:
            group.attrs["n_pca_comps"] = self.n_pca_comps
        if self.n_diffusion_comps is not None:
            group.attrs["n_diffusion_comps"] = self.n_diffusion_comps
        if self.scvi_key is not None:
            group.attrs["scvi_key"] = self.scvi_key
        if self.num_vertices is not None:
            group.attrs["num_vertices"] = self.num_vertices
        if self.indices_downsample is not None:
            kw = {"compression": "gzip"} if compress else {}
            group.create_dataset(
                "indices_downsample",
                data=np.array(self.indices_downsample, dtype=np.int64),
                **kw,
            )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> PreprocessMeta:
        kwargs_downsample = None
        indices_downsample = None
        if "indices_downsample" in group:
            indices_downsample = np.asarray(group["indices_downsample"]).tolist()

        feature_sel_method = str(group.attrs["feature_selection_method"])
        return cls(
            library_normalized=bool(group.attrs["library_normalized"]),
            target_sum=float(group.attrs["target_sum"]),  # type: ignore[arg-type]
            feature_selection_method=feature_sel_method,  # type: ignore[arg-type]
            batch_key=group.attrs.get("batch_key"),
            n_top_genes=group.attrs.get("n_top_genes"),
            embedding_method=group.attrs.get("embedding_method"),
            embedding_neighbors=group.attrs.get("embedding_neighbors"),
            scale_before_pca=bool(group.attrs["scale_before_pca"]),
            n_pca_comps=group.attrs.get("n_pca_comps"),
            n_neighbors=int(group.attrs["n_neighbors"]),  # type: ignore[arg-type]
            n_diffusion_comps=group.attrs.get("n_diffusion_comps"),
            scvi_key=group.attrs.get("scvi_key"),
            indices_downsample=indices_downsample,
            kwargs_downsample=kwargs_downsample,
            num_vertices=group.attrs.get("num_vertices"),
        )


# allow downsample as well?
# TODO: store parameters for bootstraping
# TODO: store significant loop data
class BootstrapMeta(BaseModel):
    indices_resample: list[IndexListDownSample] | None = None
    life_pct: Percent_t | None = None

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "BootstrapMeta"
        if self.life_pct is not None:
            group.attrs["life_pct"] = self.life_pct
        if self.indices_resample is not None:
            resample_grp = group.create_group("indices_resample")
            kw = {"compression": "gzip"} if compress else {}
            for i, indices in enumerate(self.indices_resample):
                resample_grp.create_dataset(
                    str(i), data=np.array(indices, dtype=np.int64), **kw
                )

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> BootstrapMeta:
        life_pct = group.attrs.get("life_pct")
        indices_resample = None
        if "indices_resample" in group:
            resample_grp: h5py.Group = group["indices_resample"]  # type: ignore[assignment]
            indices_resample = []
            for i in range(len(resample_grp)):
                indices_resample.append(np.asarray(resample_grp[str(i)]).tolist())
        return cls(indices_resample=indices_resample, life_pct=life_pct)


class ScloopMeta(BaseModel):
    preprocess: PreprocessMeta | None = None
    bootstrap: BootstrapMeta | None = None

    def to_hdf5_group(self, group: h5py.Group, compress: bool = True) -> None:
        group.attrs["_type"] = "ScloopMeta"
        if self.preprocess is not None:
            preprocess_grp = group.create_group("preprocess")
            self.preprocess.to_hdf5_group(preprocess_grp, compress=compress)
        if self.bootstrap is not None:
            bootstrap_grp = group.create_group("bootstrap")
            self.bootstrap.to_hdf5_group(bootstrap_grp, compress=compress)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> ScloopMeta:
        preprocess = None
        bootstrap = None
        if "preprocess" in group:
            preprocess_grp: h5py.Group = group["preprocess"]  # type: ignore[assignment]
            preprocess = PreprocessMeta.from_hdf5_group(preprocess_grp)
        if "bootstrap" in group:
            bootstrap_grp: h5py.Group = group["bootstrap"]  # type: ignore[assignment]
            bootstrap = BootstrapMeta.from_hdf5_group(bootstrap_grp)
        return cls(preprocess=preprocess, bootstrap=bootstrap)


class CrossDatasetMatchingMeta(BaseModel):
    shared_embedding_key: str
    reference_embedding_keys: list[str]
    reference_idx: int
    model_type: CrossMatchModelTypes
