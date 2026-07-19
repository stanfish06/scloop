# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
from packaging.version import Version

if TYPE_CHECKING:
    from anndata import AnnData

    from ..data.containers import HomologyData

from ..data.constants import (
    CROSS_MATCH_RESULT_KEY,
    SCLOOP_META_UNS_KEY,
    SCLOOP_UNS_KEY,
)

SCHEMA_VERSION = "1.0.0"

# uns keys that hold non-anndata-serializable scloop objects
_SCLOOP_UNS_KEYS = (SCLOOP_UNS_KEY, SCLOOP_META_UNS_KEY, CROSS_MATCH_RESULT_KEY)

# group under which the companion (preprocessed) AnnData is embedded, paired
_ADATA_GROUP = "_adata"
_MINIFICATION_UNS_KEY = "scloop_minification"

_MIGRATIONS: dict[str, Callable[[h5py.Group], None]] = {}


def check_native_write_safe(adata: AnnData) -> None:
    """Raise if ``adata`` holds scloop objects that break ``write_h5ad``.

    ``anndata.write_h5ad`` cannot serialize scloop's rich ``uns`` objects and a
    partial write leaves a readable-but-empty ``.h5ad`` (silent data loss). Call
    this before a native write, or use :func:`strip_scloop_uns` to drop them.
    """
    present = [k for k in _SCLOOP_UNS_KEYS if k in adata.uns]
    if present:
        raise TypeError(
            f"adata.uns holds non-serializable scloop objects {present}; "
            "anndata.write_h5ad will crash. Use scloop.io.save_scloop(adata, path), "
            "or scloop.io.strip_scloop_uns(adata) to drop them first."
        )


def strip_scloop_uns(adata: AnnData) -> AnnData:
    """Drop scloop's non-serializable ``uns`` objects in place and return adata."""
    for k in _SCLOOP_UNS_KEYS:
        adata.uns.pop(k, None)
    return adata


def _normalize_version(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


def _check_schema_version(raw_version) -> None:
    version = _normalize_version(raw_version)
    try:
        found = Version(version)
        current = Version(SCHEMA_VERSION)
    except Exception:
        if version != SCHEMA_VERSION:
            raise ValueError(f"Schema version {version!r} not supported")
        return

    if found.major != current.major:
        raise ValueError(
            f"Schema version {version} is incompatible with the current "
            f"reader ({SCHEMA_VERSION}). No migration path is available; "
            "re-save the data with a matching scloop version."
        )
    if found > current:
        warnings.warn(
            f"File schema version {version} is newer than the reader "
            f"({SCHEMA_VERSION}); attempting to load anyway.",
            stacklevel=2,
        )
    if version in _MIGRATIONS:
        # migration hooks run lazily inside load_scloop once the file is open
        pass


def _build_companion_adata(adata: AnnData, minify: bool) -> AnnData:
    """Build the preprocessed companion adata to embed alongside HomologyData.

    Keeps full ``obs`` and every ``obsm``
    ``minify=True`` drops ``X`` and all ``layers``
    """
    import anndata as ad

    uns = {k: v for k, v in adata.uns.items() if k not in _SCLOOP_UNS_KEYS}

    dropped_layers: list[str] = []
    if minify:
        X = None
        layers: dict = {}
        dropped_layers = list(adata.layers.keys())
    else:
        X = adata.X
        layers = dict(adata.layers)

    companion = ad.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm=dict(adata.obsm),
        varm=dict(adata.varm),
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
        layers=layers,
        uns=uns,
    )
    companion.uns[_MINIFICATION_UNS_KEY] = {
        "level": "minified" if minify else "full",
        "dropped_X": bool(minify),
        "dropped_layers": dropped_layers,
        "n_obs": int(adata.n_obs),
    }
    return companion


def save_scloop(
    adata: AnnData,
    filepath: str | Path,
    compress: bool = True,
    overwrite: bool = False,
    save_adata: bool = True,
    minify: bool = False,
) -> None:
    """scloop results and the paired preprocessed adata (default)

    Parameters
    ----------
    save_adata
        Embed the preprocessed AnnData in the same file (group ``_adata``) so it
        stays paired one-to-one with the stored row indices. ``load_scloop_adata``
        reconstructs it. Set ``False`` to store only ``HomologyData``.
    minify
        scVI-style minification: drop ``X`` and ``layers`` from the embedded
        adata. Plotting and geometric re-analysis still work; ``sanity`` bootstrap
    """
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} exists. Use overwrite=True to replace.")

    if CROSS_MATCH_RESULT_KEY in adata.uns and SCLOOP_UNS_KEY not in adata.uns:
        raise ValueError(
            f"adata.uns holds cross-dataset matching results "
            f"('{CROSS_MATCH_RESULT_KEY}') but no '{SCLOOP_UNS_KEY}'. "
            "Use scloop.io.save_cross_match(matcher, path) instead."
        )

    if SCLOOP_UNS_KEY not in adata.uns:
        raise ValueError(f"No scloop data found in adata.uns['{SCLOOP_UNS_KEY}']")

    hd: HomologyData = adata.uns[SCLOOP_UNS_KEY]

    companion = _build_companion_adata(adata, minify) if save_adata else None

    fd, tmp_name = tempfile.mkstemp(
        dir=str(filepath.parent), prefix=filepath.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with h5py.File(tmp, "w") as f:
            schema_grp = f.create_group("_scloop_schema")
            schema_grp.attrs["version"] = SCHEMA_VERSION
            hd.to_hdf5_group(f, compress=compress)
            if companion is not None:
                from anndata.io import write_elem

                try:
                    write_elem(f, _ADATA_GROUP, companion)
                except Exception as e:
                    raise ValueError(
                        "Some objects in adata.uns cannot be serialized. Remove or convert the offending "
                        "entry, or call save_scloop(..., save_adata=False) to store "
                        f"only scloop results. Original error: {e!r}"
                    ) from e
        os.replace(tmp, filepath)
    finally:
        if tmp.exists():
            tmp.unlink()


def _open_validated(filepath: Path) -> h5py.File:
    f = h5py.File(filepath, "r")
    if "_scloop_schema" not in f:
        f.close()
        raise ValueError("Not a valid scloop HDF5 file (missing schema)")
    raw_version = f["_scloop_schema"].attrs["version"]
    _check_schema_version(raw_version)
    migrate = _MIGRATIONS.get(_normalize_version(raw_version))
    if migrate is not None:
        migrate(f)
    return f


def load_scloop(
    filepath: str | Path,
    adata: AnnData | None = None,
) -> HomologyData:
    from ..data.containers import HomologyData

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")

    with _open_validated(filepath) as f:
        hd = HomologyData.from_hdf5_group(f)

    if adata is not None:
        adata.uns[SCLOOP_UNS_KEY] = hd
        adata.uns[SCLOOP_META_UNS_KEY] = hd.meta

    return hd


def load_scloop_adata(filepath: str | Path) -> AnnData:
    """Reconstruct the embedded companion adata with scloop results attached.

    Returns the preprocessed AnnData saved by ``save_scloop(..., save_adata=True)``
    with ``uns['scloop']`` (HomologyData) and ``uns['scloop_meta']`` repopulated.
    Raises if the file was written with ``save_adata=False``.
    """
    from anndata.io import read_elem

    from ..data.containers import HomologyData

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")

    with _open_validated(filepath) as f:
        if _ADATA_GROUP not in f:
            raise ValueError(
                "No embedded AnnData in this file (saved with save_adata=False). "
                "Use load_scloop(path, adata=your_adata) instead."
            )
        adata = read_elem(f[_ADATA_GROUP])
        hd = HomologyData.from_hdf5_group(f)

    if hd.meta.preprocess is not None and hd.meta.preprocess.num_vertices is not None:
        if adata.n_obs != hd.meta.preprocess.num_vertices:
            warnings.warn(
                f"Embedded adata has {adata.n_obs} cells but scloop_meta expects "
                f"{hd.meta.preprocess.num_vertices}; stored row indices may not "
                "align with this adata.",
                stacklevel=2,
            )

    adata.uns[SCLOOP_UNS_KEY] = hd
    adata.uns[SCLOOP_META_UNS_KEY] = hd.meta
    return adata


__all__ = [
    "save_scloop",
    "load_scloop",
    "load_scloop_adata",
    "check_native_write_safe",
    "strip_scloop_uns",
]
