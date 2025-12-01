# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass, field


@dataclass
class BootstrapAnalysis:
    loops_eidx_boot: list[list[list[np.ndarray]]] = field(default_factory=list)
    persistence_diagram_boot: list[np.ndarray] = field(default_factory=list)
    matching_df: list[pd.DataFrame] = field(default_factory=list)
    n_booted: int = 0
    loop_rank: pd.DataFrame = field(default_factory=pd.DataFrame)
    parameters: dict = field(default_factory=dict)

@dataclass
class HodgeAnalysis:
    loop_id: str
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    loops_edges_embedding: list[np.ndarray] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

@dataclass
class PseudotimeAnalysis:
    edge_pseudotime_deltas: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudotime_source: str = ""
    parameters: dict = field(default_factory=dict)

@dataclass
class VelocityAnalysis:
    edge_velocity_deltas: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity_source: str = ""
    parameters: dict = field(default_factory=dict)
