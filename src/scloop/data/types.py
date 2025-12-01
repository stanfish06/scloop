# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Literal

FeatureSelectionMethod = Literal["hvg", "delve", "none"]
EmbeddingMethod = Literal["pca", "diffmap", "scvi"]
EmbeddingNeighbors = Literal["pca", "scvi"]
