# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from ..data.analysis_containers import BootstrapAnalysis
from ..data.base_components import LoopClass
from ..data.types import MultipleTestCorrectionMethod


def test_loops_significance(
    bootstrap_data: BootstrapAnalysis,
    selected_loop_classes: list[LoopClass | None],
    method_pval_correction: MultipleTestCorrectionMethod = "benjamini-hochberg",
) -> None:
    if bootstrap_data.num_bootstraps == 0:
        return

    bootstrap_data.fisher_presence_results = bootstrap_data.fisher_test_presence(
        method_pval_correction=method_pval_correction
    )

    bootstrap_data.gamma_persistence_results = bootstrap_data.gamma_test_persistence(
        selected_loop_classes, method_pval_correction
    )
