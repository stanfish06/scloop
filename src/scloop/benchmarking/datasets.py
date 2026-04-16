from abc import ABC, abstractmethod
from typing import Any

from anndata import AnnData
from pydantic import BaseModel, ConfigDict, Field, model_validator

"""
============================== Dataset generation ==============================
1. start with 2D dynamical system (assumed underlying dynamical system)
    ┌───┐  ┌─────────────────────────────────────────────────────────────┐
     dx1    f11(t)x1 + f12(t)x2 + g11(x1, t)dW11 + g12(x2, t)dW12 + h1(t)
          =
     dx2    f21(t)x1 + f22(t)x2 + g21(x1, t)dW21 + g22(x2, t)dW12 + h2(t)
    └───┘  └────────────────────────────────────────────────────────────-┘
    - optionally, replace f, g, and/or h with some stochastic processes
(1.5). sample trajectories for stochastic systems
    - trajectoies can be same deterministically and differ by noise
    - or different deterministically due to different GRN/force dynamics
2. sample points along trajectories with Gaussian noise
3. linear/nonlinear embedding into high-D space
    - may combine with k meaningful genes and (n - k) background genes
4. benchmark different aspects:
    - current framework vs raw PH (threshold based loop detection)
    - preprocessing steps
        - feature selection
        - normalization
        - denoising
        - embedding
    - loop/trajectory reconstruction and true underlying trajectories
    - current framework vs other trajectory analysis tools (e.g. PAGA)
    - cross-datasets matching
    - computation efficiency
================================================================================
"""


class BenchDataMeta(BaseModel, ABC):
    n_true_loops: int | None = None
    true_trajectories: list[list[list[float]]] | None = None


class DynamicDataMeta(BenchDataMeta):
    pass


class SplatterDataMeta(BenchDataMeta):
    pass


class RealDataMeta(BenchDataMeta):
    pass


class BenchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method_name: str
    result_loop_finding: Any | None = None
    result_loop_analysis: Any | None = None
    result_loop_matching: Any | None = None


class MethodRunner(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method_name: str
    result: BenchResult | None = None

    @abstractmethod
    def prepare_adata(self, data):
        pass

    @abstractmethod
    def find_loops(self, data):
        pass

    @abstractmethod
    def analyze_loops(self, data):
        pass

    @abstractmethod
    def match_loops(self, data):
        pass


class ScloopRunner(MethodRunner):
    pass


class PAGARunner(MethodRunner):
    pass


class RawPHRunner(MethodRunner):
    pass


class BenchContainer(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: AnnData | list[AnnData]
    meta: BenchDataMeta
    runners: list[MethodRunner] = Field(default_factory=list)
    results: list[BenchResult] = Field(default_factory=list)

    @abstractmethod
    def run_methods(self):
        pass

    @abstractmethod
    def evaluate_results(self):
        pass


class BenchSingleData(BenchContainer, ABC):
    @model_validator(mode="after")
    def check_match_data(self):
        if type(self.data) is not AnnData:
            raise ValueError("Benchmark data is not a single AnnData object")
        # more checking to make sure each anndata has been processed
        return self

    def run_methods(self):
        self.results = []
        for runner in self.runners:
            runner.prepare_adata(self.data)
            runner.find_loops(self.data)
            runner.analyze_loops(self.data)
            runner.result = BenchResult(
                method_name=runner.method_name,
            )
            self.results.append(runner.result)

    def evaluate_results(self):
        pass


class DynamicData(BenchSingleData):
    pass


class SplatterData(BenchSingleData):
    pass


class RealData(BenchSingleData):
    pass


class BenchDataCollection(BenchContainer, ABC):
    @model_validator(mode="after")
    def check_match_data(self):
        if type(self.data) is not list and len(self.data) < 2:
            raise ValueError(
                "Benchmark data does not contain at least 2 AnnData objects"
            )
        # more checking to make sure each anndata has been processed
        return self

    def run_methods(self):
        self.results = []
        for runner in self.runners:
            matches = runner.match_loops(self.data)
            runner.result = BenchResult(
                method_name=runner.method_name,
            )
            self.results.append(runner.result)

    def evaluate_results(self):
        pass
