from abc import ABC, abstractmethod
from typing import Any, Callable, NamedTuple

import numpy as np
from anndata import AnnData
from pydantic import BaseModel, ConfigDict, Field, model_validator

"""
============================== Dataset generation ==============================
1. start with 2D dynamical system (assumed underlying dynamical system)
    ┌───┐  ┌─────────────────────────────────────────────────────────────────┐
     dx1    f11(t)x1 + f12(t)x2 + g11(x1, t)dW11 + g12(x2, t)dW12 + h1(x1, t)
          =
     dx2    f21(t)x1 + f22(t)x2 + g21(x1, t)dW21 + g22(x2, t)dW12 + h2(x2, t)
    └───┘  └─────────────────────────────────────────────────────────────────┘
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

    data: AnnData | list[AnnData] | None
    meta: BenchDataMeta
    runners: list[MethodRunner] = Field(default_factory=list)
    results: list[BenchResult] = Field(default_factory=list)

    @abstractmethod
    def generate_data(self, **kwargs):
        pass

    @abstractmethod
    def run_methods(self):
        pass

    @abstractmethod
    def evaluate_results(self):
        pass


class BenchSingleData(BenchContainer, ABC):
    @model_validator(mode="after")
    def check_match_data(self):
        # Allow None (data not yet generated), but reject wrong types
        if self.data is not None and not isinstance(self.data, AnnData):
            raise ValueError(f"Expected AnnData, got {type(self.data)}")
        return self

    def run_methods(self):
        if self.data is None:
            raise RuntimeError("Data not generated. Call generate_data() first.")
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


def _zero(*args, **kwargs):
    return 0.0


class BenchDiffEq(BaseModel, ABC):
    @abstractmethod
    def solve(
        self,
        t0,
        t1,
        dt,
        y0,
        *,
        t_eval: list | np.ndarray | None = None,
        **integrator_kwargs,
    ):
        pass


class BenchODE(BenchDiffEq):
    F_terms: list[list[Callable]]
    F_jacobian: list[Callable]  # diagonal: F_jacobian[i] = ∂f_i/∂y_i
    force_func: list[Callable | None] | None = None
    force_func_jac: list[Callable | None] | None = None

    @property
    def ndims(self):
        return len(self.F_terms)

    def _resolve(self, ff: list[Callable | None] | None) -> list[Callable]:
        if ff is None:
            return [_zero] * self.ndims
        return [fn if fn is not None else _zero for fn in ff]

    @property
    def f(self) -> Callable:
        n = self.ndims
        forces = self._resolve(self.force_func)

        def _f(t, y):
            return [
                sum(self.F_terms[i][j](t, y[j]) for j in range(n)) + forces[i](t, y[i])
                for i in range(n)
            ]

        return _f

    @property
    def jac(self) -> Callable:
        n = self.ndims
        forces = self._resolve(self.force_func_jac)

        def _jac(t, y):
            return [
                [
                    (self.F_jacobian[i](t, y[i]) + forces[i](t, y[i]))
                    if i == j
                    else 0.0
                    for j in range(n)
                ]
                for i in range(n)
            ]

        return _jac

    def solve(
        self,
        t0,
        t1,
        dt,
        y0,
        *,
        t_eval: list | np.ndarray | None = None,
        **integrator_kwargs,
    ):
        from scipy.integrate import solve_ivp

        return solve_ivp(
            fun=self.f,
            t_span=(t0, t1),
            y0=y0,
            jac=self.jac,
            t_eval=np.arange(t0, t1 + dt, dt) if t_eval is None else t_eval,
            **integrator_kwargs,
        )


# TODO: implement solver
class BenchSDE(BenchDiffEq):
    pass


class ModelParam(BaseModel, ABC):
    @abstractmethod
    def build(self) -> BenchDiffEq: ...


class BenchODEParam(ModelParam):
    F_terms: list[list[Callable]]
    F_jacobian: list[Callable]
    force_func: list[Callable | None] | None = None
    force_func_jac: list[Callable | None] | None = None

    def build(self) -> BenchODE:
        return BenchODE(
            F_terms=self.F_terms,
            F_jacobian=self.F_jacobian,
            force_func=self.force_func,
            force_func_jac=self.force_func_jac,
        )


class TrajectoryConfig(NamedTuple):
    model: ModelParam
    initial_condition: list[float]


class EnsembleSpec(BaseModel, ABC):
    n_trajectories_per: int = 1
    seed: int = 0

    @abstractmethod
    def sample(self) -> list[TrajectoryConfig]: ...


class DynamicData(BenchSingleData):
    ensemble: EnsembleSpec
    t0: float = 0.0
    t1: float = 1.0
    dt: float = 0.01
    uneven_trajectory_sampling: bool = True
    embedding_dim: int = 200
    embedding_seed: int = 1
    low_dim_noise_std: float = 0.0
    high_dim_noise_std: float = 0.0

    def generate_data(self, **integrator_kwargs):
        def _embedding_isometric(X: np.ndarray, target_dim: int, seed: int = 1):
            import random

            source_dim = X.shape[1]
            np.random.seed(seed)
            random_matrix = np.random.randn(target_dim, target_dim)
            Q, _ = np.linalg.qr(random_matrix)
            Q_sub = Q[:, random.sample(range(target_dim), source_dim)]
            return X @ Q_sub.T

        def _latin_hypercube_time_sampling(
            n_samples: int,
            n_t_bins_fine_ratio: int = 100,
            noise_const: float = 1.0,
            evenness_const: float = 0.1,
        ):
            n_t_bins_fine = n_samples * n_t_bins_fine_ratio
            random_probs = (
                np.tanh(np.random.normal(scale=noise_const, size=n_t_bins_fine)) + 1
            ) + evenness_const
            random_probs = random_probs / np.sum(random_probs)
            empirical_cdf = np.cumsum(random_probs)
            bin_width = 1.0 / n_samples
            offsets = np.arange(n_samples) * bin_width
            t_vals = np.random.uniform(size=n_samples) * bin_width + offsets
            return np.searchsorted(empirical_cdf, t_vals) / n_t_bins_fine

        configs = self.ensemble.sample()
        trajectories = []
        traj_ids = []
        tid = 0
        for config in configs:
            for _ in range(self.ensemble.n_trajectories_per):
                if self.uneven_trajectory_sampling:
                    t_evals = self.t0 + (self.t1 - self.t0) * (
                        _latin_hypercube_time_sampling(
                            int((self.t1 - self.t0) / self.dt)
                        )
                    )
                else:
                    t_evals = None
                diffeq = config.model.build()
                sol = diffeq.solve(
                    t0=self.t0,
                    t1=self.t1,
                    dt=self.dt,
                    y0=config.initial_condition,
                    t_eval=t_evals,
                    **integrator_kwargs,
                )
                assert sol is not None
                traj = np.asarray(sol.y).T
                trajectories.append(traj)
                traj_ids.extend([tid] * traj.shape[0])
                tid += 1

        X_clean = np.vstack(trajectories)
        rng = np.random.default_rng(self.embedding_seed)
        X_low = X_clean
        if self.low_dim_noise_std > 0:
            X_low = X_low + rng.normal(0.0, self.low_dim_noise_std, X_low.shape)
        X_high = _embedding_isometric(
            X_low, self.embedding_dim, seed=self.embedding_seed
        )
        if self.high_dim_noise_std > 0:
            X_high = X_high + rng.normal(0.0, self.high_dim_noise_std, X_high.shape)

        self.data = AnnData(X=X_high)
        self.data.obs["trajectory_id"] = np.asarray(traj_ids)
        self.meta.true_trajectories = [traj.tolist() for traj in trajectories]


class SplatterData(BenchSingleData):
    pass


class RealData(BenchSingleData):
    pass


class BenchDataCollection(BenchContainer, ABC):
    @model_validator(mode="after")
    def check_match_data(self):
        # Allow None (data not yet generated)
        if self.data is None:
            raise ValueError(
                "Cross-match benchmark requires individually processed datasets"
            )
        if not isinstance(self.data, list) or len(self.data) < 2:
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
