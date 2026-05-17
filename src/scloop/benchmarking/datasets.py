from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Iterator, Literal, NamedTuple

import numpy as np
from anndata import AnnData
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

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
        method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
        **integrator_kwargs,
    ):
        pass


class BenchODE(BenchDiffEq):
    F_terms: list[list[Callable]]
    F_jacobian: list[list[Callable]]  # F_jacobian[i][j] = ∂f_i/∂y_j
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
                    self.F_jacobian[i][j](t, y[j])
                    + (forces[i](t, y[i]) if i == j else 0.0)
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
        method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
        **integrator_kwargs,
    ):
        from scipy.integrate import solve_ivp

        if t_eval is None:
            n_eval = int(np.round((t1 - t0) / dt)) + 1
            t_eval = np.linspace(t0, t1, n_eval)
        return solve_ivp(
            fun=self.f,
            t_span=(t0, t1),
            y0=y0,
            jac=self.jac,
            method=method,
            t_eval=t_eval,
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
    F_jacobian: list[list[Callable]]
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
    n_trajectories: int = 1


class EnsembleSpec(BaseModel, ABC):
    seed: int = 0

    @abstractmethod
    def sample(
        self, manual_configs: list[TrajectoryConfig] | None = None
    ) -> list[TrajectoryConfig]: ...


class DynEnsembleSpec(EnsembleSpec):
    def _fourier_basis(
        self,
        domain: tuple | list,
        freqs: np.ndarray,
        amps: np.ndarray,
        phases: np.ndarray,
    ) -> Callable:
        return lambda t, y: np.sum(
            amps
            * np.cos(
                freqs * 2 * np.pi * (t - domain[0]) / (domain[1] - domain[0]) + phases
            )
        )

    def _chebyshev_basis(
        self,
        domain: tuple | list,
        degs: np.ndarray,
        amps: np.ndarray,
    ):
        return lambda t, y: np.sum(
            amps
            * np.cos(
                degs
                * np.arccos((2 * t - (domain[0] + domain[1])) / (domain[1] - domain[0]))
            )
        )

    def random_forcing_generator(
        self,
        domain: tuple | list = (0, 1),
        recipe: Literal["fourier", "chebyshev"] = "fourier",
        n_basis: int = 6,
        with_gaussian_noise: bool = False,
        seed: int = 1,
        **kwargs,
    ) -> Iterator[Callable]:
        rng = np.random.default_rng(seed)
        amp_lo, amp_hi = kwargs.get("amp_range", (-1.0, 1.0))
        while True:
            if recipe == "fourier":
                lo, hi = kwargs["fourier_freq_range"]
                freqs = rng.uniform(lo, hi, size=n_basis)
                amps = rng.uniform(amp_lo, amp_hi, size=n_basis)
                phases = rng.uniform(0.0, 2 * np.pi, size=n_basis)
                yield self._fourier_basis(
                    domain=domain, freqs=freqs, amps=amps, phases=phases
                )
            else:
                lo, hi = kwargs["chebyshev_deg_range"]
                # this can produce repeated degrees (ignore for now, not a big concern)
                degs = rng.integers(lo, hi + 1, size=n_basis)
                amps = rng.uniform(amp_lo, amp_hi, size=n_basis)
                yield self._chebyshev_basis(domain=domain, degs=degs, amps=amps)


class LinearEnsembleSpec(DynEnsembleSpec):
    """Simulate linear trajectories. Assume start at steady state and drift away due to external forcing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ndims: int
    n_configs: int
    A: np.ndarray
    t_domain: tuple[float, float] = (0.0, 1.0)
    initial_condition: list[float] | None = None
    n_trajectories_per_config: int = 1
    forcing_recipe: Literal["fourier", "chebyshev"] = "fourier"
    forcing_n_basis: int = 6
    forcing_seed: int = 0
    forcing_amp_range: tuple[float, float] = (-1.0, 1.0)
    fourier_freq_range: tuple[float, float] = (0.0, 4.0)
    chebyshev_deg_range: tuple[int, int] = (0, 6)

    def sample(
        self, manual_configs: list[TrajectoryConfig] | None = None
    ) -> list[TrajectoryConfig]:
        if manual_configs is not None:
            return manual_configs
        ic = (
            self.initial_condition
            if self.initial_condition is not None
            else [0.0] * self.ndims
        )
        gen = self.random_forcing_generator(
            domain=self.t_domain,
            recipe=self.forcing_recipe,
            n_basis=self.forcing_n_basis,
            seed=self.forcing_seed,
            amp_range=self.forcing_amp_range,
            fourier_freq_range=self.fourier_freq_range,
            chebyshev_deg_range=self.chebyshev_deg_range,
        )
        configs = []
        for _ in range(self.n_configs):
            forces: list[Callable | None] = [next(gen) for _ in range(self.ndims)]
            F_terms: list[list[Callable]] = [
                [(lambda t, y, a=self.A[i, j]: a * y) for j in range(self.ndims)]
                for i in range(self.ndims)
            ]
            F_jacobian: list[list[Callable]] = [
                [(lambda t, y, a=self.A[i, j]: a) for j in range(self.ndims)]
                for i in range(self.ndims)
            ]
            model = BenchODEParam(
                F_terms=F_terms,
                F_jacobian=F_jacobian,
                force_func=forces,
                force_func_jac=None,
            )
            configs.append(
                TrajectoryConfig(
                    model=model,
                    initial_condition=ic,
                    n_trajectories=self.n_trajectories_per_config,
                )
            )
        return configs


class LoopEnsembleSpec(DynEnsembleSpec):
    """Simulate cyclic trajectories via 2D rotation `A = [[0, -ω], [ω, 0]]`. Optional forcing perturbs the orbit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    omega: float | list[float] = 2 * np.pi
    radius: float | list[float] = 1.0
    n_configs: int
    t_domain: tuple[float, float] = (0.0, 1.0)
    n_trajectories_per_config: int = 1
    phase1_fraction: float | list[float] = 0.9
    phase2_fraction: float | list[float] | None = 0.2
    return_strength: float | list[float] = 10.0
    with_forcing: bool = False
    forcing_scale: float = 0.01
    forcing_recipe: Literal["fourier", "chebyshev"] = "fourier"
    forcing_n_basis: int = 6
    forcing_seed: int = 0
    forcing_amp_range: tuple[float, float] = (-1.0, 1.0)
    fourier_freq_range: tuple[float, float] = (0.0, 4.0)
    chebyshev_deg_range: tuple[int, int] = (0, 6)

    def _broadcast(self, v: float | list[float], name: str) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v)] * self.n_configs
        if len(v) != self.n_configs:
            raise ValueError(
                f"{name} list length {len(v)} != n_configs {self.n_configs}"
            )
        return [float(x) for x in v]

    def _phase_schedule(self) -> tuple[list[float], list[float], list[float]]:
        span = self.t_domain[1] - self.t_domain[0]
        p1s = self._broadcast(self.phase1_fraction, "phase1_fraction")
        if self.phase2_fraction is None:
            p2s = [1.0 - p for p in p1s]
        else:
            p2s = self._broadcast(self.phase2_fraction, "phase2_fraction")
        t_switches = [self.t_domain[0] + p1 * span for p1 in p1s]
        t_ends = [ts + p2 * span for ts, p2 in zip(t_switches, p2s)]
        return p1s, t_switches, t_ends

    @property
    def solve_t_domain(self) -> tuple[float, float]:
        _, _, t_ends = self._phase_schedule()
        return (self.t_domain[0], max(t_ends))

    def sample(
        self, manual_configs: list[TrajectoryConfig] | None = None
    ) -> list[TrajectoryConfig]:
        if manual_configs is not None:
            return manual_configs
        omegas = self._broadcast(self.omega, "omega")
        radii = self._broadcast(self.radius, "radius")
        gen = (
            self.random_forcing_generator(
                domain=self.t_domain,
                recipe=self.forcing_recipe,
                n_basis=self.forcing_n_basis,
                seed=self.forcing_seed,
                amp_range=self.forcing_amp_range,
                fourier_freq_range=self.fourier_freq_range,
                chebyshev_deg_range=self.chebyshev_deg_range,
            )
            if self.with_forcing
            else None
        )
        _, t_switches, _ = self._phase_schedule()
        return_strengths = self._broadcast(self.return_strength, "return_strength")
        configs = []
        for k in range(self.n_configs):
            omega_k, radius_k = omegas[k], radii[k]
            t_switch = t_switches[k]
            rs_k = return_strengths[k]
            A1 = np.array([[0.0, -omega_k], [omega_k, 0.0]])
            A2 = -rs_k * np.eye(2)
            ic = [radius_k, 0.0]
            scale = self.forcing_scale * omega_k * radius_k
            F_terms: list[list[Callable]] = [
                [
                    (
                        lambda t, y, a1=A1[i, j], a2=A2[i, j], ts=t_switch: (
                            (a1 if t < ts else a2) * y
                        )
                    )
                    for j in range(2)
                ]
                for i in range(2)
            ]
            F_jacobian: list[list[Callable]] = [
                [
                    (
                        lambda t, y, a1=A1[i, j], a2=A2[i, j], ts=t_switch: (
                            a1 if t < ts else a2
                        )
                    )
                    for j in range(2)
                ]
                for i in range(2)
            ]
            if gen is not None:
                raw = [next(gen) for _ in range(2)]
                phase1_forces: list[Callable] = [
                    (lambda t, y, f=fn, s=scale: s * f(t, y)) for fn in raw
                ]
            else:
                phase1_forces = [_zero, _zero]
            phase2_drifts = [rs_k * radius_k, 0.0]
            forces: list[Callable | None] = [
                (
                    lambda t, y, p1=phase1_forces[i], p2=phase2_drifts[i], ts=t_switch: (
                        p1(t, y) if t < ts else p2
                    )
                )
                for i in range(2)
            ]
            model = BenchODEParam(
                F_terms=F_terms,
                F_jacobian=F_jacobian,
                force_func=forces,
                force_func_jac=None,
            )
            configs.append(
                TrajectoryConfig(
                    model=model,
                    initial_condition=ic,
                    n_trajectories=self.n_trajectories_per_config,
                )
            )
        return configs


class DynamicData(BenchSingleData):
    ensemble: EnsembleSpec
    t0: float = 0.0
    t1: float = 1.0
    dt: float = 0.01
    uneven_trajectory_sampling: bool = True
    embedding_dim: int = 200
    seed: int = 1
    low_dim_noise_std: float = 0.0
    high_dim_noise_std: float = 0.0

    @staticmethod
    def _embedding_isometric(X: np.ndarray, target_dim: int, rng):
        source_dim = X.shape[1]
        random_matrix = rng.standard_normal((target_dim, target_dim))
        Q, _ = np.linalg.qr(random_matrix)
        cols = rng.choice(target_dim, size=source_dim, replace=False)
        Q_sub = Q[:, cols]
        return X @ Q_sub.T, Q_sub

    @staticmethod
    def _latin_hypercube_time_sampling(
        n_samples: int,
        rng,
        n_t_bins_fine_ratio: int = 100,
        noise_const: float = 1.0,
        evenness_const: float = 0.1,
    ):
        n_t_bins_fine = n_samples * n_t_bins_fine_ratio
        random_probs = (
            np.tanh(rng.normal(scale=noise_const, size=n_t_bins_fine)) + 1
        ) + evenness_const
        random_probs = random_probs / np.sum(random_probs)
        empirical_cdf = np.concatenate([[0.0], np.cumsum(random_probs)])
        grid = np.arange(n_t_bins_fine + 1) / n_t_bins_fine
        bin_width = 1.0 / n_samples
        offsets = np.arange(n_samples) * bin_width
        t_vals = rng.uniform(size=n_samples) * bin_width + offsets
        return np.interp(t_vals, empirical_cdf, grid)

    def _simulate(
        self,
        diffeq: BenchDiffEq,
        ic,
        t_span: tuple[float, float],
        *,
        t_eval=None,
        dense_output: bool = False,
        method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
        **integrator_kwargs,
    ) -> Any:
        t0, t1 = t_span
        if t_eval is None and not dense_output:
            n_eval = int(np.round((t1 - t0) / self.dt)) + 1
            t_eval = np.linspace(t0, t1, n_eval)
        return diffeq.solve(
            t0=t0,
            t1=t1,
            dt=self.dt,
            y0=ic,
            t_eval=t_eval,
            method=method,
            dense_output=dense_output,
            **integrator_kwargs,
        )

    def _sample_trajectories(
        self,
        configs: list[TrajectoryConfig],
        t_span: tuple[float, float],
        rng,
        **integrator_kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        t0, t1 = t_span
        trajectories: list[np.ndarray] = []
        t_trajectories: list[np.ndarray] = []
        for config in configs:
            for _ in range(config.n_trajectories):
                if self.uneven_trajectory_sampling:
                    t_evals = t0 + (t1 - t0) * self._latin_hypercube_time_sampling(
                        int((t1 - t0) / self.dt), rng
                    )
                else:
                    t_evals = None
                diffeq = config.model.build()
                sol = self._simulate(
                    diffeq,
                    config.initial_condition,
                    (t0, t1),
                    t_eval=t_evals,
                    **integrator_kwargs,
                )
                assert sol is not None
                trajectories.append(np.asarray(sol.y).T)
                t_trajectories.append(np.asarray(sol.t))
        return trajectories, t_trajectories

    def generate_data(self, **integrator_kwargs):
        rng = np.random.default_rng(self.seed)
        ensemble_t_domain = getattr(
            self.ensemble, "solve_t_domain", getattr(self.ensemble, "t_domain", None)
        )
        t0, t1 = (
            ensemble_t_domain if ensemble_t_domain is not None else (self.t0, self.t1)
        )

        configs = self.ensemble.sample()
        trajectories, t_trajectories = self._sample_trajectories(
            configs, (t0, t1), rng, **integrator_kwargs
        )

        traj_ids: list[int] = []
        config_ids: list[int] = []
        tid = 0
        for cid, config in enumerate(configs):
            for _ in range(config.n_trajectories):
                n_points = trajectories[tid].shape[0]
                traj_ids.extend([tid] * n_points)
                config_ids.extend([cid] * n_points)
                tid += 1

        X_low = np.vstack(trajectories)
        if self.low_dim_noise_std > 0:
            X_low = X_low + rng.normal(0.0, self.low_dim_noise_std, X_low.shape)
        X_high, embedding_basis = self._embedding_isometric(
            X_low, self.embedding_dim, rng
        )
        if self.high_dim_noise_std > 0:
            X_high = X_high + rng.normal(0.0, self.high_dim_noise_std, X_high.shape)

        self.data = AnnData(X=X_high)
        self.data.obsm["X_true_manifold"] = X_low
        self.data.obs["trajectory_id"] = np.asarray(traj_ids)
        self.data.obs["config_id"] = np.asarray(config_ids)
        self.data.obs["t"] = np.concatenate(t_trajectories)
        self.data.uns["embedding_basis"] = embedding_basis
        self.meta.true_trajectories = [traj.tolist() for traj in trajectories]


@dataclass
class TreeNode:
    solution: Any
    y0: list[float]
    t_start: float
    t_end: float
    global_t: float
    depth: int


# make a compound data class alongside BenchSingleData?
class TreeData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_spec: LinearEnsembleSpec
    initial_condition: list[float]

    p_bisect: float
    p_add: float
    p_stop: float
    min_branch: int
    target_branches: int
    max_segments: int

    dt: float = 0.01
    embedding_dim: int = 200
    seed: int = 1
    low_dim_noise_std: float = 0.0
    high_dim_noise_std: float = 0.0
    uneven_trajectory_sampling: bool = True

    data: AnnData | None = None
    meta: BenchDataMeta = Field(default_factory=DynamicDataMeta)
    runners: list[MethodRunner] = Field(default_factory=list)
    results: list[BenchResult] = Field(default_factory=list)

    def _select_action(self, depth: int, branch_count: int, rng) -> str:
        can_extend = (
            depth + 1 <= self.max_segments and branch_count + 1 <= self.target_branches
        )
        can_stop = branch_count >= self.min_branch
        if not can_extend:
            return "stop"
        if not can_stop:
            actions = ["add", "bisect"]
            weights = np.asarray([self.p_add, self.p_bisect], dtype=float)
        else:
            actions = ["add", "bisect", "stop"]
            weights = np.asarray([self.p_add, self.p_bisect, self.p_stop], dtype=float)
        weights = weights / weights.sum()
        return str(rng.choice(actions, p=weights))

    def _spawn_segment(
        self,
        ic,
        parent_global_t: float,
        depth: int,
        forcing_seed: int,
        rng,
        integrator_kwargs: dict,
    ) -> tuple[TreeNode, np.ndarray, np.ndarray]:
        ic_list = list(np.asarray(ic).tolist()) if hasattr(ic, "tolist") else list(ic)
        spec = self.segment_spec.model_copy(
            update={
                "n_configs": 1,
                "n_trajectories_per_config": 1,
                "forcing_seed": forcing_seed,
            }
        )
        segment_dd = DynamicData(
            data=None,
            ensemble=spec,
            dt=self.dt,
            uneven_trajectory_sampling=self.uneven_trajectory_sampling,
            embedding_dim=self.embedding_dim,
            seed=int(rng.integers(0, 2**31 - 1)),
            low_dim_noise_std=0.0,
            high_dim_noise_std=0.0,
            meta=DynamicDataMeta(),
        )
        configs = spec.sample()
        config = configs[0]
        diffeq = config.model.build()
        t_span = spec.t_domain
        sol_dense = segment_dd._simulate(
            diffeq,
            config.initial_condition,
            t_span,
            dense_output=True,
            **integrator_kwargs,
        )
        assert sol_dense.sol is not None
        trajectories, t_trajectories = segment_dd._sample_trajectories(
            configs, t_span, rng, **integrator_kwargs
        )
        traj = trajectories[0] + np.array(ic_list)
        t_local = t_trajectories[0]
        sim_t_start, sim_t_end = t_span
        node_global_t = parent_global_t + (sim_t_end - sim_t_start)
        t_global = parent_global_t + (t_local - sim_t_start)
        node = TreeNode(
            solution=sol_dense.sol,
            y0=ic_list,
            t_start=sim_t_start,
            t_end=sim_t_end,
            global_t=node_global_t,
            depth=depth,
        )
        return node, traj, t_global

    def generate_data(self, **integrator_kwargs):
        rng = np.random.default_rng(self.seed)
        forcing_seed_counter = 0
        next_node_id = 0

        all_trajectories: list[np.ndarray] = []
        all_t_global: list[np.ndarray] = []
        all_depths: list[np.ndarray] = []
        all_node_ids: list[np.ndarray] = []

        root_ic = np.asarray(self.initial_condition, dtype=float)
        root_node, root_traj, root_t = self._spawn_segment(
            root_ic, 0.0, 0, forcing_seed_counter, rng, integrator_kwargs
        )
        forcing_seed_counter += 1
        all_trajectories.append(root_traj)
        all_t_global.append(root_t)
        all_depths.append(np.full(root_traj.shape[0], 0, dtype=int))
        all_node_ids.append(np.full(root_traj.shape[0], next_node_id, dtype=int))
        next_node_id += 1

        queue: deque[TreeNode] = deque([root_node])
        branch_count = 1

        while queue:
            node = queue.popleft()
            action = self._select_action(node.depth, branch_count, rng)

            if action == "stop":
                continue

            if action == "bisect":
                t_local = node.t_start + rng.uniform(0.25, 0.75) * (
                    node.t_end - node.t_start
                )
                child_ic = np.asarray(node.solution(t_local)) + np.array(node.y0)
                child_parent_global_t = node.global_t - (node.t_end - t_local)
                child_node, child_traj, child_t = self._spawn_segment(
                    child_ic,
                    child_parent_global_t,
                    node.depth + 1,
                    forcing_seed_counter,
                    rng,
                    integrator_kwargs,
                )
                forcing_seed_counter += 1
                all_trajectories.append(child_traj)
                all_t_global.append(child_t)
                all_depths.append(
                    np.full(child_traj.shape[0], node.depth + 1, dtype=int)
                )
                all_node_ids.append(
                    np.full(child_traj.shape[0], next_node_id, dtype=int)
                )
                next_node_id += 1
                queue.append(child_node)
                branch_count += 1
            elif action == "add":
                child_ic = np.asarray(node.solution(node.t_end)) + np.array(node.y0)
                for _ in range(2):
                    child_node, child_traj, child_t = self._spawn_segment(
                        child_ic,
                        node.global_t,
                        node.depth + 1,
                        forcing_seed_counter,
                        rng,
                        integrator_kwargs,
                    )
                    forcing_seed_counter += 1
                    all_trajectories.append(child_traj)
                    all_t_global.append(child_t)
                    all_depths.append(
                        np.full(child_traj.shape[0], node.depth + 1, dtype=int)
                    )
                    all_node_ids.append(
                        np.full(child_traj.shape[0], next_node_id, dtype=int)
                    )
                    next_node_id += 1
                    queue.append(child_node)
                branch_count += 1

        X_low = np.vstack(all_trajectories)
        if self.low_dim_noise_std > 0:
            X_low = X_low + rng.normal(0.0, self.low_dim_noise_std, X_low.shape)
        X_high, embedding_basis = DynamicData._embedding_isometric(
            X_low, self.embedding_dim, rng
        )
        if self.high_dim_noise_std > 0:
            X_high = X_high + rng.normal(0.0, self.high_dim_noise_std, X_high.shape)

        t_global_arr = np.concatenate(all_t_global)
        depths_arr = np.concatenate(all_depths)
        node_ids_arr = np.concatenate(all_node_ids)

        self.data = AnnData(X=X_high)
        self.data.obsm["X_true_manifold"] = X_low
        self.data.obs["t"] = t_global_arr
        self.data.obs["depth"] = depths_arr
        self.data.obs["node_id"] = node_ids_arr
        self.data.uns["embedding_basis"] = embedding_basis
        self.meta.true_trajectories = [traj.tolist() for traj in all_trajectories]


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
