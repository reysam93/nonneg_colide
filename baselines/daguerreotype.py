"""DAGuerreotype baseline wrapper around the original implementation.

Original code:
https://github.com/vzantedeschi/DAGuerreotype
"""

import importlib.util
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


_REPO_CANDIDATES = (
    Path(os.environ["DAGUERREOTYPE_DIR"]) if "DAGUERREOTYPE_DIR" in os.environ else None,
    Path(__file__).resolve().parents[1] / "code_aux" / "DAGuerreotype",
    Path(__file__).resolve().parents[1] / "code_aux" / "code_aux" / "DAGuerreotype",
)


class _WandbStub(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.log = lambda *args, **kwargs: None
        self.Image = lambda image: image
        self.init = lambda *args, **kwargs: self
        self.finish = lambda *args, **kwargs: None


class _CausalDagStub(types.ModuleType):
    def __init__(self):
        super().__init__("causaldag")

    class DAG:
        @classmethod
        def from_amat(cls, *args, **kwargs):
            raise RuntimeError("causaldag is required for DAGuerreotype CPDAG utilities.")


def _module_available(name):
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None


def _ensure_non_algorithm_stubs():
    if not _module_available("wandb"):
        sys.modules.setdefault("wandb", _WandbStub())
    if not _module_available("causaldag"):
        sys.modules.setdefault("causaldag", _CausalDagStub())


def _repo_path():
    for candidate in _REPO_CANDIDATES:
        if candidate is not None and (candidate / "daguerreo").is_dir():
            return candidate
    tried = ", ".join(str(path) for path in _REPO_CANDIDATES)
    raise FileNotFoundError(f"Could not find DAGuerreotype repository. Tried: {tried}")


def _load_original_modules():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    _ensure_non_algorithm_stubs()

    repo = None
    try:
        repo = _repo_path()
    except FileNotFoundError:
        pass
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from daguerreo import utils
        from daguerreo.models import Daguerro
    except ModuleNotFoundError as exc:
        missing = exc.name
        location = f" from {repo / 'linux-install.sh'}" if repo is not None else " with scripts/create_repro_env.sh"
        msg = (
            "DAGuerreotype cannot be imported because an original dependency is "
            f"missing: {missing!r}. Install the original dependencies{location} "
            "to run the faithful wrapper."
        )
        raise RuntimeError(msg) from exc
    return Daguerro, utils


class DAGuerreotype:
    """Faithful wrapper for the original DAGuerreotype training code."""

    def __init__(
        self,
        structure="sp_map",
        sparsifier="l0_ber_ste",
        equations="linear",
        loss="nll_ev",
        joint=False,
        seed=0,
        dtype=np.float64,
        nogpu=False,
        standardize=False,
        hidden=50,
        optimizer="adam",
        lr=1e-1,
        num_epochs=5000,
        es_tol=100,
        es_delta=1e-4,
        pruning_reg=0.001,
        l2_theta=0.0005,
        l2_eq=0.0005,
        lr_theta=1e-1,
        eq_optimizer="sgd",
        es_tol_inner=10,
        es_delta_inner=1e-4,
        num_inner_iters=200,
        init_theta="zeros",
        smap_init=False,
        smap_iter_k=100,
        smax_max_k=100,
        project="DAGuerreotype",
        results_path="./results/",
        entity="default",
    ):
        self.structure = structure
        self.sparsifier = sparsifier
        self.equations = equations
        self.loss = loss
        self.joint = joint
        self.seed = seed
        self.dtype = dtype
        self.nogpu = nogpu
        self.standardize = standardize
        self.hidden = hidden
        self.optimizer = optimizer
        self.lr = lr
        self.num_epochs = num_epochs
        self.es_tol = es_tol
        self.es_delta = es_delta
        self.pruning_reg = pruning_reg
        self.l2_theta = l2_theta
        self.l2_eq = l2_eq
        self.lr_theta = lr_theta
        self.eq_optimizer = eq_optimizer
        self.es_tol_inner = es_tol_inner
        self.es_delta_inner = es_delta_inner
        self.num_inner_iters = num_inner_iters
        self.init_theta = init_theta
        self.smap_init = smap_init
        self.smap_iter_k = smap_iter_k
        self.smax_max_k = smax_max_k
        self.project = project
        self.results_path = results_path
        self.entity = entity

    def _args(self, X, **overrides):
        values = {
            "project": self.project,
            "nogpu": self.nogpu,
            "wandb": False,
            "entity": self.entity,
            "results_path": self.results_path,
            "model": "daguerreo",
            "joint": self.joint,
            "structure": self.structure,
            "sparsifier": self.sparsifier,
            "equations": self.equations,
            "loss": self.loss,
            "hidden": self.hidden,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "es_tol": self.es_tol,
            "es_delta": self.es_delta,
            "pruning_reg": self.pruning_reg,
            "l2_theta": self.l2_theta,
            "l2_eq": self.l2_eq,
            "lr_theta": self.lr_theta,
            "eq_optimizer": self.eq_optimizer,
            "es_tol_inner": self.es_tol_inner,
            "es_delta_inner": self.es_delta_inner,
            "num_inner_iters": self.num_inner_iters,
            "init_theta": self.init_theta,
            "smap_init": self.smap_init,
            "smap_iter_k": self.smap_iter_k,
            "smax_max_k": self.smax_max_k,
            "standardize": self.standardize,
            "num_nodes": X.shape[1],
            "num_samples": X.shape[0],
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return Namespace(**values)

    def fit(self, X, seed=None, standardize=None, **overrides):
        """Run DAGuerreotype and store the estimated binary adjacency in ``W_est``."""
        Daguerro, utils = _load_original_modules()
        import torch

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)
        wandb_run = None
        try:
            X = np.asarray(X, dtype=self.dtype)
            args = self._args(X, standardize=standardize, **overrides)
            run_seed = self.seed if seed is None else seed
            utils.init_seeds(seed=run_seed)

            try:
                import wandb
                wandb_run = wandb.init(
                    project=args.project,
                    entity=args.entity,
                    reinit=True,
                    mode="disabled",
                )
            except Exception:
                wandb_run = None

            scaler = StandardScaler(with_std=args.standardize)
            X_scaled = scaler.fit_transform(X)
            X_torch = torch.from_numpy(np.asarray(X_scaled, dtype=self.dtype))

            daguerro = Daguerro.initialize(X_torch, args, args.joint)
            daguerro, X_torch = utils.maybe_gpu(args, daguerro, X_torch)

            loss_fun = utils.AVAILABLE[args.loss]
            self.log_dict_ = daguerro(X_torch, loss_fun, args)
            daguerro.eval()
            _, dags = daguerro(X_torch, loss_fun, args)

            self.model_ = daguerro
            self.args_ = args
            self.W_est = np.asarray(dags[0].detach().cpu().numpy(), dtype=self.dtype)
            self.W_raw = self.W_est.copy()
            return self.W_est
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            torch.set_default_dtype(old_dtype)


class DAGuerreotypeSparseMAP(DAGuerreotype):
    def __init__(self, **kwargs):
        super().__init__(structure="sp_map", **kwargs)


class DAGuerreotypeTopKSparseMax(DAGuerreotype):
    def __init__(self, **kwargs):
        super().__init__(structure="tk_sp_max", **kwargs)
