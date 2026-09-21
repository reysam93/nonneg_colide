#!/usr/bin/env python3
# coding: utf-8

# Number-of-samples experiment for the nonnegative CoLiDE paper. Mirrors
# NOMAD's synthetic/scripts/number_samples.py (same metrics, aggregation and saved
# tables) with a CoLiDE-centred set of methods, two noise scenarios run
# sequentially (var=1, heteroscedastic) and an extra metric: the
# scale-invariant error of the estimated noise std Sigma, for the methods
# that estimate it.
#
# Background run (from the repo root, N_CPUS optional):
#   nohup .venv-dag/bin/python experiments/scripts/number_samples.py \
#       > logs/salida_nonneg_colide_samples.txt 2>&1 &
# Environment switches:
#   SCENARIOS=er4_N100_var1,er4_N100_hetero   run only these scenarios
#   N_DAGS=20                                  override the number of DAGs
#   SMOKE_TEST=1                               tiny run to check the pipeline
# The notebook experiments/notebooks/number_samples.ipynb imports this
# module to load the saved results and plot them (or to run it optionally).

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random
import signal
from datetime import datetime
from time import perf_counter

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
import warnings

# Methods without Sigma leave NaN columns in err_sig; silence the nan-aggregation warnings.
for _msg in ("All-NaN slice encountered", "Mean of empty slice", "Degrees of freedom <= 0 for slice"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

import src.utils as utils
from src.model import MetMulDagma, MetMulColide

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear
from baselines.golem import GOLEM_TF_EV, GOLEM_TF_NV
from baselines.notears import notears_linear


SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 5)))
SELECTED_SCENARIOS = {
    scenario.strip()
    for scenario in os.environ.get("SCENARIOS", "").split(",")
    if scenario.strip()
}
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") not in ("", "0", "false", "False")
N_DAGS_OVERRIDE = os.environ.get("N_DAGS")

SAVE = True
LOAD = False
PATH = str(ROOT / "results" / ("samples_smoke" if SMOKE_TEST else "samples")) + os.sep

# Heteroscedastic scenario: sigma_j ~ U(HETERO_SIGMA_RANGE), var_j = sigma_j^2,
# drawn once (seeded) and shared by all DAGs, as in synthetic/scripts/different_vars.py.
HETERO_SIGMA_RANGE = (.5, 5.)

np.random.seed(SEED)
random.seed(SEED)  # networkx graph generators use python's random module
os.makedirs(PATH, exist_ok=True)


def log_status(message):
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp} pid={os.getpid()}] {message}", flush=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def seed_task(g, i=0):
    """
    Seed the numpy and python RNGs for DAG `g` and x-value index `i` inside the worker.
    Module-level seeds are re-executed by every joblib worker that imports this module,
    which made all workers generate the same DAG; per-task seeds give independent and
    reproducible data regardless of the worker assignment. They also neutralize the
    baselines that reset the global numpy RNG in their constructor (colide/dagma seed=0).
    """
    seed = (SEED * 1_000_003 + g * 1_009 + i) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
METRIC_NAMES = ("shd", "tpr", "fdr", "fscore", "err", "rel_err", "err_sig", "rel_err_sig",
                "acyc", "runtime", "dag_count")


def compute_sigma_errors(sigma_true, sigma_est):
    """
    Errors of the estimated noise std. `sigma_est` may be a scalar (EV methods) or a
    vector (NV methods); it is broadcast to the N nodes. `err_sig` normalizes both
    vectors before comparing them (same scale-invariant criterion as the Frobenius
    error of W), `rel_err_sig` is the plain relative squared error.
    """
    sigma_true = np.asarray(sigma_true, dtype=float).ravel()
    sigma_est = np.broadcast_to(np.asarray(sigma_est, dtype=float).ravel(), sigma_true.shape)
    if not np.all(np.isfinite(sigma_est)):
        return np.nan, np.nan
    err_sig = utils.compute_norm_sq_err(sigma_true, sigma_est)
    rel_err_sig = np.linalg.norm(sigma_true - sigma_est) ** 2 / np.linalg.norm(sigma_true) ** 2
    return err_sig, rel_err_sig


def fit_model(exp, X, args):
    """Fit one method and return (model, W_est, Sigma_est or None, runtime)."""
    if exp["model"] is notears_linear:
        # NoTEARS is a plain function (no model object, no Sigma estimate).
        t_init = perf_counter()
        W_est = notears_linear(X, **args)
        return None, W_est, None, perf_counter() - t_init

    model = exp["model"](**exp["init"]) if "init" in exp else exp["model"]()
    t_init = perf_counter()
    out = model.fit(X, **args)
    runtime = perf_counter() - t_init

    W_est = model.W_est
    Sigma_est = None
    if exp.get("sigma", False):
        if isinstance(out, tuple) and len(out) == 2:
            Sigma_est = out[1]
        elif hasattr(model, "sig_est"):
            Sigma_est = model.sig_est
        elif hasattr(model, "Sigma"):
            Sigma_est = model.Sigma
    return model, W_est, Sigma_est, runtime


def run_samples_exp(g, data_p, n_samples_values, exps, thr=.2, verb=False):
    metrics = {name: np.zeros((len(n_samples_values), len(exps))) for name in METRIC_NAMES}
    metrics["err_sig"][:] = np.nan
    metrics["rel_err_sig"][:] = np.nan

    sigma_true = np.sqrt(np.broadcast_to(np.asarray(data_p["var"], dtype=float), data_p["n_nodes"]))

    for i, n_samples in enumerate(n_samples_values):
        if g % N_CPUS == 0:
            log_status(f"Graph: {g + 1}, samples: {n_samples}")

        seed_task(g, i)
        data_p_aux = data_p.copy()
        data_p_aux["n_samples"] = n_samples

        W_true, _, X = utils.simulate_sem(**data_p_aux)
        X_std = utils.standarize(X)
        W_true_bin = utils.to_bin(W_true, thr)
        norm_W_true = np.linalg.norm(W_true)

        for j, exp in enumerate(exps):
            X_aux = X_std if exp.get("standarize", False) else X

            arg_aux = exp["args"].copy()
            if exp.get("adapt_lamb", False):
                if "lamb" in arg_aux:
                    arg_aux["lamb"] = get_lamb_value(data_p["n_nodes"], n_samples, arg_aux["lamb"])
                elif "lambda1" in arg_aux:
                    arg_aux["lambda1"] = get_lamb_value(data_p["n_nodes"], n_samples, arg_aux["lambda1"])

            try:
                model, W_est, Sigma_est, runtime = fit_model(exp, X_aux, arg_aux)
            except Exception as exc:
                raise RuntimeError(
                    f"DAG {g} samples={n_samples} method={exp['leg']} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            if np.isnan(W_est).any():
                W_est = np.zeros_like(W_est)
                W_est_bin = np.zeros_like(W_est)
            else:
                W_est_bin = utils.to_bin(W_est, thr)

            metrics["shd"][i, j], metrics["tpr"][i, j], metrics["fdr"][i, j] = \
                utils.count_accuracy(W_true_bin, W_est_bin)
            metrics["fscore"][i, j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            metrics["err"][i, j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            metrics["rel_err"][i, j] = np.linalg.norm(W_true - W_est, "fro") ** 2 / norm_W_true ** 2
            if Sigma_est is not None:
                metrics["err_sig"][i, j], metrics["rel_err_sig"][i, j] = \
                    compute_sigma_errors(sigma_true, Sigma_est)
            metrics["acyc"][i, j] = model.dagness(W_est) if hasattr(model, "dagness") else 1
            metrics["runtime"][i, j] = runtime
            metrics["dag_count"][i, j] += 1 if utils.is_dag(W_est_bin) else 0

            if verb and (g % N_CPUS == 0):
                sig_text = (f"  -  err_sig: {metrics['err_sig'][i, j]:.4f}"
                            if Sigma_est is not None else "")
                log_status(
                    f"\t-{exp['leg']}: shd {metrics['shd'][i, j]}  -  err: {metrics['err'][i, j]:.4f}"
                    f"{sig_text}  -  time: {runtime:.2f}"
                )

    return tuple(metrics[name] for name in METRIC_NAMES)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def samples_results_prefix(scenario_name):
    return f"{PATH}samples_{scenario_name}"


def save_samples_results(file_prefix, metrics, exps, n_samples_values, scenario_name):
    metrics = dict(zip(METRIC_NAMES, metrics))
    np.savez(file_prefix, exps=exps, xvals=n_samples_values, scenario=scenario_name, **metrics)
    log_status(f"SAVED in file: {file_prefix}.npz")

    def save_stats(metric, tag, prctiles=True):
        data = metrics[metric]
        utils.data_to_csv(f"{file_prefix}_{tag}_mean.csv", exps, n_samples_values, np.nanmean(data, axis=0))
        utils.data_to_csv(f"{file_prefix}_{tag}_std.csv", exps, n_samples_values, np.nanstd(data, axis=0))
        if prctiles:
            utils.data_to_csv(f"{file_prefix}_{tag}_med.csv", exps, n_samples_values, np.nanmedian(data, axis=0))
            utils.data_to_csv(f"{file_prefix}_{tag}_prctile25.csv", exps, n_samples_values,
                              np.nanpercentile(data, 25, axis=0))
            utils.data_to_csv(f"{file_prefix}_{tag}_prctile75.csv", exps, n_samples_values,
                              np.nanpercentile(data, 75, axis=0))

    save_stats("err", "err")
    save_stats("shd", "shd", prctiles=False)
    save_stats("err_sig", "errsig")
    save_stats("runtime", "time", prctiles=False)


def load_samples_results(scenario_name):
    file_name = f"{samples_results_prefix(scenario_name)}.npz"
    data = np.load(file_name, allow_pickle=True)
    log_status(f"Loaded samples results from {file_name}")
    metrics = tuple(data[name] for name in METRIC_NAMES)
    return (*metrics, data["exps"].tolist(), data["xvals"])


def run_or_load_samples_results(data_p, n_samples_values, exps, n_dags, scenario_name, thr=.2, verb=False):
    if SELECTED_SCENARIOS and scenario_name not in SELECTED_SCENARIOS:
        log_status(f"SKIP scenario={scenario_name} selected={sorted(SELECTED_SCENARIOS)}")
        return None

    if LOAD:
        return load_samples_results(scenario_name)

    var = np.asarray(data_p["var"], dtype=float)
    var_text = f"{var:.3g}" if var.ndim == 0 else f"hetero[{var.min():.3g},{var.max():.3g}] mean={var.mean():.3g}"
    n_jobs = max(1, min(N_CPUS, n_dags))
    log_status(
        f"START scenario={scenario_name} nodes={data_p['n_nodes']} edges={data_p['edges']} "
        f"var={var_text} dags={n_dags} samples={list(n_samples_values)} "
        f"methods={[exp['leg'] for exp in exps]} workers={n_jobs}"
    )

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_samples_exp)(g, data_p, n_samples_values, exps, thr, verb)
            for g in range(n_dags)
        )
    except (KeyboardInterrupt, SystemExit):
        backend = getattr(parallel, "_backend", None)
        if backend is not None and hasattr(backend, "terminate"):
            backend.terminate()
        raise
    finally:
        backend = getattr(parallel, "_backend", None)
        if backend is not None and hasattr(backend, "terminate"):
            backend.terminate()

    log_status(f"DONE scenario={scenario_name} elapsed_minutes={(perf_counter() - t_init) / 60:.3f}")

    metrics = tuple(np.asarray(metric) for metric in zip(*results))
    if SAVE:
        save_samples_results(samples_results_prefix(scenario_name), metrics, exps, n_samples_values, scenario_name)

    return (*metrics, exps, n_samples_values)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def build_experiments():
    """
    Nonnegative CoLiDE (SCA+Adam, NV and EV) with the hyperparameters of
    experiments/scripts/preliminary_exp.py; NOMAD, NoTEARS, DAGMA, CoLiDE and
    GOLEM (EV and NV) with the configuration of synthetic/scripts/number_samples.py.
    GOLEM-NV is initialized with the GOLEM-EV solution, as in the original paper.
    'sigma': True marks the methods that estimate the noise std.
    """
    nn_colide_args = {
        "stepsize": 3e-4,
        "step_type": "fixed",
        "alpha_0": .01,
        "rho_0": .05,
        "s": 1,
        "lamb": .1,
        "iters_in": 30000,
        "iters_out": 10,
        "beta": 2,
        "sca_adam": True,
    }
    colide_args = {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003}
    golem_args = {
        "lambda2": 5.0,
        "num_iter": 100000,
        "learning_rate": 1e-3,
        "w_threshold": 0.3,
        "postprocess": True,
        "checkpoint": None,
    }

    exps = [
        ### NONNEGATIVE COLIDE ###
        {
            "model": MetMulColide,
            "args": nn_colide_args.copy(),
            "init": {"primal_opt": "sca", "acyclicity": "logdet", "equal_var": False},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": True,
            "fmt": "o-",
            "leg": "NN-CoLiDE-NV",
        },
        {
            "model": MetMulColide,
            "args": nn_colide_args.copy(),
            "init": {"primal_opt": "sca", "acyclicity": "logdet", "equal_var": True},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": True,
            "fmt": "o--",
            "leg": "NN-CoLiDE-EV",
        },
        ### NOMAD (previous paper) ###
        {
            "model": MetMulDagma,
            "args": {
                "stepsize": 5e-3,
                "step_type": "fixed",
                "alpha_0": .1,
                "rho_0": .1,
                "s": 1,
                "lamb": .2,
                "iters_in": 5000,
                "iters_out": 10,
                "beta": 1.5,
            },
            "init": {"primal_opt": "adam", "acyclicity": "logdet"},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": False,
            "fmt": "s-",
            "leg": "NOMAD-adam",
        },
        ### BASELINES ###
        {
            "model": notears_linear,
            "args": {"loss_type": "l2", "lambda1": .1, "max_iter": 10},
            "adapt_lamb": False,
            "standarize": False,
            "sigma": False,
            "fmt": "D-",
            "leg": "NoTears",
        },
        {
            "model": DAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": False,
            "fmt": "^-",
            "leg": "DAGMA",
        },
        {
            "model": colide_ev,
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": True,
            "fmt": "v--",
            "leg": "CoLiDE-EV",
        },
        {
            "model": colide_nv,
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": True,
            "fmt": "v-",
            "leg": "CoLiDE-NV",
        },
        {
            "model": GOLEM_TF_EV,
            "args": {**golem_args, "lambda1": 2e-2},
            "adapt_lamb": False,
            "standarize": False,
            "sigma": False,
            "fmt": ">--",
            "leg": "GOLEM-EV",
        },
        {
            "model": GOLEM_TF_NV,
            "args": {**golem_args, "lambda1": 2e-3},
            "adapt_lamb": False,
            "standarize": False,
            "sigma": False,
            "fmt": ">-",
            "leg": "GOLEM-NV",
        },
    ]

    if SMOKE_TEST:
        for exp in exps:
            if exp["model"] is MetMulColide or exp["model"] is MetMulDagma:
                exp["args"].update({"iters_in": 200, "iters_out": 2})
            elif exp["model"] is notears_linear:
                exp["args"].update({"max_iter": 2})
            elif exp["model"] in (GOLEM_TF_EV, GOLEM_TF_NV):
                exp["args"].update({"num_iter": 200})
            else:
                exp["args"].update({"T": 2, "warm_iter": 100, "max_iter": 200})
    return exps


def build_scenarios(n_nodes=100):
    base = {
        "n_nodes": n_nodes,
        "graph_type": "er",
        "edges": 4 * n_nodes,
        "edge_type": "positive",
        "w_range": (.5, 1),
    }
    rng = np.random.default_rng(SEED)
    hetero_var = rng.uniform(low=HETERO_SIGMA_RANGE[0], high=HETERO_SIGMA_RANGE[1], size=n_nodes) ** 2

    return [
        {"name": f"er4_N{n_nodes}_var1", "title": "var = 1", "data_params": {**base, "var": 1}},
        {"name": f"er4_N{n_nodes}_hetero", "title": "heteroscedastic", "data_params": {**base, "var": hetero_var}},
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def sigma_skip_idx(exps, skip_idx=()):
    """Indices to skip in the Sigma plots: methods without Sigma plus the given ones."""
    return sorted(set(skip_idx) | {i for i, exp in enumerate(exps) if not exp.get("sigma", False)})


def plot_sigma_error(err_sig, exps, n_samples_values, skip_idx=(), agg="median", deviation="prctile",
                     alpha=.25, title=None, figsize=(6, 4.5), ylabel="Sigma error (normalized)"):
    """Single panel with the scale-invariant Sigma error of the methods that estimate it."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    utils.plot_data(ax, err_sig, exps, n_samples_values, "Number of samples", ylabel,
                    sigma_skip_idx(exps, skip_idx), agg=agg, deviation=deviation, alpha=alpha,
                    plot_func="loglog")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_results(metrics, exps, n_samples_values, scenario, skip_idx=(), save=True):
    metrics = dict(zip(METRIC_NAMES, metrics))
    prefix = samples_results_prefix(scenario["name"])
    skip = list(skip_idx)

    fig, _ = utils.plot_shd_error_pair(
        metrics["shd"], metrics["err"], exps, n_samples_values,
        xlabel="Number of samples",
        shd_ylabel="SHD",
        err_ylabel="Fro Error",
        skip_idx=skip,
        alpha=0.25,
        shd_agg="mean",
        shd_deviation="std",
        err_agg="median",
        err_deviation="prctile",
        err_plot_func="loglog",
        figsize=(10, 5),
        title=scenario["title"],
    )
    if save:
        fig.savefig(f"{prefix}_summary.png", bbox_inches="tight")

    fig, _ = plot_sigma_error(metrics["err_sig"], exps, n_samples_values, skip_idx=skip,
                              title=f"{scenario['title']} - Sigma error")
    if save:
        fig.savefig(f"{prefix}_sigma_error.png", bbox_inches="tight")

    for agg in ("mean", "median"):
        utils.plot_all_metrics(metrics["shd"], metrics["tpr"], metrics["fdr"], metrics["fscore"], metrics["err"],
                               metrics["acyc"], metrics["runtime"], metrics["dag_count"], n_samples_values, exps,
                               skip_idx=skip, agg=agg)
        if save:
            plt.savefig(f"{prefix}_all_metrics_{agg}.png", bbox_inches="tight")
    if save:
        plt.close("all")


# ---------------------------------------------------------------------------
def main():
    n_dags = 100
    n_samples_values = np.array([50, 60, 80, 100, 200, 500, 1000, 5000, 10000])
    thr = .2
    verb = True

    if N_DAGS_OVERRIDE:
        n_dags = int(N_DAGS_OVERRIDE)
    if SMOKE_TEST:
        n_dags = min(n_dags, 2)
        n_samples_values = np.array([100, 500])
        log_status("SMOKE TEST: tiny configuration, results go to a separate folder")

    exps = build_experiments()
    for scenario in build_scenarios(n_nodes=100):
        out = run_or_load_samples_results(
            scenario["data_params"], n_samples_values, exps, n_dags, scenario["name"], thr=thr, verb=verb
        )
        if out is None:
            continue
        *metrics, exps_out, xvals = out
        plot_results(metrics, exps_out, xvals, scenario)


if __name__ == "__main__":
    main()
