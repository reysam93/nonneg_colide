#!/usr/bin/env python3
# coding: utf-8

# Different-variances experiment for the nonnegative CoLiDE paper. Mirrors
# NOMAD's synthetic/scripts/different_vars.py (variance grid, normalized SHD, saved
# tables and joint hom-vs-hetero figures) with the same methods as
# experiments/scripts/number_samples.py, which are imported from there so
# both experiments always share the same baselines and hyperparameters.
# Two scenarios run sequentially: homoscedastic (all nodes with variance
# `var`) and heteroscedastic (per-node profile sigma_j^2 ~ U(0.5, 5)^2 scaled
# by `var`). Methods that estimate Sigma also report its scale-invariant error.
#
# Background run (from the repo root):
#   nohup .venv-dag/bin/python experiments/scripts/different_vars.py \
#       > logs/salida_nonneg_colide_vars.txt 2>&1 &
# Environment switches:
#   SCENARIOS=hom,hetero   run only these scenarios
#   N_DAGS=20              override the number of DAGs
#   SMOKE_TEST=1           tiny run to check the pipeline
# The notebook experiments/notebooks/different_vars.ipynb imports this
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
from time import perf_counter

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

import src.utils as utils
import experiments.scripts.number_samples as ns

# Shared with the number-of-samples experiment
build_experiments = ns.build_experiments
compute_sigma_errors = ns.compute_sigma_errors
fit_model = ns.fit_model
get_lamb_value = ns.get_lamb_value
seed_task = ns.seed_task
log_status = ns.log_status
sigma_skip_idx = ns.sigma_skip_idx
METRIC_NAMES = ns.METRIC_NAMES
SMOKE_TEST = ns.SMOKE_TEST
SEED = ns.SEED
N_CPUS = ns.N_CPUS
JOBLIB_VERBOSE = ns.JOBLIB_VERBOSE
SELECTED_SCENARIOS = ns.SELECTED_SCENARIOS
N_DAGS_OVERRIDE = ns.N_DAGS_OVERRIDE

SAVE = True
LOAD = False
PATH = str(ROOT / "results" / ("var_smoke" if SMOKE_TEST else "var")) + os.sep

N_DAGS = 50
THR = .2
VERB = True
VAR_VALUES = np.array([1, 5, 10, 15, 20, 25, 30])
HETERO_SIGMA_RANGE = (.5, 5.)   # sigma_j ~ U(range); profile var_j = sigma_j^2 (mean ~ 9), scaled by `var`
JOINT_AGGS = ("mean", "median")
SKIP_IDX = []

BASE_DATA_PARAMS = {
    "graph_type": "er",
    "n_nodes": 100,
    "edges": 4,  # Edges per node; converted to total edges inside run_var_exp.
    "edge_type": "positive",
    "w_range": (.5, 1),
    "n_samples": 1000,
}

np.random.seed(SEED)
random.seed(SEED)  # networkx graph generators use python's random module
os.makedirs(PATH, exist_ok=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def build_scenarios(data_p):
    rng = np.random.default_rng(SEED)
    hetero_profile = rng.uniform(low=HETERO_SIGMA_RANGE[0], high=HETERO_SIGMA_RANGE[1], size=data_p["n_nodes"]) ** 2

    return [
        {
            "name": "homoscedastic",
            "suffix": "hom",
            "data_params": data_p.copy(),
            "noise_profile": None,
            "line_style": "-",
        },
        {
            "name": "heteroscedastic",
            "suffix": "hetero",
            "data_params": data_p.copy(),
            "noise_profile": hetero_profile,
            "line_style": "--",
        },
    ]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_var_exp(g, data_p, var_values, exps, noise_profile=None, thr=.2, verb=False):
    metrics = {name: np.zeros((len(var_values), len(exps))) for name in METRIC_NAMES}
    metrics["err_sig"][:] = np.nan
    metrics["rel_err_sig"][:] = np.nan

    for i, var in enumerate(var_values):
        if g % N_CPUS == 0:
            log_status(f"Graph: {g + 1}, variance: {var}")

        seed_task(g, i)
        data_p_aux = data_p.copy()
        data_p_aux["edges"] *= data_p_aux["n_nodes"]
        data_p_aux["var"] = var if noise_profile is None else noise_profile * var
        data_p_aux["n_samples"] = (
            10 * data_p_aux["n_nodes"] if data_p_aux["n_samples"] is None else data_p_aux["n_samples"]
        )
        n_nodes, n_samples = data_p_aux["n_nodes"], data_p_aux["n_samples"]
        sigma_true = np.sqrt(np.broadcast_to(np.asarray(data_p_aux["var"], dtype=float), n_nodes))

        W_true, _, X = utils.simulate_sem(**data_p_aux)
        X_std = utils.standarize(X)
        W_true_bin = utils.to_bin(W_true, thr)
        norm_W_true = np.linalg.norm(W_true)

        for j, exp in enumerate(exps):
            X_aux = X_std if exp.get("standarize", False) else X

            arg_aux = exp["args"].copy()
            if exp.get("adapt_lamb", False):
                if "lamb" in arg_aux:
                    arg_aux["lamb"] = get_lamb_value(n_nodes, n_samples, arg_aux["lamb"])
                elif "lambda1" in arg_aux:
                    arg_aux["lambda1"] = get_lamb_value(n_nodes, n_samples, arg_aux["lambda1"])

            if exp.get("sigma_known", False) or exp.get("know_var", False):
                arg_aux["Sigma"] = data_p_aux["var"]

            try:
                model, W_est, Sigma_est, runtime = fit_model(exp, X_aux, arg_aux)
            except Exception as exc:
                raise RuntimeError(
                    f"DAG {g} var={var} method={exp['leg']} failed: {type(exc).__name__}: {exc}"
                ) from exc

            if np.isnan(W_est).any():
                W_est = np.zeros_like(W_est)
                W_est_bin = np.zeros_like(W_est)
            else:
                W_est_bin = utils.to_bin(W_est, thr)

            metrics["shd"][i, j], metrics["tpr"][i, j], metrics["fdr"][i, j] = \
                utils.count_accuracy(W_true_bin, W_est_bin)
            metrics["shd"][i, j] /= n_nodes  # normalized SHD, as in the original experiment
            metrics["fscore"][i, j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            metrics["err"][i, j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            metrics["rel_err"][i, j] = np.linalg.norm(W_true - W_est, "fro") ** 2 / norm_W_true ** 2
            if Sigma_est is not None:
                metrics["err_sig"][i, j], metrics["rel_err_sig"][i, j] = compute_sigma_errors(sigma_true, Sigma_est)
            metrics["acyc"][i, j] = (
                model.dagness(W_est) if hasattr(model, "dagness") else float(not utils.is_dag(W_est_bin))
            )
            metrics["runtime"][i, j] = runtime
            metrics["dag_count"][i, j] += 1 if utils.is_dag(W_est_bin) else 0

            if verb and (g % N_CPUS == 0):
                sig_text = f"  -  err_sig: {metrics['err_sig'][i, j]:.4f}" if Sigma_est is not None else ""
                log_status(
                    f"\t-{exp['leg']}: nshd {metrics['shd'][i, j]:.3f}  -  err: {metrics['err'][i, j]:.4f}"
                    f"{sig_text}  -  time: {runtime:.2f}"
                )

    return tuple(metrics[name] for name in METRIC_NAMES)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def vars_results_prefix(data_p, scenario_suffix):
    return f"{PATH}var_{scenario_suffix}_{data_p['graph_type'].upper()}graph_{data_p['edges']}N"


def save_vars_results(file_prefix, metrics, exps, var_values, scenario_suffix):
    os.makedirs(PATH, exist_ok=True)
    metrics = dict(zip(METRIC_NAMES, metrics))
    np.savez(file_prefix, exps=exps, xvals=var_values, scenario=scenario_suffix, **metrics)
    log_status(f"SAVED in file: {file_prefix}.npz")

    def save_stats(metric, tag, prctiles=True):
        data = metrics[metric]
        base = f"{PATH}vars_{scenario_suffix}_{tag}"
        utils.data_to_csv(f"{base}_mean.csv", exps, var_values, np.nanmean(data, axis=0))
        utils.data_to_csv(f"{base}_std.csv", exps, var_values, np.nanstd(data, axis=0))
        if prctiles:
            utils.data_to_csv(f"{base}_med.csv", exps, var_values, np.nanmedian(data, axis=0))
            utils.data_to_csv(f"{base}_prctile25.csv", exps, var_values, np.nanpercentile(data, 25, axis=0))
            utils.data_to_csv(f"{base}_prctile75.csv", exps, var_values, np.nanpercentile(data, 75, axis=0))

    save_stats("err", "err")
    save_stats("shd", "shd", prctiles=False)
    save_stats("err_sig", "errsig")
    save_stats("runtime", "time", prctiles=False)


def load_vars_results(file_prefix):
    file_name = f"{file_prefix}.npz"
    data = np.load(file_name, allow_pickle=True)
    log_status(f"Loaded variance results from {file_name}")
    metrics = tuple(data[name] for name in METRIC_NAMES)
    return (*metrics, data["exps"].tolist(), data["xvals"])


def run_or_load_vars_results(scenario, var_values, exps, n_dags, thr=.2, verb=False):
    if SELECTED_SCENARIOS and scenario["suffix"] not in SELECTED_SCENARIOS:
        log_status(f"SKIP scenario={scenario['suffix']} selected={sorted(SELECTED_SCENARIOS)}")
        return None

    data_p = scenario["data_params"]
    file_prefix = vars_results_prefix(data_p, scenario["suffix"])

    if LOAD:
        return load_vars_results(file_prefix)

    n_jobs = max(1, min(N_CPUS, n_dags))
    profile = scenario["noise_profile"]
    profile_text = "none" if profile is None else f"[{profile.min():.3g},{profile.max():.3g}] mean={profile.mean():.3g}"
    log_status(
        f"START scenario={scenario['name']} nodes={data_p['n_nodes']} edges_per_node={data_p['edges']} "
        f"samples={data_p['n_samples']} var_values={list(var_values)} noise_profile={profile_text} "
        f"dags={n_dags} methods={[exp['leg'] for exp in exps]} workers={n_jobs}"
    )

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_var_exp)(g, data_p, var_values, exps, profile, thr, verb)
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

    log_status(f"DONE scenario={scenario['name']} elapsed_minutes={(perf_counter() - t_init) / 60:.3f}")

    metrics = tuple(np.asarray(metric) for metric in zip(*results))
    if SAVE:
        save_vars_results(file_prefix, metrics, exps, var_values, scenario["suffix"])

    return (*metrics, exps, var_values)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_sigma_error(err_sig, exps, var_values, skip_idx=(), agg="median", deviation="prctile", alpha=.25,
                     title=None, figsize=(6, 4.5), ylabel="Sigma error (normalized)"):
    """Single panel with the scale-invariant Sigma error of the methods that estimate it."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    utils.plot_data(ax, err_sig, exps, var_values, "Noise variance", ylabel, sigma_skip_idx(exps, skip_idx),
                    agg=agg, deviation=deviation, alpha=alpha, plot_func="semilogy")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_results(metrics, exps, var_values, scenario_suffix, skip_idx=None, save=True):
    os.makedirs(PATH, exist_ok=True)
    m = dict(zip(METRIC_NAMES, metrics))
    skip = [] if skip_idx is None else list(skip_idx)
    base = f"{PATH}vars_{scenario_suffix}"

    for agg, dev in (("mean", "std"), ("median", "prctile")):
        fig, _ = utils.plot_shd_error_pair(
            m["shd"], m["err"], exps, var_values,
            xlabel="Noise variance",
            shd_ylabel="Normalized SHD",
            err_ylabel="Fro Error",
            skip_idx=skip,
            agg=agg,
            deviation=dev,
            alpha=0.25,
            shd_plot_func="plot",
            err_plot_func="semilogy",
            title=f"{scenario_suffix} - {agg}",
            figsize=(8, 4),
        )
        if save:
            fig.savefig(f"{base}_summary_{agg}.png", bbox_inches="tight")
            plt.close(fig)

        fig, _ = plot_sigma_error(m["err_sig"], exps, var_values, skip_idx=skip, agg=agg, deviation=dev,
                                  title=f"{scenario_suffix} - Sigma error - {agg}")
        if save:
            fig.savefig(f"{base}_sigma_error_{agg}.png", bbox_inches="tight")
            plt.close(fig)

        utils.plot_all_metrics(m["shd"], m["tpr"], m["fdr"], m["fscore"], m["err"], m["acyc"], m["runtime"],
                               m["dag_count"], var_values, exps, skip_idx=skip, agg=agg, dev=dev,
                               xlabel="Noise variance")
        plt.gcf().suptitle(f"{scenario_suffix} - all metrics - {agg}")
        if save:
            plt.savefig(f"{base}_all_metrics_{agg}.png", bbox_inches="tight")
            plt.close("all")


def scenario_experiments(exps, suffix, line_style):
    return [
        {**exp, "leg": f"{exp['leg']}-{suffix}", "fmt": exp.get("fmt", "o-")[0] + line_style}
        for exp in exps
    ]


def joint_data(scenario_results, metric, skip_idx=None, sigma_only=False):
    """Concatenate one metric of every scenario along the method axis with '-hom'/'-hetero' legends."""
    skip = set([] if skip_idx is None else skip_idx)
    parts, joint_exps = [], []
    for result in scenario_results:
        exps = result["exps"]
        keep_idx = [i for i in range(len(exps)) if i not in skip and (not sigma_only or exps[i].get("sigma", False))]
        parts.append(result["metrics"][METRIC_NAMES.index(metric)][:, :, keep_idx])
        joint_exps.extend(scenario_experiments([exps[i] for i in keep_idx], result["scenario"]["suffix"],
                                               result["scenario"]["line_style"]))
    return np.concatenate(parts, axis=2), joint_exps


def plot_joint_results(scenario_results, agg="mean", skip_idx=None, save=True):
    os.makedirs(PATH, exist_ok=True)
    if len(scenario_results) < 2:
        return

    reference_xvals = scenario_results[0]["var_values"]
    for result in scenario_results[1:]:
        if not np.array_equal(reference_xvals, result["var_values"]):
            raise ValueError("Cannot plot joint results with different variance grids")

    deviation = "std" if agg == "mean" else "prctile"
    shd_joint, joint_exps = joint_data(scenario_results, "shd", skip_idx)
    err_joint, _ = joint_data(scenario_results, "err", skip_idx)

    fig, _ = utils.plot_shd_error_pair(
        shd_joint, err_joint, joint_exps, reference_xvals,
        xlabel="Noise variance",
        shd_ylabel="Normalized SHD",
        err_ylabel="Fro Error",
        skip_idx=[],
        agg=agg,
        deviation=deviation,
        alpha=0.25,
        shd_plot_func="plot",
        err_plot_func="semilogy",
        title=f"hom vs hetero - {agg}",
        figsize=(10, 5),
    )
    if save:
        fig.savefig(f"{PATH}vars_hom_hetero_joint_{agg}.png", bbox_inches="tight")
        plt.close(fig)

    errsig_joint, sig_exps = joint_data(scenario_results, "err_sig", skip_idx, sigma_only=True)
    fig, _ = plot_sigma_error(errsig_joint, sig_exps, reference_xvals, agg=agg, deviation=deviation,
                              title=f"hom vs hetero - Sigma error - {agg}", figsize=(7, 5))
    if save:
        fig.savefig(f"{PATH}vars_hom_hetero_joint_sigma_error_{agg}.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    n_dags = N_DAGS
    var_values = np.asarray(VAR_VALUES)
    if N_DAGS_OVERRIDE:
        n_dags = int(N_DAGS_OVERRIDE)
    if SMOKE_TEST:
        n_dags = min(n_dags, 2)
        var_values = np.array([1, 10])
        log_status("SMOKE TEST: tiny configuration, results go to a separate folder")

    exps = build_experiments()
    scenario_results = []
    for scenario in build_scenarios(BASE_DATA_PARAMS):
        out = run_or_load_vars_results(scenario, var_values, exps, n_dags, thr=THR, verb=VERB)
        if out is None:
            continue
        *metrics, scenario_exps, scenario_var_values = out
        plot_results(metrics, scenario_exps, scenario_var_values, scenario["suffix"], skip_idx=SKIP_IDX)
        scenario_results.append({
            "scenario": scenario,
            "metrics": metrics,
            "exps": scenario_exps,
            "var_values": scenario_var_values,
        })

    for agg in JOINT_AGGS:
        plot_joint_results(scenario_results, agg=agg, skip_idx=SKIP_IDX)


if __name__ == "__main__":
    main()
