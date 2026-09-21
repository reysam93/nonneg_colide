#!/usr/bin/env python
# coding: utf-8

# Preliminary experiments for the nonnegative CoLiDE paper. Mirrors
# NOMAD's synthetic/scripts/preliminary_exp.py (same metrics, logs and saved tables)
# with a CoLiDE-centred set of methods and three noise scenarios.

# %%

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning:sklearn.linear_model._base")

import signal
import numpy as np
import pandas as pd
from numpy import linalg as la
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import f1_score
import random
import time
from datetime import datetime
from joblib import Parallel, delayed

from src.model import MetMulDagma, MetMulColide
import src.utils as utils

from baselines.colide import colide_ev, colide_nv
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model\._base")

SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 10)))
SELECTED_SCENARIOS = {
    scenario.strip()
    for scenario in os.environ.get("SCENARIOS", "").split(",")
    if scenario.strip()
}
LOAD = False
SAVE_RESULTS = True
RESULTS_PATH = ROOT / 'results' / 'preliminary'
os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(SEED)


def log_status(message):
    timestamp = datetime.now().isoformat(timespec='seconds')
    print(f'[{timestamp} pid={os.getpid()}] {message}', flush=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)
random.seed(SEED)


# %%

# Hyperparameters of the nonnegative CoLiDE variants come from
# experiments/notebooks/run_algorithms.ipynb. 'lamb' values are multipliers of
# sqrt(log(N)/M) (fix_lamb=False). NOMAD and the baselines keep the
# configuration of synthetic/scripts/preliminary_exp.py.
Exps = [
    ### NONNEGATIVE COLIDE - NV (one sigma per node) ###
    {'model': MetMulColide, 'args': {'stepsize': 5e-5, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-NV-fista'},

    {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 2}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-NV-adam'},

    {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .5, 'rho_0': .5, 's': 1, 'lamb': 1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 2, 'sca_adam': False}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca'},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-NV-sca'},

    {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 2, 'sca_adam': True}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca'},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-NV-sca-adam'},

    ### NONNEGATIVE COLIDE - EV (single shared sigma) ###
    {'model': MetMulColide, 'args': {'stepsize': 5e-5, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True, 'equal_var': True},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-EV-fista'},

    {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .5, 'rho_0': .5, 's': 1, 'lamb': 1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 2, 'sca_adam': False}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca', 'equal_var': True},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-EV-sca'},

    {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 30000,
     'iters_out': 10, 'beta': 2, 'sca_adam': True}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca', 'equal_var': True},
     'standarize': False, 'fix_lamb': False, 'leg': 'NN-CoLiDE-EV-sca-adam'},

    ### NOMAD (previous paper) ###
    {'model': MetMulDagma, 'args': {'stepsize': 5e-3, 'alpha_0': .1, 'rho_0': .1, 's': 1, 'lamb': .2, 'iters_in': 10000, 'step_type': 'fixed',
     'iters_out': 10, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-adam'},

    {'model': MetMulDagma, 'args': {'stepsize': 1e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .2, 'iters_in': 5000, 'step_type': 'fixed',
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-fista'},

    ### BASELINES ###
    # NonnegDAGMA
    {'model': NonnegativeDAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'NonDAGMA'},

    # Colide
    {'model': colide_ev, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-EV'},

    {'model': colide_nv, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-NV'},
]


# %%

def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def run_parallel_exps(data_p, exps, n_dags, scenario_name, thr=.3, verb=False):
    n_jobs = max(1, min(N_CPUS, n_dags))
    log_status(
        f'RUN scenario={scenario_name} dags={n_dags} workers={n_jobs} '
        f'joblib_verbose={JOBLIB_VERBOSE}'
    )
    t_init = time.time()

    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_exps)(g, data_p, exps, thr=thr, verb=verb)
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

    elapsed_time = (time.time() - t_init)/60
    log_status(f'DONE scenario={scenario_name} elapsed_minutes={elapsed_time:.3f}')
    return results


def run_exps(g, data_p, exps, thr=.3, verb=False):
    A_true, _, X = utils.simulate_sem(**data_p)
    A_true_bin = utils.to_bin(A_true, thr)
    X_std = utils.standarize(X)
    X_scale = X.std(axis=0)
    A_true_std = A_true * X_scale[:, None] / X_scale[None, :]

    M, N = X.shape

    Z = X - X @ A_true
    fidelity = 1/data_p['n_samples']*la.norm(Z, 'fro')**2
    fidelity_std = 1/data_p['n_samples']*la.norm(X_std - X_std @ A_true_std, 'fro')**2

    Sigma_hat = la.norm(Z, axis=0) / np.sqrt(M)

    log_status(
        f'DAG {g}: fidelity={fidelity:.3f} fidelity_std={fidelity_std:.3f} '
        f'sigma_min={np.min(Sigma_hat):.4f} sigma_max={np.max(Sigma_hat):.4f}'
    )

    shd, tpr, fdr, fscore, sid_norm, err, rel_err, err_th, acyc, runtime = [
        np.zeros(len(exps)) for _ in range(10)
    ]
    for i, exp in enumerate(exps):
        standardized = exp.get('standarize', False)
        X_aux = X_std if standardized else X
        A_true_aux = A_true_std if standardized else A_true

        args = exp['args'].copy()
        if 'fix_lamb' in exp.keys() and not exp['fix_lamb']:
            if 'lamb' in args:
                args['lamb'] = get_lamb_value(N, M, args['lamb'])
            elif 'lambda1' in args:
                args['lambda1'] = get_lamb_value(N, M, args['lambda1'])

        if 'know_var' in exp.keys() and exp['know_var']:
            args['Sigma'] = data_p['var']

        model = None
        try:
            model = exp['model'](**exp['init']) if 'init' in exp.keys() else exp['model']()
            t_i = time.time()
            model.fit(X_aux, **args)
            t_solved = time.time() - t_i

            A_est = model.W_est
            if getattr(model, 'cycle_repair_applied_', False):
                log_status(
                    f'DAG {g} method={exp["leg"]} exact-DAG safeguard removed '
                    f'{model.cycle_repair_count_} edge(s): '
                    f'{model.cycle_repair_edges_}'
                )
        except Exception as exc:
            raise RuntimeError(
                f'DAG {g} method={exp["leg"]} failed: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

        A_est_bin = utils.to_bin(A_est, thr)
        try:
            shd[i], tpr[i], fdr[i], sid_norm[i] = utils.count_accuracy(
                A_true_bin,
                A_est_bin,
                compute_sid=True,
                sid_normalize=True,
            )
        except ValueError:
            shd[i], tpr[i], fdr[i] = utils.count_accuracy(A_true_bin, A_est_bin)
            sid_norm[i] = np.nan
        fscore[i] = f1_score(A_true_bin.flatten(), A_est_bin.flatten())
        err[i] = utils.compute_norm_sq_err(A_true_aux, A_est)
        rel_err[i] = la.norm(A_true_aux - A_est, 'fro')**2 / la.norm(A_true_aux, 'fro')**2
        A_est_th = A_est.copy()
        A_est_th[np.abs(A_est_th) < thr] = 0
        err_th[i] = utils.compute_norm_sq_err(A_true_aux, A_est_th)
        acyc[i] = model.dagness(A_est) if model is not None and hasattr(model, 'dagness') else float(not utils.is_dag(A_est_bin))
        runtime[i] = t_solved

        if verb and (g % N_CPUS == 0):
            sid_text = f'{sid_norm[i]:.3f}' if np.isfinite(sid_norm[i]) else 'nan'
            log_status(
                f'DAG {g} method={exp["leg"]} standardized={standardized} '
                f'shd={shd[i]} tpr={tpr[i]:.3f} fdr={fdr[i]:.3f} sid={sid_text} '
                f'err={err[i]:.3f} rel_err={rel_err[i]:.3f} err_th={err_th[i]:.3f} '
                f'time={runtime[i]:.3f}'
            )

    return shd, tpr, fdr, fscore, sid_norm, err, rel_err, err_th, acyc, runtime


def preliminary_results_prefix(scenario_name):
    return f'{RESULTS_PATH}/preliminary_{scenario_name}'


def load_preliminary_results(scenario_name):
    tables = {}
    exps_leg = None
    for agg in ('mean', 'median'):
        file_name = f'{preliminary_results_prefix(scenario_name)}_{agg}.csv'
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'Results file not found: {file_name}')
        tables[agg] = pd.read_csv(file_name)
        if 'leg' not in tables[agg].columns:
            raise ValueError(f'Results file has no experiment legend column: {file_name}')

        table_exps_leg = tables[agg]['leg'].astype(str).tolist()
        if exps_leg is None:
            exps_leg = table_exps_leg
        elif table_exps_leg != exps_leg:
            raise ValueError(f'Experiment legends differ between saved tables for {scenario_name}')

        print(f'Loaded {agg} results from {file_name}')
        display(tables[agg])
    return tables, exps_leg


def run_or_load_preliminary_results(data_p, exps, n_dags, scenario_name, thr=.3, verb=False):
    if SELECTED_SCENARIOS and scenario_name not in SELECTED_SCENARIOS:
        log_status(f'SKIP scenario={scenario_name} selected={sorted(SELECTED_SCENARIOS)}')
        return None, None, None

    standardized_count = sum(exp.get('standarize', False) for exp in exps)
    mode = 'load' if LOAD else 'run'
    var = np.asarray(data_p['var'])
    var_text = f'{var:.3g}' if var.ndim == 0 else f'hetero[{var.min():.3g},{var.max():.3g}] mean={var.mean():.3g}'
    log_status(
        f'START scenario={scenario_name} mode={mode} nodes={data_p["n_nodes"]} '
        f'samples={data_p["n_samples"]} edges={data_p["edges"]} var={var_text} dags={n_dags} '
        f'standardized_methods={standardized_count}/{len(exps)}'
    )

    if LOAD:
        tables, exps_leg = load_preliminary_results(scenario_name)
        log_status(f'LOADED scenario={scenario_name} methods={len(exps_leg)}')
        return None, tables, exps_leg

    results = run_parallel_exps(
        data_p,
        exps,
        n_dags,
        scenario_name=scenario_name,
        thr=thr,
        verb=verb,
    )
    shd, tpr, fdr, fscore, sid_norm, err, rel_err, err_th, acyc, runtime = zip(*results)
    metrics = {
        'shd': shd,
        'tpr': tpr,
        'fdr': fdr,
        'fscore': fscore,
        'sid_norm': sid_norm,
        'err': err,
        'rel_err': rel_err,
        'err_th': err_th,
        'acyc': acyc,
        'time': runtime,
    }

    exps_leg = [exp['leg'] for exp in exps]
    file_prefix = preliminary_results_prefix(scenario_name) if SAVE_RESULTS else None
    utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
    utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)
    log_status(f'SAVED scenario={scenario_name} prefix={file_prefix}')
    return metrics, None, exps_leg


# %% [markdown]
# ## CASE 1 - N=100, Homocedastic, var=1, Weights - [0.5, 1]

# %%

N = 100
SCENARIO_NAME = 'er4_N100_var1'

n_dags = 100
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),
    'var': 1
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# %% [markdown]
# ## CASE 2 - N=100, Homocedastic, var=15, Weights - [0.5, 1]

# %%

N = 100
SCENARIO_NAME = 'er4_N100_var15'

n_dags = 100
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),
    'var': 15
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# %% [markdown]
# ## CASE 3 - N=100, Heterocedastic, mean var=1, Weights - [0.5, 1]

# %%

N = 100
SCENARIO_NAME = 'er4_N100_hetero_var1'

# Per-node variances drawn once (seeded) and shared by all DAGs, as in the original script.
var = np.random.uniform(low=0.5, high=1.5, size=N)

n_dags = 100
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),
    'var': var
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)
