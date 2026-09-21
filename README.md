# Nonnegative CoLiDE

Convex learning of non-negative weighted DAGs with concomitant estimation of the noise scale.

This repository contains the implementation of **nonnegative CoLiDE** and the code to
reproduce the experiments of the accompanying paper (in preparation). Nonnegative CoLiDE
brings together two ideas:

- **NOMAD** ([Rey, Saboksayr and Mateos, 2024](https://arxiv.org/abs/2409.07880)): when the
  DAG has non-negative edge weights, acyclicity can be enforced with a *convex*
  log-determinant function, so linear DAG structure learning becomes a convex program that
  is solved with the method of multipliers.
- **CoLiDE** (Saboksayr, Mateos and Tepper, ICLR 2024): the standard deviations of the
  exogenous noise are estimated jointly with the edge weights (concomitant estimation), which
  decouples the sparsity regularization from the unknown noise level.

The resulting estimator, `MetMulColide` in [`src/model.py`](src/model.py), fits a linear
structural equation model with non-negative weights, an l1 sparsity penalty and the
log-det acyclicity constraint, while the noise scales are treated as optimization variables.
Two variants are available:

| Variant | Constructor flag | Noise model | Output of `fit` |
|---|---|---|---|
| **NV** (default) | `equal_var=False` | one standard deviation per node | `(W_est, sigma)` with `sigma` of shape `(N,)` |
| **EV** | `equal_var=True` | one standard deviation shared by all nodes | `(W_est, sigma)` with scalar `sigma` |

The inner solver is selected with `primal_opt` (`'pgd'`, `'adam'`, `'fista'` or `'sca'`);
the experiments use successive convex approximation with Adam steps (`primal_opt='sca'`,
`sca_adam=True`). The NOMAD estimator (`MetMulDagma`) is included in the same module and is
used as a reference method in the experiments.

## Repository layout

```text
src/
  model.py            Nonneg_dagma, MetMulDagma (NOMAD) and MetMulColide (nonnegative CoLiDE)
  utils.py            DAG and SEM data generation, metrics, result tables and plots
baselines/            Adapted baseline implementations (each file cites its original repository)
experiments/
  scripts/            Experiment code; run in background and save tables/figures under results/
    preliminary_exp.py    Fixed N and M, three noise scenarios (var = 1, var = 15, heteroscedastic)
    number_samples.py     Recovery metrics vs. number of samples (var = 1 and heteroscedastic)
    different_vars.py     Recovery metrics vs. noise variance (homoscedastic and heteroscedastic)
  notebooks/          Load the saved results and plot them; they can also launch the runs
    run_algorithms.ipynb  Single-run diagnostics of the solvers on one synthetic DAG
    preliminary_exp.ipynb, number_samples.ipynb, different_vars.ipynb
results/              Created by the scripts (tables .csv, raw metrics .npz, figures .png); not versioned
logs/                 Standard output of the background runs; not versioned
```

## Installation

The code was run with Python 3.10. Create a virtual environment and install the pinned
dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

TensorFlow is listed because the GOLEM baseline needs it; it is only required to rerun
`number_samples.py` and `different_vars.py`. The DAGuerreotype wrapper
(`baselines/daguerreotype.py`) is not used by the experiments and needs the original
DAGuerreotype code (and PyTorch) under `code_aux/`, which is ignored by Git.

## Quick start

```python
import numpy as np
import src.utils as utils
from src.model import MetMulColide

# Synthetic data: ER DAG with 4N edges, weights in [0.5, 1], heteroscedastic Gaussian noise
N, M = 50, 1000
noise_var = np.random.uniform(.5, 1.5, N)
W_true, dag, X = utils.simulate_sem(N, M, 'er', edges=4 * N, w_range=(.5, 1), var=noise_var)

# Nonnegative CoLiDE, NV variant (equal_var=True for EV)
model = MetMulColide(primal_opt='sca', acyclicity='logdet', equal_var=False)
W_est, sigma_est = model.fit(X, lamb=.1 * np.sqrt(np.log(N) / M), stepsize=3e-4, s=1,
                             iters_in=30000, iters_out=10, alpha_0=.01, rho_0=.05, beta=2,
                             sca_adam=True)

W_hat = utils.to_dag(W_est, thr=.3)   # threshold small weights and prune any remaining cycle
```

`X` has one sample per row. The hyperparameters above are the ones used in the experiments;
`lamb` is scaled by `sqrt(log N / M)` as in the scripts.

## Reproducing the experiments

Run the scripts from the repository root (they resolve the root from their own location, so
any working directory works). Each run writes its tables and figures under `results/`:

```bash
mkdir -p results logs
nohup .venv/bin/python experiments/scripts/preliminary_exp.py > logs/preliminary.txt 2>&1 &
nohup .venv/bin/python experiments/scripts/number_samples.py  > logs/samples.txt 2>&1 &
nohup .venv/bin/python experiments/scripts/different_vars.py  > logs/vars.txt 2>&1 &
```

`number_samples.py` and `different_vars.py` accept the following environment variables:

| Variable | Effect |
|---|---|
| `SMOKE_TEST=1` | Tiny run to check the pipeline; results go to `results/*_smoke/` |
| `N_DAGS=20` | Override the number of random DAGs |
| `SCENARIOS=...` | Run only the listed scenarios (`er4_N100_var1,er4_N100_hetero` or `hom,hetero`) |
| `N_CPUS=8` | Number of parallel workers (default: all cores) |

Results are saved to `results/preliminary/`, `results/samples/` and `results/var/`.

The notebooks in `experiments/notebooks/` locate the repository root automatically (the
first parent directory containing `src/`) and import the experiment modules from
`experiments/scripts/`. With `LOAD = True` (the default) they only load the saved results
and produce the figures; set `LOAD = False` to launch the experiment from the notebook.

## Baselines

| File | Method | Original code |
|---|---|---|
| `colide.py` | CoLiDE (EV and NV) | https://github.com/alexisbellot/CoLiDE |
| `dagma_linear.py` | DAGMA | https://github.com/kevinsbello/dagma |
| `nonnegative_dagma_linear.py` | DAGMA restricted to non-negative weights | https://github.com/kevinsbello/dagma |
| `golem.py` | GOLEM (EV and NV) | https://github.com/ignavierng/golem |
| `notears.py` | NOTEARS | https://github.com/xunzheng/notears |
| `daguerreotype.py` | DAGuerreotype (wrapper, optional) | https://github.com/vzantedeschi/DAGuerreotype |
| `nofears.py` | NoFears (NOTEARS-KKTS) | https://github.com/skypea/DAG_No_Fear |
| `sortnregress.py` | SortNRegress | https://github.com/CausalDisco/CausalDisco |

The experiments use CoLiDE, DAGMA, nonnegative DAGMA, GOLEM, NOTEARS and NOMAD. The adapted
files keep the license of their original repositories.

## License

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
