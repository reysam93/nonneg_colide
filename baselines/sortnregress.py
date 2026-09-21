"""SortNRegress baselines adapted from CausalDisco.

Original code:
https://github.com/CausalDisco/CausalDisco
"""

import numpy as np
import networkx as nx
from scipy import linalg
from sklearn.linear_model import LassoLarsIC, LinearRegression


def r2coeff(X):
    """Compute per-variable R^2 values from partial correlations."""
    try:
        return 1 - 1 / np.diag(linalg.inv(np.corrcoef(X)))
    except linalg.LinAlgError:
        d = X.shape[0]
        r2s = np.zeros(d)
        lr = LinearRegression()
        X = X.T
        for k in range(d):
            parents = np.arange(d) != k
            lr.fit(X[:, parents], X[:, k])
            r2s[k] = lr.score(X[:, parents], X[:, k])
        return r2s


def sort_regress(X, scores):
    """Regress each variable on its predecessors in the score-induced order."""
    lr = LinearRegression()
    lars = LassoLarsIC(criterion="bic")
    X = np.asarray(X)
    d = X.shape[1]
    W = np.zeros((d, d))
    ordering = np.argsort(scores)

    for k in range(1, d):
        covariates = ordering[:k]
        target = ordering[k]
        lr.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(lr.coef_)
        lars.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = lars.coef_ * weight
    return W


def random_sort_regress(X, seed=None):
    """SortNRegress with a random ordering."""
    if seed is None:
        seed = np.random.randint(0, np.iinfo("int").max)
    rng = np.random.default_rng(seed)
    return sort_regress(X, rng.permutation(np.asarray(X).shape[1]))


def var_sort_regress(X):
    """SortNRegress with marginal variances as ordering scores."""
    X = np.asarray(X)
    return sort_regress(X, np.var(X, axis=0))


def r2_sort_regress(X):
    """SortNRegress with R^2 scores as ordering criterion."""
    X = np.asarray(X)
    return sort_regress(X, r2coeff(X.T))


class SortNRegress:
    """Class wrapper compatible with the project baseline interface."""

    def __init__(self, score="variance", seed=None, dtype=np.float64):
        self.score = score
        self.seed = seed
        self.dtype = dtype

    def _scores(self, X, scores=None):
        if scores is not None:
            return np.asarray(scores)
        if self.score in {"variance", "var"}:
            return np.var(X, axis=0)
        if self.score in {"r2", "R2"}:
            return r2coeff(X.T)
        if self.score == "random":
            rng = np.random.default_rng(self.seed)
            return rng.permutation(X.shape[1])
        raise ValueError("score must be one of {'variance', 'var', 'r2', 'R2', 'random'}.")

    @staticmethod
    def is_dag(W):
        return nx.is_directed_acyclic_graph(nx.DiGraph(W))

    def dagness(self, W):
        """Return 0 for a DAG, matching the convention used by acyclicity metrics."""
        return 0.0 if self.is_dag(W) else 1.0

    def fit(self, X, scores=None, w_threshold=None, graph_thres=None):
        """Fit the baseline and store the weighted adjacency in ``W_est``."""
        X = np.asarray(X, dtype=self.dtype)
        threshold = graph_thres if graph_thres is not None else w_threshold
        self.scores_ = self._scores(X, scores=scores)
        self.W_raw = sort_regress(X, self.scores_)
        self.W_est = np.asarray(self.W_raw, dtype=self.dtype)
        if threshold is not None:
            self.W_est = self.W_est.copy()
            self.W_est[np.abs(self.W_est) <= threshold] = 0
        return self.W_est


class VarSortNRegress(SortNRegress):
    def __init__(self, seed=None, dtype=np.float64):
        super().__init__(score="variance", seed=seed, dtype=dtype)


class R2SortNRegress(SortNRegress):
    def __init__(self, seed=None, dtype=np.float64):
        super().__init__(score="r2", seed=seed, dtype=dtype)


class RandomSortNRegress(SortNRegress):
    def __init__(self, seed=None, dtype=np.float64):
        super().__init__(score="random", seed=seed, dtype=dtype)
