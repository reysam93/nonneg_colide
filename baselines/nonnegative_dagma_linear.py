"""Nonnegative DAGMA baseline adapted from the official DAGMA implementation.

Original code:
https://github.com/kevinsbello/dagma

Variant of DAGMA_linear that keeps the classical DAGMA acyclicity penalty
based on ``sI - W o W`` and projects each Adam step onto nonnegative matrices.
"""

import numpy as np
import scipy.linalg as sla
from scipy.special import expit as sigmoid

from baselines.dagma_linear import DAGMA_linear


class NonnegativeDAGMA_linear(DAGMA_linear):
    """DAGMA with projected Adam steps onto W >= 0 and zero diagonal."""

    @staticmethod
    def _project(W):
        W_proj = np.maximum(W, 0)
        np.fill_diagonal(W_proj, 0)
        return W_proj

    def _domain_matrix(self, W, s):
        try:
            return sla.inv(s * self.Id - W * W) + 1e-16
        except (ValueError, sla.LinAlgError):
            return None

    def _in_domain(self, W, s):
        M = self._domain_matrix(W, s)
        return M is not None and not np.any(M < 0)

    def minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        W = self._project(W)
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')

        for iter in range(1, max_iter + 1):
            M = self._domain_matrix(W, s)
            if M is None or np.any(M < 0):
                self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                return W, False

            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W)
            elif self.loss_type == 'logistic':
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            grad = self._adam_update(Gobj, iter, beta_1, beta_2)

            lr_step = lr
            W_next = self._project(W - lr_step * grad)
            while not self._in_domain(W_next, s):
                lr_step *= 0.5
                if lr_step <= 1e-16:
                    return W, True
                W_next = self._project(W - lr_step * grad)
                self.vprint(f'Learning rate decreased to lr: {lr_step}')

            W = W_next
            lr = lr_step

            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W_est): {h:.4e}')
                self.vprint(f'\tscore(W_est): {score:.4e}')
                self.vprint(f'\tobj(W_est): {obj_new:.4e}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new

        return W, True
