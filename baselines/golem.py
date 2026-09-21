"""GOLEM baselines adapted from the official repository.

Original code:
https://github.com/ignavierng/golem
"""

import os

import numpy as np
import scipy.linalg as sla
from tqdm.auto import tqdm


class GOLEM:
    """GOLEM baseline with a NumPy/SciPy Adam optimizer.

    Based on Ng et al. (2020), "On the Role of Sparsity and DAG Constraints
    for Learning Linear DAGs" and the official GOLEM TensorFlow implementation.
    """

    def __init__(self, equal_variances=True, dtype=np.float32, seed=0, verbose=False):
        self.equal_variances = equal_variances
        self.dtype = dtype
        self.seed = seed
        self.verbose = verbose
        self.vprint = print if verbose else lambda *a, **k: None

    def dagness(self, W):
        return np.trace(sla.expm(W * W)) - W.shape[0]

    def _h(self, W):
        E = sla.expm(W * W)
        h = np.trace(E) - self.d
        G_h = 2 * W * E.T
        return h, G_h

    def _likelihood(self, W):
        I_minus_W = self.Id - W
        R = self.X @ I_minus_W
        xt_r = self.X.T @ R

        sign, log_abs_det = np.linalg.slogdet(I_minus_W)
        if sign == 0:
            log_abs_det = -np.inf

        if self.equal_variances:
            residual_sq = max(np.sum(R * R), self.eps)
            likelihood = 0.5 * self.d * np.log(residual_sq) - log_abs_det
            G_likelihood = -self.d * xt_r / residual_sq
        else:
            residual_sq = np.maximum(np.sum(R * R, axis=0), self.eps)
            likelihood = 0.5 * np.sum(np.log(residual_sq)) - log_abs_det
            G_likelihood = -xt_r @ np.diag(1.0 / residual_sq)

        try:
            G_logdet = np.linalg.inv(I_minus_W).T
        except np.linalg.LinAlgError:
            G_logdet = np.linalg.pinv(I_minus_W).T

        return likelihood, G_likelihood + G_logdet

    def _score(self, W):
        likelihood, G_likelihood = self._likelihood(W)
        h, G_h = self._h(W)
        l1 = np.abs(W).sum()
        score = likelihood + self.lambda1 * l1 + self.lambda2 * h
        grad = G_likelihood + self.lambda1 * np.sign(W) + self.lambda2 * G_h
        np.fill_diagonal(grad, 0)
        return score, likelihood, h, grad

    def _adam_update(self, grad, iteration, beta_1, beta_2, adam_epsilon):
        self.opt_m = beta_1 * self.opt_m + (1 - beta_1) * grad
        self.opt_v = beta_2 * self.opt_v + (1 - beta_2) * (grad ** 2)
        step_scale = np.sqrt(1 - beta_2 ** iteration) / (1 - beta_1 ** iteration)
        return step_scale * self.opt_m / (np.sqrt(self.opt_v) + adam_epsilon)

    def threshold_till_dag(self, W):
        """Remove smallest absolute edges until the weighted graph is a DAG."""
        W = np.copy(W)
        if self.is_dag(W):
            return W, 0

        nonzero_indices = np.where(W != 0)
        weight_indices = list(zip(W[nonzero_indices], nonzero_indices[0], nonzero_indices[1]))
        dag_threshold = 0
        for weight, row, col in sorted(weight_indices, key=lambda item: abs(item[0])):
            if self.is_dag(W):
                break
            W[row, col] = 0
            dag_threshold = abs(weight)

        return W, dag_threshold

    @staticmethod
    def is_dag(W):
        import networkx as nx
        return nx.is_directed_acyclic_graph(nx.DiGraph(W))

    def postprocess(self, W, graph_thres=0.3):
        """Official GOLEM postprocessing: threshold, then prune cycles."""
        W_processed = np.copy(W)
        W_processed[np.abs(W_processed) <= graph_thres] = 0
        W_processed, self.dag_threshold_ = self.threshold_till_dag(W_processed)
        return W_processed

    def fit(self, X, lambda1=None, lambda2=5.0, lambda_1=None, lambda_2=None,
            num_iter=100000, learning_rate=1e-3, w_threshold=0.3,
            graph_thres=None, postprocess=True, B_init=None, checkpoint=1000,
            tol=None, beta_1=0.9, beta_2=0.999, adam_epsilon=1e-8,
            tf_return_compat=True, disable_tqdm=True):
        """Fit GOLEM and return the estimated weighted adjacency matrix.

        Args:
            X: Data matrix with shape [n_samples, n_nodes].
            lambda1/lambda_1: L1 sparsity penalty. Defaults follow GOLEM-EV/NV.
            lambda2/lambda_2: DAG penalty coefficient.
            num_iter: Number of Adam iterations.
            learning_rate: Adam learning rate.
            w_threshold/graph_thres: Postprocessing threshold.
            postprocess: If True, threshold and prune cycles before returning.
            B_init: Optional weighted matrix initialization.
            checkpoint: Iterations between convergence checks/logging.
            tol: Optional relative objective tolerance for early stopping.
            tf_return_compat: If True, match the official TensorFlow repo's
                returned matrix, which is one optimizer step behind because
                B is fetched together with train_op.
        """
        if lambda_1 is not None:
            lambda1 = lambda_1
        if lambda_2 is not None:
            lambda2 = lambda_2
        if lambda1 is None:
            lambda1 = 2e-2 if self.equal_variances else 2e-3
        if graph_thres is not None:
            w_threshold = graph_thres

        self.X = np.asarray(X, dtype=self.dtype)
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.n, self.d = self.X.shape
        self.Id = np.eye(self.d, dtype=self.dtype)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = np.finfo(self.dtype).eps

        if B_init is None:
            W = np.zeros((self.d, self.d), dtype=self.dtype)
        else:
            W = np.asarray(B_init, dtype=self.dtype).copy()
            if W.shape != (self.d, self.d):
                raise ValueError("B_init must have shape [n_nodes, n_nodes].")
            np.fill_diagonal(W, 0)

        self.opt_m = np.zeros_like(W)
        self.opt_v = np.zeros_like(W)
        self.history_ = {'score': [], 'likelihood': [], 'h': []}
        score_prev = np.inf
        W_return = W.copy()

        with tqdm(total=int(num_iter), disable=disable_tqdm) as pbar:
            for iteration in range(1, int(num_iter) + 1):
                score, likelihood, h, grad = self._score(W)
                W_return = W.copy()
                adam_step = self._adam_update(grad, iteration, beta_1, beta_2, adam_epsilon)
                W -= learning_rate * adam_step
                np.fill_diagonal(W, 0)

                should_check = (
                    iteration == int(num_iter)
                    or (checkpoint is not None and iteration % checkpoint == 0)
                )
                if should_check:
                    score, likelihood, h, _ = self._score(W)
                    self.history_['score'].append(score)
                    self.history_['likelihood'].append(likelihood)
                    self.history_['h'].append(h)
                    self.vprint(
                        f'Iter {iteration}: score {score:.4e} | '
                        f'likelihood {likelihood:.4e} | h {h:.4e}'
                    )
                    if tol is not None and np.isfinite(score_prev):
                        rel_change = abs((score_prev - score) / max(abs(score_prev), self.eps))
                        if rel_change <= tol:
                            pbar.update(int(num_iter) - iteration + 1)
                            break
                    score_prev = score

                pbar.update(1)

        self.W_optimizer_final = W
        self.W_raw = W_return if tf_return_compat else W
        self.h_final = self.dagness(self.W_raw)
        self.score_final, self.likelihood_final, _, _ = self._score(self.W_raw)
        self.W_est = self.postprocess(self.W_raw, w_threshold) if postprocess else self.W_raw
        return self.W_est


class GOLEM_EV(GOLEM):
    def __init__(self, dtype=np.float32, seed=0, verbose=False):
        super().__init__(equal_variances=True, dtype=dtype, seed=seed, verbose=verbose)


class GOLEM_NV(GOLEM):
    def __init__(self, dtype=np.float32, seed=0, verbose=False, init_with_ev=True):
        super().__init__(equal_variances=False, dtype=dtype, seed=seed, verbose=verbose)
        self.init_with_ev = init_with_ev

    def fit(self, X, lambda1=None, lambda2=5.0, lambda_1=None, lambda_2=None,
            lambda1_ev=2e-2, lambda2_ev=None, num_iter=100000, num_iter_ev=None,
            learning_rate=1e-3, learning_rate_ev=None, w_threshold=0.3,
            graph_thres=None, postprocess=True, B_init=None, checkpoint=1000,
            init_with_ev=None,
            tol=None, beta_1=0.9, beta_2=0.999, adam_epsilon=1e-8,
            tf_return_compat=True, disable_tqdm=True):
        if lambda_1 is not None:
            lambda1 = lambda_1
        if lambda_2 is not None:
            lambda2 = lambda_2
        if lambda1 is None:
            lambda1 = 2e-3
        if lambda2_ev is None:
            lambda2_ev = lambda2
        if num_iter_ev is None:
            num_iter_ev = num_iter
        if learning_rate_ev is None:
            learning_rate_ev = learning_rate
        if init_with_ev is None:
            init_with_ev = self.init_with_ev

        if B_init is None and init_with_ev:
            self.ev_model_ = GOLEM(
                equal_variances=True,
                dtype=self.dtype,
                seed=self.seed,
                verbose=self.verbose,
            )
            self.ev_model_.fit(
                X,
                lambda1=lambda1_ev,
                lambda2=lambda2_ev,
                num_iter=num_iter_ev,
                learning_rate=learning_rate_ev,
                w_threshold=w_threshold,
                graph_thres=graph_thres,
                postprocess=False,
                B_init=None,
                checkpoint=checkpoint,
                tol=tol,
                beta_1=beta_1,
                beta_2=beta_2,
                adam_epsilon=adam_epsilon,
                tf_return_compat=tf_return_compat,
                disable_tqdm=disable_tqdm,
            )
            B_init = self.ev_model_.W_raw

        return super().fit(
            X,
            lambda1=lambda1,
            lambda2=lambda2,
            num_iter=num_iter,
            learning_rate=learning_rate,
            w_threshold=w_threshold,
            graph_thres=graph_thres,
            postprocess=postprocess,
            B_init=B_init,
            checkpoint=checkpoint,
            tol=tol,
            beta_1=beta_1,
            beta_2=beta_2,
            adam_epsilon=adam_epsilon,
            tf_return_compat=tf_return_compat,
            disable_tqdm=disable_tqdm,
        )


class GOLEM_TF(GOLEM):
    """GOLEM baseline using the TensorFlow graph from the official repository."""

    def __init__(self, equal_variances=True, dtype=np.float32, seed=0, verbose=False):
        super().__init__(equal_variances=equal_variances, dtype=dtype, seed=seed, verbose=verbose)

    @staticmethod
    def _load_tensorflow():
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
        return tf

    def _build_tf_graph(self, tf, n, d, lambda1, lambda2, B_init):
        tf.compat.v1.reset_default_graph()

        lr = tf.compat.v1.placeholder(tf.float32)
        X_ph = tf.compat.v1.placeholder(tf.float32, shape=[n, d])
        if B_init is None:
            B_var = tf.Variable(tf.zeros([d, d], tf.float32))
        else:
            B_var = tf.Variable(tf.convert_to_tensor(B_init, tf.float32))
        B = tf.linalg.set_diag(B_var, tf.zeros(d, dtype=tf.float32))

        residual = X_ph - X_ph @ B
        if self.equal_variances:
            likelihood = (
                0.5 * d * tf.math.log(tf.square(tf.linalg.norm(residual)))
                - tf.linalg.slogdet(tf.eye(d) - B)[1]
            )
        else:
            likelihood = (
                0.5 * tf.math.reduce_sum(
                    tf.math.log(tf.math.reduce_sum(tf.square(residual), axis=0))
                )
                - tf.linalg.slogdet(tf.eye(d) - B)[1]
            )

        l1_penalty = tf.norm(B, ord=1)
        h = tf.linalg.trace(tf.linalg.expm(B * B)) - d
        score = likelihood + lambda1 * l1_penalty + lambda2 * h
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(score)
        return lr, X_ph, B, score, likelihood, h, train_op

    def fit(self, X, lambda1=None, lambda2=5.0, lambda_1=None, lambda_2=None,
            num_iter=100000, learning_rate=1e-3, w_threshold=0.3,
            graph_thres=None, postprocess=True, B_init=None, checkpoint=1000,
            tol=None, disable_tqdm=True):
        """Fit GOLEM with TensorFlow and return a NumPy adjacency matrix.

        This intentionally matches the official repository's TF1-style graph
        and fetch order. The optional tol argument is accepted for interface
        compatibility but ignored, because the official implementation always
        runs the requested number of iterations.
        """
        _ = tol
        if lambda_1 is not None:
            lambda1 = lambda_1
        if lambda_2 is not None:
            lambda2 = lambda_2
        if lambda1 is None:
            lambda1 = 2e-2 if self.equal_variances else 2e-3
        if graph_thres is not None:
            w_threshold = graph_thres

        tf = self._load_tensorflow()
        self.X = np.asarray(X, dtype=np.float32)
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.n, self.d = self.X.shape
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        B_init_tf = None
        if B_init is not None:
            B_init_tf = np.asarray(B_init, dtype=np.float32).copy()
            if B_init_tf.shape != (self.d, self.d):
                raise ValueError("B_init must have shape [n_nodes, n_nodes].")
            np.fill_diagonal(B_init_tf, 0)

        lr, X_ph, B, score, likelihood, h, train_op = self._build_tf_graph(
            tf, self.n, self.d, lambda1, lambda2, B_init_tf
        )
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            device_count={'GPU': 0},
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        ))

        self.history_ = {'score': [], 'likelihood': [], 'h': []}
        try:
            sess.run(tf.compat.v1.global_variables_initializer())
            with tqdm(total=int(num_iter), disable=disable_tqdm) as pbar:
                for iteration in range(0, int(num_iter) + 1):
                    if iteration == 0:
                        score_val, likelihood_val, h_val, B_est = sess.run(
                            [score, likelihood, h, B],
                            feed_dict={X_ph: self.X, lr: learning_rate},
                        )
                    else:
                        _, score_val, likelihood_val, h_val, B_est = sess.run(
                            [train_op, score, likelihood, h, B],
                            feed_dict={X_ph: self.X, lr: learning_rate},
                        )
                        pbar.update(1)

                    if checkpoint is not None and iteration % checkpoint == 0:
                        self.history_['score'].append(score_val)
                        self.history_['likelihood'].append(likelihood_val)
                        self.history_['h'].append(h_val)
                        self.vprint(
                            f'Iter {iteration}: score {score_val:.4e} | '
                            f'likelihood {likelihood_val:.4e} | h {h_val:.4e}'
                        )

            self.W_raw = np.asarray(B_est, dtype=self.dtype)
            self.W_optimizer_final = np.asarray(
                sess.run(B, feed_dict={X_ph: self.X, lr: learning_rate}),
                dtype=self.dtype,
            )
            self.score_final = float(score_val)
            self.likelihood_final = float(likelihood_val)
            self.h_final = float(h_val)
        finally:
            sess.close()

        self.W_est = self.postprocess(self.W_raw, w_threshold) if postprocess else self.W_raw
        return np.asarray(self.W_est, dtype=self.dtype)


class GOLEM_TF_EV(GOLEM_TF):
    def __init__(self, dtype=np.float32, seed=0, verbose=False):
        super().__init__(equal_variances=True, dtype=dtype, seed=seed, verbose=verbose)


class GOLEM_TF_NV(GOLEM_TF):
    def __init__(self, dtype=np.float32, seed=0, verbose=False, init_with_ev=True):
        super().__init__(equal_variances=False, dtype=dtype, seed=seed, verbose=verbose)
        self.init_with_ev = init_with_ev

    def fit(self, X, lambda1=None, lambda2=5.0, lambda_1=None, lambda_2=None,
            lambda1_ev=2e-2, lambda2_ev=None, num_iter=100000, num_iter_ev=None,
            learning_rate=1e-3, learning_rate_ev=None, w_threshold=0.3,
            graph_thres=None, postprocess=True, B_init=None, checkpoint=1000,
            init_with_ev=None, tol=None, disable_tqdm=True):
        if lambda_1 is not None:
            lambda1 = lambda_1
        if lambda_2 is not None:
            lambda2 = lambda_2
        if lambda1 is None:
            lambda1 = 2e-3
        if lambda2_ev is None:
            lambda2_ev = lambda2
        if num_iter_ev is None:
            num_iter_ev = num_iter
        if learning_rate_ev is None:
            learning_rate_ev = learning_rate
        if init_with_ev is None:
            init_with_ev = self.init_with_ev

        if B_init is None and init_with_ev:
            self.ev_model_ = GOLEM_TF(
                equal_variances=True,
                dtype=self.dtype,
                seed=self.seed,
                verbose=self.verbose,
            )
            self.ev_model_.fit(
                X,
                lambda1=lambda1_ev,
                lambda2=lambda2_ev,
                num_iter=num_iter_ev,
                learning_rate=learning_rate_ev,
                w_threshold=w_threshold,
                graph_thres=graph_thres,
                postprocess=False,
                B_init=None,
                checkpoint=checkpoint,
                tol=tol,
                disable_tqdm=disable_tqdm,
            )
            B_init = self.ev_model_.W_raw

        return super().fit(
            X,
            lambda1=lambda1,
            lambda2=lambda2,
            num_iter=num_iter,
            learning_rate=learning_rate,
            w_threshold=w_threshold,
            graph_thres=graph_thres,
            postprocess=postprocess,
            B_init=B_init,
            checkpoint=checkpoint,
            tol=tol,
            disable_tqdm=disable_tqdm,
        )
