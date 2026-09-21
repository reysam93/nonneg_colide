"""NoFears (NOTEARS-KKTS) baseline adapted from the official repository.

Original code:
https://github.com/skypea/DAG_No_Fear

The KKTS local search below is adapted from the official implementation's
``local_search_given_matrix.py`` so this baseline is self-contained. The wrapper
exposes the same ``fit(X, ...)`` interface as the other project baselines.
"""

import inspect
import warnings

import networkx as nx
import numpy as np
from scipy import linalg
import scipy.optimize as sopt
from sklearn.linear_model import LassoLars


_LASSO_LARS_ACCEPTS_NORMALIZE = (
    "normalize" in inspect.signature(LassoLars).parameters
)


def _make_lasso_lars(**kwargs):
    """Create LassoLars with the original no-normalization setting when valid."""
    if _LASSO_LARS_ACCEPTS_NORMALIZE:
        kwargs.setdefault("normalize", False)
    return LassoLars(**kwargs)


def _warn(message):
    warnings.warn(str(message), RuntimeWarning, stacklevel=2)


# Adapted from DAG_No_Fear/local_search_given_matrix.py (Apache-2.0).
def local_search_given_W(X, W, Wtol, penTol, tau, hTol, revEdges, noPen, initNoPen=False, minimizeZ=True):
    # Prepare inputs for local search
    n, d = X.shape
    # Adjacency matrix
    A = np.abs(W)
    # Active set
    active = A > Wtol
    W[~active] = 0
    A[~active] = 0
    # Constraint and penalty matrix
    h, pen = eval_grad_h(A)
    # Set of zero-value constraints
    Z = ~active & ((pen > penTol) | initNoPen)

    Wstar = np.zeros_like(W)
    reg = _make_lasso_lars(alpha=tau, fit_intercept=False)
    # _warn(W.shape())
    for j in range(d):
        if (~Z[:, j]).any():
            reg.fit(X[:, ~Z[:, j]], X[:, j])
            Wstar[~Z[:, j], j] = reg.coef_
    # Adjacency matrix
    A = np.abs(Wstar)
    # Active set
    activeStar = A > Wtol
    Wstar[~activeStar] = 0
    A[~activeStar] = 0
    # Compute entire loss gradient matrix i.e. correlations with residuals
    Gram = np.dot(X.T, X) / n
    corrStar = Gram - np.dot(Gram, Wstar)
    # Constraint and penalty matrix
    h, pen = eval_grad_h(A)
    tau = np.full(d, tau)

    if h > hTol:
        _warn('WARNING: Post-processed augmented Lagrangian algorithm did not converge to h = 0, h = {}'.format(h))
        # Cholesky factorizations of all Gram submatrices
        chol = {}
        for j in range(d):
            chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])
        tauMat = np.tile(tau, [d, 1])
    # Add zero-value constraints if still infeasible
    while h > hTol:
        # Determine edge (i,j) to remove
        i, j = lars_path_matrix_single(Wstar, Z, activeStar, corrStar, chol, pen, tauMat,
                                       Gram, penTol=penTol)
        # Add (i,j) to Z, re-optimize and update
        Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
            remove_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, augmentZ=True)

    # Local search
    Wstar, Z, activeStar, corrStar, A, h, pen, itSearch = \
        restore_reverse(Wstar, Z, activeStar, corrStar, A, h, pen, tau, Gram, X=X,
                        revEdges=revEdges, noPen=noPen, hTol=hTol, penTol=penTol, minimizeZ=minimizeZ)

    return Wstar, h, itSearch


def lars_path_matrix_single(Wstar, Z, activeStar, corrStar, chol, pen, tauMat, Gram, Wtol=1e-10, penTol=0,
                            checkLARS=False, X=None):
    """Follow weighted Lasso path using LARS algorithm until one matrix element set to zero
    """

    d = Wstar.shape[1]
    # Compute LARS directions column by column
    dW = np.zeros_like(Wstar)
    for j in range(d):
        if activeStar[:, j].any():
            dW[activeStar[:, j], j] = linalg.cho_solve(chol[j],
                                                       np.sign(
                                                           Wstar[activeStar[:, j], j] + corrStar[activeStar[:, j], j]) *
                                                       pen[activeStar[:, j], j])
    # Increments to correlations
    a = np.dot(Gram, dW)

    # Bounds on step size
    gamma = np.full_like(Wstar, np.inf)
    ind = activeStar & (np.abs(dW) > 0)  # avoid divide-by-zero warning but infinities are actually correct
    gamma[ind] = Wstar[ind] / dW[ind]
    gamma[gamma < 0] = np.inf
    ind = ~activeStar & ~Z & (np.abs(a) > pen)
    gamma[ind] = (tauMat[ind] - np.sign(a[ind]) * corrStar[ind]) \
                 / (np.abs(a[ind]) - pen[ind])
    if np.isinf(gamma).all():
        _warn('WARNING: gamma is all np.inf!')
        _warn('activeStar =')
        _warn(activeStar)
        _warn('Wstar =')
        _warn(Wstar)
        _warn('dW =')
        _warn(dW)
        _warn('pen =')
        _warn(pen)
        _warn('corrStar =')
        _warn(corrStar)
        _warn('a =')
        _warn(a)
    # Limiting index
    (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)

    # The following should happen only rarely
    if ~(activeStar[i, j] & (pen[i, j] > penTol)):
        # Instantiate penalized regression quantities
        W = Wstar.copy()
        W[np.abs(W) < Wtol] = 0
        active = activeStar.copy()
        corr = corrStar.copy()
        alpha = 0

        # The following should happen only rarely
        while ~(activeStar[i, j] & (pen[i, j] > penTol)) & (pen[active] > penTol).any():
            # Update penalized regression quantities
            gammaMin = gamma[i, j]
            W[active] -= gammaMin * dW[active]
            W[np.abs(W) < Wtol] = 0
            active[i, j] = ~active[i, j]
            corr += gammaMin * a
            alpha += gammaMin
            gamma -= gammaMin

            if checkLARS:
                # Check LARS solution
                reg = _make_lasso_lars()
                Wcheck = np.zeros_like(W)
                for jj in range(d):
                    if (~Z[:, jj]).any():
                        Wcheck[~Z[:, jj], jj] = pen_regress(reg, X[:, ~Z[:, jj]], X[:, jj], 1,
                                                            tauMat[~Z[:, jj], jj] + alpha * pen[~Z[:, jj], jj])
                assert np.allclose(Wcheck, W)

            # Update LARS direction and correlation increments for column j
            dW[:, j] = 0
            dW[active[:, j], j] = linalg.solve(Gram[np.ix_(active[:, j], active[:, j])],
                                               np.sign(W[active[:, j], j] + corr[active[:, j], j]) * pen[
                                                   active[:, j], j],
                                               assume_a='pos')
            a[:, j] = np.dot(Gram[:, active[:, j]], dW[active[:, j], j])

            # Update bounds on step size from column j
            gamma[:, j] = np.inf
            ind = active[:, j] & (np.abs(dW[:, j]) > 0)
            gamma[ind, j] = W[ind, j] / dW[ind, j]
            gamma[gamma < 0] = np.inf
            if active[i, j]:
                # Special treatment for previously inactive (i,j)
                gamma[i, j] = np.inf
            ind = ~active[:, j] & ~Z[:, j] & (np.abs(a[:, j]) > pen[:, j])
            gamma[ind, j] = (tauMat[ind, j] + alpha * pen[ind, j] - np.sign(a[ind, j]) * corr[ind, j]) \
                            / (np.abs(a[ind, j]) - pen[ind, j])
            if ~active[i, j]:
                # Special treatment for previously active (i,j)
                if np.abs(a[i, j]) > pen[i, j]:
                    if np.sign(a[i, j] * corr[i, j]) == 1:
                        gamma[i, j] = 0
                    else:
                        gamma[i, j] = 2 * (tauMat[i, j] + alpha * pen[i, j]) / (np.abs(a[i, j]) - pen[i, j])
                else:
                    gamma[i, j] = np.inf
            if np.isinf(gamma).all():
                _warn('WARNING: gamma is all np.inf!')
                _warn('active =')
                _warn(active)
                _warn('W =')
                _warn(W)
                _warn('dW =')
                _warn(dW)
                _warn('pen =')
                _warn(pen)
                _warn('corr =')
                _warn(corr)
                _warn('a =')
                _warn(a)

            # Limiting index
            (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)

    return i, j


def eval_h(A):
    # Compute tr(exp(A)) - d
    # _warn(A)
    # h = np.expm1(np.linalg.eigvals(A)).real.sum() # TODO  worse than below

    # polynomial h
    d = A.shape[0]
    M = np.eye(d) + np.abs(A) / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d

    # new h
    # h= np.expm1(np.linalg.eigvals(A)).real.sum()

    return h


def eval_h_deri(A):
    # Compute tr(exp(A)) - d
    # _warn(A)
    # h = np.expm1(np.linalg.eigvals(A)).real.sum() # TODO  worse than below

    # polynomial h
    d = A.shape[0]
    M = np.eye(d) + np.abs(A) / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    E = E.T

    # new derivative
    # E = linalg.expm(A).T

    return E

def eval_grad_h(A):
    """Evaluate h(A) and gradient"""
    d = A.shape[0]
    M = np.eye(d) + A / d
    grad = np.linalg.matrix_power(M, d - 1).T
    h = (grad * M).sum() - d

    return h, grad


def remove_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, augmentZ=True, eps=1e-10, checkLARS=False,
                X=None):
    """Remove edge (i,j) (add to Z) and update all quantities
    """

    # Convert row index i to index of non-Z elements
    j0 = ((~Z[:, j]).nonzero()[0] == i).nonzero()[0][0]
    # Re-optimize column j of Wstar given new constraint
    Wstar[~Z[:, j], j], activeStar[~Z[:, j], j], corrStar[~Z[:, j], j] = \
        lars_set_zero(Wstar[~Z[:, j], j], activeStar[~Z[:, j], j], j0, corrStar[~Z[:, j], j],
                      chol[j], tau[j], Gram[np.ix_(~Z[:, j], ~Z[:, j])], eps)
    # Add to Z
    Z[i, j] = True

    if checkLARS:
        # Check LARS solution
        if (~Z[:, j]).any():
            reg = _make_lasso_lars(alpha=tau[j])
            reg.fit(X[:, ~Z[:, j]], X[:, j])
            assert np.allclose(Wstar[~Z[:, j], j], reg.coef_)
        assert np.abs(Wstar[i, j]) < eps

    # Update Z components of gradient
    corrStar[Z[:, j], j] = Gram[Z[:, j], j] - np.dot(Gram[np.ix_(Z[:, j], ~Z[:, j])], Wstar[~Z[:, j], j])
    # Update Cholesky factorization for column j
    chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])
    # Adjacency matrix
    A[:, j] = np.abs(Wstar[:, j])
    # Evaluate constraint
    h = eval_h(A)
    # Penalty matrix
    pen = eval_h_deri(A)  # linalg.expm(A).T
    # Add "necessary" zeros to Z
    if augmentZ:
        Z |= (~activeStar & (pen > eps))

    return Wstar, Z, activeStar, corrStar, chol, A, h, pen


def restore_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, eps=1e-10, checkLARS=False, X=None):
    """Restore edge (i,j) (remove from Z) and update all quantities
    """

    # Remove from Z
    Z[i, j] = False
    # Convert row index i to index of non-Z elements
    j0 = ((~Z[:, j]).nonzero()[0] == i).nonzero()[0][0]
    # Re-optimize column j
    Wstar[~Z[:, j], j], activeStar[~Z[:, j], j], corrStar[~Z[:, j], j] = \
        lars_relax_zero(Wstar[~Z[:, j], j], activeStar[~Z[:, j], j], j0,
                        corrStar[~Z[:, j], j], tau[j], Gram[np.ix_(~Z[:, j], ~Z[:, j])], eps)

    if checkLARS:
        # Check LARS solution
        reg = _make_lasso_lars(alpha=tau[j])
        reg.fit(X[:, ~Z[:, j]], X[:, j])
        assert np.allclose(Wstar[~Z[:, j], j], reg.coef_)

    # Update Z components of gradient
    corrStar[Z[:, j], j] = Gram[Z[:, j], j] - np.dot(Gram[np.ix_(Z[:, j], ~Z[:, j])], Wstar[~Z[:, j], j])
    # Update Cholesky factorization for column j
    chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])
    # Adjacency matrix
    A[:, j] = np.abs(Wstar[:, j])
    # Evaluate constraint
    h = eval_h(A)
    # Penalty matrix
    pen = eval_h_deri(A)  # linalg.expm(A).T

    return Wstar, Z, activeStar, corrStar, chol, A, pen


def reverse_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, h, pen, tau, Gram, revEdges='both', eps=1e-10,
                 checkLARS=False, X=None):
    """Try reversing edge (i,j) in current solution
    """

    reg = _make_lasso_lars()
    # Current loss
    F = linalg.norm(X - np.dot(X, Wstar)) ** 2

    # Initialize reversed solution
    Wrev = Wstar.copy()
    Zrev = Z.copy()
    activeRev = activeStar.copy()
    corrRev = corrStar.copy()
    Arev = A.copy()

    # Remove edge (i,j)
    # Convert row index i to index of non-Z elements
    j0 = ((~Zrev[:, j]).nonzero()[0] == i).nonzero()[0][0]
    # Re-optimize column j given new constraint
    Wrev[~Zrev[:, j], j], activeRev[~Zrev[:, j], j], corrRev[~Zrev[:, j], j] = \
        lars_set_zero(Wrev[~Zrev[:, j], j], activeRev[~Zrev[:, j], j], j0, corrRev[~Zrev[:, j], j],
                      chol[j], tau[j], Gram[np.ix_(~Zrev[:, j], ~Zrev[:, j])], eps)
    Zrev[i, j] = True

    if checkLARS:
        # Check LARS solution
        if (~Zrev[:, j]).any():
            reg = _make_lasso_lars(alpha=tau[j])
            reg.fit(X[:, ~Zrev[:, j]], X[:, j])
            assert np.allclose(Wrev[~Zrev[:, j], j], reg.coef_)
        assert np.abs(Wrev[i, j]) < eps

    # Allow edge (j,i)
    Zrev[j, i] = False
    # Convert row index j to index of non-Z elements
    j0 = ((~Zrev[:, i]).nonzero()[0] == j).nonzero()[0][0]
    # Re-optimize column i
    Wrev[~Zrev[:, i], i], activeRev[~Zrev[:, i], i], corrRev[~Zrev[:, i], i] = \
        lars_relax_zero(Wrev[~Zrev[:, i], i], activeRev[~Zrev[:, i], i], j0,
                        corrRev[~Zrev[:, i], i], tau[i], Gram[np.ix_(~Zrev[:, i], ~Zrev[:, i])], eps)

    if checkLARS:
        # Check LARS solution
        reg.alpha = tau[i]
        reg.fit(X[:, ~Zrev[:, i]], X[:, i])
        assert np.allclose(Wrev[~Zrev[:, i], i], reg.coef_)

    # Evaluate loss and constraint for reversed solution
    Frev = linalg.norm(X - np.dot(X, Wrev)) ** 2
    Arev[:, [i, j]] = np.abs(Wrev[:, [i, j]])
    hRev = eval_h(Arev)

    # Reverse if both loss and constraint are no worse
    #    _warn(Frev - F, hRev - h)
    # TODO: either is true, and more strict than this
    if (Frev - F < eps) & ((revEdges == 'loss') | (hRev - h < eps)):  # TODO HERE check
        Wstar = Wrev
        Z = Zrev
        activeStar = activeRev
        # Update gradient for columns i, j
        corrStar[:, [i, j]] = Gram[:, [i, j]] - np.dot(Gram, Wstar[:, [i, j]])
        # Update Cholesky factorization for columns i, j
        chol[i] = linalg.cho_factor(Gram[np.ix_(activeStar[:, i], activeStar[:, i])])
        chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])
        A = Arev
        h = hRev
        pen = eval_h_deri(A)  # linalg.expm(A).T linalg.expm(A).T

    return Wstar, Z, activeStar, corrStar, chol, A, h, pen


def restore_reverse(Wstar, Z, activeStar, corrStar, A, h, pen, tau, Gram, minimizeZ=True, revEdges='alt-full',
                    restoreFirst=True, noPen=False, Wtol=1e-10, hTol=1e-10, penTol=0, checkLARS=False, X=None):
    """Perform local search around feasible (acyclic) solution by restoring and reversing edges
    """

    # Initialize local search
    d = Wstar.shape[0]
    it = 0
    unnec = np.zeros_like(Wstar, dtype=bool)
    rev = np.zeros_like(Wstar, dtype=bool)

    if revEdges != 'old':
        # Edge reversal memory
        reved = np.zeros_like(Wstar, dtype=bool)

        # Unnecessary zero-value constraints
        if minimizeZ & (h < hTol):
            unnec = Z & (pen <= penTol)
        # Edge reversal candidates based on marginal change to constraint and loss
        if revEdges:
            rev = activeStar & ~activeStar.T & ((pen.T - A - pen < max(Wtol, penTol)) | noPen) & (
                        np.abs(corrStar.T) > tau + Wtol) & ~reved

        if (unnec | rev).any():
            # Cholesky factorizations of all Gram submatrices
            chol = {}
            for j in range(d):
                chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])

        # Iterate while there are candidates
        restore = restoreFirst
        while (unnec | rev).any():
            # Choose candidate with largest loss gradient
            if revEdges.startswith('alt'):
                if restore & unnec.any():
                    corrCand = unnec * np.abs(corrStar)
                elif rev.any():
                    corrCand = rev * np.abs(corrStar.T)
                else:
                    # Switch to restoring
                    restore = True
                    corrCand = unnec * np.abs(corrStar)
            elif revEdges == 'lower':
                if unnec.any():
                    corrCand = unnec * np.abs(corrStar)
                else:
                    corrCand = rev * np.abs(corrStar.T)
            else:
                corrCand = unnec * np.abs(corrStar) + rev * np.abs(corrStar.T)
            (i, j) = np.unravel_index(corrCand.argmax(), corrCand.shape)

            if unnec[i, j]:
                # Restore edge (i,j)
                if np.abs(corrStar[i, j]) < tau[j] + Wtol:
                    # Loss gradient too small to change Wstar but still remove from Z
                    Z[i, j] = False
                else:
                    Wstar, Z, activeStar, corrStar, chol, A, pen = \
                        restore_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, Wtol,
                                     checkLARS=checkLARS, X=X)
                # Column j updated, clear reversal memory
                reved[j, :] = False
                reved[:, j] = False
                # Update unnecessary zero-value constraints
                if minimizeZ:  # & (h < hTol):
                    unnec = Z & (pen <= penTol)
                if revEdges.startswith('alt'):
                    # Switch to reversing
                    restore = False
            elif rev[i, j]:
                # Try reversing edge
                Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
                    reverse_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, h, pen,
                                 tau, Gram, revEdges, Wtol, checkLARS=checkLARS, X=X)
                if Z[i, j]:
                    # Reversal succeeded, columns i, j updated, clear reversal memory
                    reved[i, :] = False
                    reved[:, i] = False
                    reved[j, :] = False
                    reved[:, j] = False
                    # No need to try re-reversal
                    reved[j, i] = True
                    # Update unnecessary zero-value constraints
                    if minimizeZ:  # & (h < hTol):
                        unnec = Z & (pen <= penTol)
                    if revEdges == 'alt-early':
                        # Switch to restoring
                        restore = True
                # Mark as attempted
                reved[i, j] = True

            # Edge reversal candidates based on marginal change to constraint and loss
            if revEdges:
                rev = activeStar & ~activeStar.T & ((pen.T - A - pen < max(Wtol, penTol)) | noPen) & (
                            np.abs(corrStar.T) > tau + Wtol) & ~reved

            it += 1

    else:  # revEdges == 'old'
        if minimizeZ & (h < hTol):
            # Unnecessary zero-value constraints
            unnec = Z & (pen <= penTol)

            if unnec.any():
                # Cholesky factorizations of all Gram submatrices
                chol = {}
                for j in range(d):
                    chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])

            while unnec.any():
                # Choose unnecessary constraint with largest gradient
                (i, j) = np.unravel_index(np.abs(unnec * corrStar).argmax(), unnec.shape)
                # Restore edge (i,j)
                Wstar, Z, activeStar, corrStar, chol, A, pen = \
                    restore_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, Wtol, checkLARS=checkLARS,
                                 X=X)

                # Candidate edges for reversal based on marginal change to constraint and loss
                rev = activeStar & ~activeStar.T & (pen.T - A - pen < max(Wtol, penTol)) & (
                            np.abs(corrStar.T) > tau + Wtol)
                if rev.any():
                    iRev, jRev = rev.nonzero()
                    # Sort by marginal change if more than one candidate
                    if len(iRev) > 1:
                        idx = np.argsort((pen.T - A - pen - np.abs(corrStar.T) + tau)[rev])
                        iRev, jRev = iRev[idx], jRev[idx]
                    for idx in range(len(iRev)):
                        # Try reversing edge
                        Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
                            reverse_edge(Wstar, Z, iRev[idx], jRev[idx], activeStar,
                                         corrStar, chol, A, h, pen, tau, Gram, 'both', Wtol, checkLARS=checkLARS, X=X)

                # Unnecessary zero-value constraints
                unnec = Z & (pen <= penTol)

                it += 1

    return Wstar, Z, activeStar, corrStar, A, h, pen, it


def lars_set_zero(w, active, j0, corr, chol, tau, Gram, eps=1e-10):
    """Compute Lasso solution subject to new constraint w[j0] = 0
    """

    # LARS direction
    # Need to re-optimize only if j0 active
    if active[j0]:
        # Right-hand side for LARS direction
        b = np.zeros_like(w)
        b[j0] = np.sign(w[j0] + corr[j0])
        useChol = True

    # Iterate until j0 no longer active
    j = None
    while active[j0]:
        # LARS direction
        dw = np.zeros_like(w)
        if useChol:
            dw[active] = linalg.cho_solve(chol, b[active])
        else:
            dw[active] = linalg.solve(Gram[np.ix_(active, active)], b[active], assume_a='pos')
        # Increments to correlations
        a = np.dot(Gram[:, active], dw[active])

        # Bounds on step size
        if tau < eps:
            # Assume special case of tau = 0
            j = j0
            gamma = w[j0] / dw[j0]
        else:
            gamma = np.full_like(w, np.inf)
            ind = active & (np.abs(dw) > 0)
            gamma[ind] = w[ind] / dw[ind]
            gamma[gamma < 0] = np.inf
            if j is not None and active[j]:
                # Special treatment for previously inactive j
                gamma[j] = np.inf
            ind = ~active & (np.abs(a) > 0)
            gamma[ind] = (tau - np.sign(a[ind]) * corr[ind]) / np.abs(a[ind])
            if j is not None and ~active[j]:
                # Special treatment for previously active j
                if np.sign(a[j] * corr[j]) == 1:
                    gamma[j] = 0
                else:
                    gamma[j] = 2 * tau / np.abs(a[j])
            if np.isnan(gamma).any():
                _warn('error')
            # Step size and limiting index
            j = gamma.argmin()
            gamma = gamma[j]

        # Update coefficients, active set, gradient
        w[active] -= gamma * dw[active]
        active[j] = ~active[j]
        useChol = False
        corr += gamma * a

        if np.isnan(corr).any():
            _warn('error')

    return w, active, corr


def pen_regress(reg, X, y, p, pen):
    # ell_p-penalized regression of y on X
    # Factors for scaling weights and columns of X
    if p == 1:
        scale = pen
    elif p == 2:
        scale = np.sqrt(pen)

    # Scale columns of X to account for unequal penalties
    if not scale.all() == 0:
        Xscaled = X / scale  # TODO solving this diviing zero, experimental mores, check rho
    else:
        scale += 1e-6
        Xscaled = X/scale
        # _warn(np.isinf(Xscaled).any())
    # Fit model
    reg.fit(Xscaled, y)
    # Re-scale coefficients before returning
    return reg.coef_ / scale

def lars_relax_zero(w, active, j0, corr, tau, Gram, eps=1e-10):
    """Compute Lasso solution after relaxing constraint w[j0] = 0
    """

    # Initial alpha
    alpha = np.abs(corr[j0]) - tau
    # Need to re-optimize only if alpha > 0
    if alpha > eps:
        # Right-hand side for LARS direction
        b = np.zeros_like(w)
        b[j0] = np.sign(corr[j0])
        # Add j0 to active set
        j = j0

    while alpha > eps:
        # Update active set
        active[j] = ~active[j]

        # LARS direction
        dw = np.zeros_like(w)
        dw[active] = linalg.solve(Gram[np.ix_(active, active)], b[active], assume_a='pos')
        # Increments to correlations
        a = np.dot(Gram[:, active], dw[active])

        # Bounds on step size (including alpha itself)
        if tau < eps:
            # Assume special case of tau = 0
            gamma = alpha
        else:
            gamma = np.full_like(w, alpha)
            ind = active & (np.abs(dw) > 0)
            gamma[ind] = -w[ind] / dw[ind]
            gamma[gamma < 0] = alpha
            if active[j]:
                # Special treatment for previously inactive j
                gamma[j] = alpha
            ind = ~active & (np.abs(a) > 0)
            gamma[ind] = (tau + np.sign(a[ind]) * corr[ind]) / np.abs(a[ind])
            if ~active[j]:
                # Special treatment for previously active j
                if np.sign(a[j] * corr[j]) == -1:
                    gamma[j] = 0
                else:
                    gamma[j] = 2 * tau / np.abs(a[j])
            # Step size and limiting index
            j = gamma.argmin()
            gamma = gamma[j]

        # Update coefficients, gradient, active set
        w[active] += gamma * dw[active]
        corr -= gamma * a
        alpha -= gamma

    return w, active, corr

class NoFearsLinear:
    """Linear NoFears baseline using NOTEARS followed by official KKTS.

    Defaults reproduce the settings reported by Wei, Gao, and Yu (2020):
    least-squares loss, L1 penalty 0.1, polynomial acyclicity function,
    acyclicity tolerance 1e-10, and edge threshold 0.3. An exact graph check
    removes the weakest edge from any residual numerical cycle.
    """

    def __init__(self, dtype=np.float64, verbose=False):
        self.dtype = dtype
        self.verbose = verbose
        self.vprint = print if verbose else lambda *args, **kwargs: None

    @staticmethod
    def is_dag(W):
        return nx.is_directed_acyclic_graph(nx.DiGraph(W))

    def dagness(self, W):
        return float(eval_h(np.abs(np.asarray(W))))

    @staticmethod
    def _validate_data(X, dtype):
        X = np.asarray(X, dtype=dtype)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional [n_samples, n_nodes] array.")
        if X.shape[0] < 2:
            raise ValueError("X must contain at least two samples.")
        if X.shape[1] < 2:
            raise ValueError("X must contain at least two variables.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X.copy()

    @staticmethod
    def _threshold(W, threshold):
        W = np.asarray(W).copy()
        W[np.abs(W) < threshold] = 0
        np.fill_diagonal(W, 0)
        return W

    @staticmethod
    def _break_cycles(W):
        """Remove the weakest edge in each detected cycle until W is a DAG."""
        W = np.asarray(W).copy()
        removed_edges = []
        graph = nx.DiGraph(W)

        while not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph)
            source, target = min(
                cycle,
                key=lambda edge: (abs(W[edge[0], edge[1]]), edge[0], edge[1]),
            )
            weight = W[source, target]
            W[source, target] = 0
            graph.remove_edge(source, target)
            removed_edges.append((int(source), int(target), float(weight)))

        return W, removed_edges

    @staticmethod
    def _notears_h(W):
        """Polynomial NOTEARS acyclicity value and gradient."""
        d = W.shape[0]
        M = np.eye(d, dtype=W.dtype) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = 2 * E.T * W
        return h, G_h

    def _fit_notears(
        self,
        X,
        lambda1,
        max_iter,
        h_tol,
        rho_init,
        rho_factor,
        rho_max,
        h_progress_rate,
    ):
        """Run the polynomial NOTEARS initialization used by NoFears."""
        n, d = X.shape

        def _adj(w):
            return (w[: d * d] - w[d * d :]).reshape(d, d)

        def _loss(W):
            residual = X - X @ W
            loss = 0.5 / n * np.sum(residual * residual)
            gradient = -X.T @ residual / n
            return loss, gradient

        def _func(w):
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = self._notears_h(W)
            objective = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            gradient = np.concatenate(
                (G_smooth + lambda1, -G_smooth + lambda1),
                axis=None,
            )
            return objective, gradient

        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(d)
            for j in range(d)
        ]
        w_est = np.zeros(2 * d * d, dtype=self.dtype)
        rho = float(rho_init)
        alpha = 0.0
        h = np.inf
        self.notears_history_ = []
        self.optimizer_results_ = []

        for iteration in range(int(max_iter)):
            w_new = None
            h_new = None
            while rho < rho_max:
                result = sopt.minimize(
                    _func,
                    w_est,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                )
                self.optimizer_results_.append(
                    {
                        "success": bool(result.success),
                        "status": int(result.status),
                        "message": str(result.message),
                        "nit": int(result.nit),
                        "nfev": int(result.nfev),
                        "fun": float(result.fun),
                    }
                )
                if not np.isfinite(result.x).all():
                    raise FloatingPointError(
                        "NoFears NOTEARS initialization produced non-finite coefficients."
                    )

                w_new = result.x
                h_new, _ = self._notears_h(_adj(w_new))
                if h_new > h_progress_rate * h:
                    rho *= rho_factor
                else:
                    break

            if w_new is None or h_new is None:
                raise RuntimeError(
                    "NoFears NOTEARS initialization reached rho_max before an update."
                )

            w_est, h = w_new, float(h_new)
            alpha += rho * h
            self.notears_history_.append(
                {"iteration": iteration + 1, "h": h, "rho": rho, "alpha": alpha}
            )
            self.vprint(
                f"NOTEARS iteration {iteration + 1}: "
                f"h={h:.4e}, rho={rho:.4e}, alpha={alpha:.4e}"
            )
            if h <= h_tol or rho >= rho_max:
                break

        return _adj(w_est), h, rho, alpha

    def fit(
        self,
        X,
        lambda1=0.1,
        w_threshold=0.3,
        max_iter=100,
        h_tol=1e-10,
        rho_init=1.0,
        rho_factor=10.0,
        rho_max=1e16,
        h_progress_rate=0.25,
        w_tol=1e-10,
        pen_tol=0.0,
        rev_edges="alt-full",
        minimize_z=True,
        init_no_pen=True,
        no_pen=False,
    ):
        """Fit NOTEARS-KKTS and store the weighted DAG in ``W_est``."""
        if lambda1 < 0:
            raise ValueError("lambda1 must be non-negative.")
        if w_threshold < 0:
            raise ValueError("w_threshold must be non-negative.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least one.")
        if h_tol <= 0 or w_tol <= 0:
            raise ValueError("h_tol and w_tol must be positive.")
        if rho_init <= 0 or rho_factor <= 1 or rho_max <= rho_init:
            raise ValueError("rho parameters must satisfy 0 < rho_init < rho_max.")

        X_work = self._validate_data(X, self.dtype)
        X_work -= X_work.mean(axis=0, keepdims=True)
        self.X_mean_ = np.asarray(X, dtype=self.dtype).mean(axis=0)

        W_raw, self.notears_h_, self.rho_final_, self.alpha_final_ = (
            self._fit_notears(
                X_work,
                lambda1=lambda1,
                max_iter=max_iter,
                h_tol=h_tol,
                rho_init=rho_init,
                rho_factor=rho_factor,
                rho_max=rho_max,
                h_progress_rate=h_progress_rate,
            )
        )
        self.W_notears_raw = W_raw.copy()
        self.W_notears = self._threshold(W_raw, w_threshold)

        W_search, search_h, search_iterations = local_search_given_W(
            X_work,
            self.W_notears.copy(),
            Wtol=w_tol,
            penTol=pen_tol,
            tau=lambda1,
            hTol=h_tol,
            revEdges=rev_edges,
            noPen=no_pen,
            initNoPen=init_no_pen,
            minimizeZ=minimize_z,
        )

        if not np.isfinite(W_search).all():
            raise FloatingPointError("NoFears KKTS produced non-finite coefficients.")

        self.W_search_raw = W_search.copy()
        self.search_h_ = float(search_h)
        self.search_iterations_ = int(search_iterations)
        self.W_thresholded_ = self._threshold(W_search, w_threshold).astype(
            self.dtype,
            copy=False,
        )
        self.h_thresholded_ = self.dagness(self.W_thresholded_)

        self.W_est, self.cycle_repair_edges_ = self._break_cycles(
            self.W_thresholded_
        )
        self.cycle_repair_count_ = len(self.cycle_repair_edges_)
        self.cycle_repair_applied_ = self.cycle_repair_count_ > 0
        if self.cycle_repair_applied_:
            max_removed = max(abs(edge[2]) for edge in self.cycle_repair_edges_)
            self.vprint(
                "NoFears exact-DAG safeguard removed "
                f"{self.cycle_repair_count_} edge(s); "
                f"largest removed magnitude={max_removed:.4e}."
            )

        self.h_final = self.dagness(self.W_est)

        if not self.is_dag(self.W_est):
            raise RuntimeError(
                "NoFears exact-DAG safeguard failed to remove all cycles."
            )

        return self.W_est
