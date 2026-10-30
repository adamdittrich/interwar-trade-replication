"""Poisson pseudo-maximum likelihood with absorbed fixed effects.

Low-dimensional coefficients are fitted by profiled Newton steps. At every step,
iterative proportional fitting solves the fixed-effect scores, and the Schur
complement of the nuisance-parameter information supplies the profile Hessian.
Convergence requires small coefficient steps and fixed-effect score errors;
a small change in likelihood alone is never a convergence criterion.

The automatic separation check covers zero-total fixed-effect groups only. It is
not a general separation detector. Failure to converge raises an exception.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass
class PPMLHDFEResult:
    beta: np.ndarray
    se: np.ndarray | None
    vcov: np.ndarray | None
    fe_effects: list[np.ndarray]
    fe_levels: list[np.ndarray]
    mu: np.ndarray
    eta: np.ndarray
    keep: np.ndarray
    converged: bool
    iterations: int
    loglik: float
    deviance: float
    nobs: int
    nclusters: int | None
    max_profile_step: float = 0.0
    max_fe_relative_score: float = 0.0
    n_dropped_zero_group_observations: int = 0
    covariance_convention: str = "cluster sandwich with G/(G-1) correction"


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, including its exact zero-outcome limit.

    A positive observation predicted as exactly zero has infinite deviance. The
    function does not conceal that failure by silently adding a positive floor.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if y.shape != mu.shape:
        raise ValueError("outcome and prediction arrays must have identical shapes")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(mu)) or np.any(y < 0) or np.any(mu < 0):
        raise ValueError("outcomes and predictions must be finite and non-negative")
    pos = y > 0
    if np.any(pos & (mu == 0)):
        return math.inf
    term = mu.copy()
    term[pos] = y[pos] * np.log(y[pos] / mu[pos]) - y[pos] + mu[pos]
    return float(max(0.0, 2.0 * np.sum(term)))


def _relabel(values: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    levels, inv = np.unique(np.asarray(values, dtype=object), return_inverse=True)
    return inv.astype(np.int64), levels


def _weighted_residualize(
    x: np.ndarray,
    groups: Sequence[np.ndarray],
    weights: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 20_000,
) -> np.ndarray:
    """Absorb nuisance dummies under the fitted-mean inner product."""
    out = np.asarray(x, dtype=float).copy()
    if out.ndim == 1:
        out = out[:, None]
    if not groups:
        return out
    denominators = [np.bincount(g, weights=weights, minlength=int(g.max()) + 1) for g in groups]
    for _ in range(max_iter):
        max_change = 0.0
        for g, den in zip(groups, denominators):
            for j in range(out.shape[1]):
                num = np.bincount(g, weights=weights * out[:, j], minlength=len(den))
                means = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                delta = means[g]
                out[:, j] -= delta
                max_change = max(max_change, float(np.max(np.abs(delta))))
        if max_change <= tol:
            return out
    raise RuntimeError("weighted fixed-effect absorption did not converge")


def _exp_finite(eta: np.ndarray) -> np.ndarray:
    if np.any(eta > 700) or np.any(eta < -700):
        raise FloatingPointError("linear predictor outside numerically representable range; inspect separation and offsets")
    return np.exp(eta)


def fit_ppml_hdfe(
    y: Sequence[float],
    x: np.ndarray | Sequence[float] | None = None,
    fixed_effects: Sequence[Sequence[object]] | None = None,
    offset: Sequence[float] | None = None,
    cluster: Sequence[object] | None = None,
    max_iter: int = 5_000,
    tol: float = 1e-9,
    eta_clip: tuple[float, float] | None = None,
    drop_separated: bool = True,
) -> PPMLHDFEResult:
    """Fit a Poisson conditional-mean model with absorbed additive fixed effects.

    Fractional outcomes are valid for the quasi-likelihood. Normalizing outcomes
    differently by reporter-year changes PPML weights and generally changes
    slope estimates; this routine makes no outcome-scale invariance claim.

    ``drop_separated`` removes observations in fixed-effect groups whose outcome
    total is exactly zero. More general forms of separation are not diagnosed.
    ``eta_clip`` is retained only for API compatibility; finite linear predictors
    are not clipped because clipping changes the objective and its score.

    For clustered inference, scores are summed within the supplied clusters.
    The sandwich is multiplied by G/(G-1), with no unverified degrees-of-freedom
    adjustment for the absorbed effects. Reported intervals use normal critical
    values and do not constitute a small-number-of-treated-clusters correction.
    """
    if eta_clip is not None:
        raise ValueError("eta clipping is unsupported: use finite offsets and inspect separation")
    if not (tol > 0) or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")
    y0 = np.asarray(y, dtype=float)
    if y0.ndim != 1 or not len(y0) or np.any(~np.isfinite(y0)) or np.any(y0 < 0):
        raise ValueError("y must be a nonempty finite, non-negative vector")
    n0 = len(y0)
    if x is None:
        x0 = np.empty((n0, 0), dtype=float)
    else:
        x0 = np.asarray(x, dtype=float)
        if x0.ndim == 1:
            x0 = x0[:, None]
        if x0.ndim != 2 or len(x0) != n0 or np.any(~np.isfinite(x0)):
            raise ValueError("x must be finite with one row per outcome")
    k = x0.shape[1]
    off0 = np.zeros(n0) if offset is None else np.asarray(offset, dtype=float)
    if off0.shape != y0.shape or np.any(~np.isfinite(off0)):
        raise ValueError("offset must be finite with one entry per outcome")
    fe_raw = [] if fixed_effects is None else [np.asarray(g, dtype=object) for g in fixed_effects]
    if any(g.ndim != 1 or len(g) != n0 for g in fe_raw):
        raise ValueError("fixed-effect labels must have one entry per outcome")
    if cluster is not None and len(cluster) != n0:
        raise ValueError("cluster labels must have one entry per outcome")
    if not fe_raw and k == 0:
        raise ValueError("model needs regressors or fixed effects")

    keep = np.ones(n0, dtype=bool)
    if drop_separated and fe_raw:
        changed = True
        while changed:
            changed = False
            for raw in fe_raw:
                if not np.any(keep):
                    raise ValueError("all observations belong to zero-total fixed-effect groups")
                inv, levels = _relabel(raw[keep])
                totals = np.bincount(inv, weights=y0[keep], minlength=len(levels))
                bad = (totals == 0.0)[inv]
                if np.any(bad):
                    keep[np.flatnonzero(keep)[bad]] = False
                    changed = True
    if not np.any(keep) or not np.any(y0[keep] > 0):
        raise ValueError("no positive outcomes remain")

    yv, xm, off = y0[keep], x0[keep], off0[keep]
    cluster_kept = None if cluster is None else np.asarray(cluster, dtype=object)[keep]
    groups, levels = [], []
    for raw in fe_raw:
        g, lev = _relabel(raw[keep])
        groups.append(g)
        levels.append(lev)
    totals = [np.bincount(g, weights=yv, minlength=len(lev)) for g, lev in zip(groups, levels)]
    if any(np.any(t == 0.0) for t in totals):
        raise ValueError("zero-total fixed-effect groups require drop_separated=True")
    # Conditional FE scores must be considerably more accurate than the final
    # coefficient tolerance to avoid contaminating the profile gradient.
    fe_tol = min(tol * 0.01, 1e-12)
    projection_tol = min(tol * 0.01, 1e-12)
    beta = np.zeros(k)
    fe = [np.zeros(len(lev)) for lev in levels]
    if fe:
        fe[0][:] = math.log(float(np.mean(yv))) - float(np.mean(off))

    def conditional_fit(b: np.ndarray, initial: list[np.ndarray]):
        effects = [v.copy() for v in initial]
        eta = off + (xm @ b if k else 0.0)
        for effect, g in zip(effects, groups):
            eta = eta + effect[g]
        if not groups:
            mu = _exp_finite(eta)
            return effects, eta, mu, float(np.sum(yv * eta - mu)), 0.0, 0
        for inner in range(1, max_iter + 1):
            for d, (g, observed) in enumerate(zip(groups, totals)):
                predicted = np.bincount(g, weights=_exp_finite(eta), minlength=len(observed))
                delta = np.log(observed / predicted)
                effects[d] += delta
                eta += delta[g]
            # Recentering changes only a nonidentified common location.
            for d in range(1, len(effects)):
                shift = float(np.mean(effects[d]))
                effects[d] -= shift
                effects[0] += shift
            mu = _exp_finite(eta)
            errors = [np.max(np.abs(np.log(np.bincount(g, weights=mu, minlength=len(t)) / t)))
                      for g, t in zip(groups, totals)]
            error = float(max(errors))
            if error <= fe_tol:
                return effects, eta, mu, float(np.sum(yv * eta - mu)), error, inner
        raise RuntimeError(f"conditional fixed-effect fit did not converge: max relative score {error:.3g}")

    fe, eta, mu, ll, fe_error, inner_iters = conditional_fit(beta, fe)
    max_profile_step = 0.0
    iteration = inner_iters if not k else 0
    converged = not k
    if k:
        for iteration in range(1, max_iter + 1):
            xt = _weighted_residualize(xm, groups, mu, tol=projection_tol)
            information = xt.T @ (mu[:, None] * xt)
            gradient = xt.T @ (yv - mu)
            eigenvalues = np.linalg.eigvalsh(information)
            if eigenvalues[0] <= max(1e-14, eigenvalues[-1] * 1e-12):
                raise ValueError("regressors are unidentified after fixed-effect absorption")
            step = np.linalg.solve(information, gradient)
            max_profile_step = float(np.max(np.abs(step)))
            if max_profile_step <= min(tol, 1e-9):
                converged = True
                break
            scale = 1.0
            accepted = False
            for _ in range(40):
                proposal = beta + scale * step
                try:
                    candidate = conditional_fit(proposal, fe)
                except FloatingPointError:
                    scale *= 0.5
                    continue
                if candidate[3] >= ll - 1e-13 * (1.0 + abs(ll)):
                    beta = proposal
                    fe, eta, mu, ll, fe_error, _ = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                raise RuntimeError("profile Newton line search failed")
        if not converged:
            raise RuntimeError(f"PPML coefficients did not converge: max profile step {max_profile_step:.3g}")

    vcov = se = None
    nclusters = None
    if k:
        xt = _weighted_residualize(xm, groups, mu, tol=projection_tol)
        bread_inv = np.linalg.inv(xt.T @ (mu[:, None] * xt))
        residual = yv - mu
        if cluster_kept is None:
            meat = xt.T @ ((residual ** 2)[:, None] * xt)
            covariance_convention = "heteroskedasticity-robust sandwich without finite-sample adjustment"
        else:
            cluster_inv, cluster_levels = _relabel(cluster_kept)
            nclusters = len(cluster_levels)
            if nclusters < 2:
                raise ValueError("clustered covariance requires at least two clusters")
            scores = np.column_stack([np.bincount(cluster_inv, weights=xt[:, j] * residual,
                                                  minlength=nclusters) for j in range(k)])
            meat = (scores.T @ scores) * nclusters / (nclusters - 1.0)
            covariance_convention = "cluster sandwich with G/(G-1) correction"
        vcov = bread_inv @ meat @ bread_inv
        vcov = 0.5 * (vcov + vcov.T)
        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    else:
        covariance_convention = "not applicable: no low-dimensional coefficients"

    mu_full = np.full(n0, np.nan)
    eta_full = np.full(n0, np.nan)
    mu_full[keep], eta_full[keep] = mu, eta
    return PPMLHDFEResult(
        beta=beta, se=se, vcov=vcov, fe_effects=fe, fe_levels=levels,
        mu=mu_full, eta=eta_full, keep=keep, converged=converged,
        iterations=iteration, loglik=ll, deviance=poisson_deviance(yv, mu),
        nobs=len(yv), nclusters=nclusters, max_profile_step=max_profile_step,
        max_fe_relative_score=fe_error,
        n_dropped_zero_group_observations=int(np.sum(~keep)),
        covariance_convention=covariance_convention,
    )
