#!/usr/bin/env python3
"""Validate actual models against independently expanded Poisson GLMs.

The reference uses a rank-reduced dummy matrix and ordinary Newton steps, not
the production absorption routine. It checks coefficients, fitted means, and
clustered covariance; additional cases verify exact zero handling.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppml_hdfe import fit_ppml_hdfe, poisson_deviance
from trade_data import load_panel, build_balanced_portfolio, build_full_internal, portfolio_arrays, geography_matrix


def dense_reference(y, x, fixed_effects, cluster):
    dummy_blocks = []
    for labels in fixed_effects:
        levels, ix = np.unique(labels, return_inverse=True)
        dummy_blocks.append(np.eye(len(levels))[ix])
    d = np.column_stack(dummy_blocks)
    _, rr, piv = qr(d, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(rr)) > 1e-10))
    z = np.column_stack([x, d[:, piv[:rank]]])
    coef = np.zeros(z.shape[1])
    for _ in range(100):
        eta = z @ coef
        mu = np.exp(eta)
        info = z.T @ (mu[:, None] * z)
        step = np.linalg.solve(info, z.T @ (y - mu))
        if np.max(np.abs(step)) < 2e-11:
            break
        ll = np.sum(y * eta - mu)
        scale = 1.0
        for _ in range(40):
            candidate = coef + scale * step
            candidate_eta = z @ candidate
            candidate_ll = np.sum(y * candidate_eta - np.exp(candidate_eta))
            if candidate_ll >= ll - 1e-12 * (1 + abs(ll)):
                coef = candidate
                break
            scale *= 0.5
        else:
            raise AssertionError("independent reference line search failed")
    else:
        raise AssertionError("independent reference did not converge")
    eta = z @ coef
    mu = np.exp(eta)
    bread = np.linalg.inv(z.T @ (mu[:, None] * z))
    levels, ci = np.unique(cluster, return_inverse=True)
    scores = np.column_stack([np.bincount(ci, weights=z[:, j] * (y - mu), minlength=len(levels))
                              for j in range(z.shape[1])])
    vcov = bread @ (scores.T @ scores) @ bread * len(levels) / (len(levels) - 1)
    return coef[:x.shape[1]], mu, vcov[:x.shape[1], :x.shape[1]]


def main():
    panel = load_panel(ROOT / "data" / "raw" / "panel_data.csv")
    for declaration in ("export", "import"):
        portfolio, _ = build_balanced_portfolio(panel, declaration)
        train = [r for r in portfolio if r.year in (1927, 1928, 1929) and not r.external]
        a = portfolio_arrays(train)
        x = geography_matrix(a["origin"], a["destination"])
        fe = [a["origin_year"], a["destination_year"]]
        fitted = fit_ppml_hdfe(a["y"], x=x, fixed_effects=fe, cluster=a["pair"], tol=1e-9)
        beta, mu, vcov = dense_reference(a["y"], x, fe, a["pair"])
        assert np.max(np.abs(fitted.beta - beta)) < 2e-8
        assert np.max(np.abs(fitted.mu - mu)) < 2e-9
        assert np.max(np.abs(fitted.vcov - vcov)) < 2e-8
        assert fitted.max_profile_step <= 1e-9
        assert fitted.max_fe_relative_score <= 1e-11
        scaled = fit_ppml_hdfe(10 * a["y"], x=x, fixed_effects=fe, cluster=a["pair"], tol=1e-9)
        assert np.max(np.abs(scaled.beta - fitted.beta)) < 2e-8
        assert np.max(np.abs(scaled.vcov - fitted.vcov)) < 2e-8
        print(f"PASS: actual {declaration} geography coefficients, fitted means, covariance, and uniform scaling")

    # The actual event specification adds a directed-pair FE family. Validate
    # this richer nuisance structure rather than relying on geography alone.
    sample = [r for r in build_full_internal(panel, "export") if 1928 <= r.year <= 1937]
    a = portfolio_arrays(sample)
    balkans = {"Bulgaria", "Greece", "Hungary", "Romania", "Yugoslavia"}
    treated = np.array([r.origin == "Germany" and r.destination in balkans for r in sample])
    event_years = [year for year in range(1928, 1938) if year != 1933]
    x = np.column_stack([treated * (a["year"] == year) for year in event_years]).astype(float)
    fe = [a["pair"], a["origin_year"], a["destination_year"]]
    fitted = fit_ppml_hdfe(a["y"], x=x, fixed_effects=fe, cluster=a["pair"], tol=1e-9)
    keep = fitted.keep
    beta, mu, vcov = dense_reference(a["y"][keep], x[keep], [g[keep] for g in fe], a["pair"][keep])
    assert np.max(np.abs(fitted.beta - beta)) < 2e-8
    assert np.max(np.abs(fitted.mu[keep] - mu)) < 2e-9
    assert np.max(np.abs(fitted.vcov - vcov)) < 2e-8
    print("PASS: actual three-FE Germany event coefficients, fitted means, and clustered covariance")

    # Only exactly zero groups are removed, never a small positive total.
    y = np.array([0.0, 0.0, 1e-20, 2e-20, 0.4, 0.6])
    group = np.array(["zero", "zero", "tiny", "tiny", "ordinary", "ordinary"])
    fitted = fit_ppml_hdfe(y, fixed_effects=[group])
    assert fitted.keep.tolist() == [False, False, True, True, True, True]
    assert np.allclose(fitted.mu[2:4], 1.5e-20, rtol=1e-9, atol=0)
    assert poisson_deviance(np.array([0.0]), np.array([0.0])) == 0.0
    assert np.isinf(poisson_deviance(np.array([1.0]), np.array([0.0])))
    print("PASS: exact zero limits and preservation of small positive groups")


if __name__ == "__main__":
    main()
