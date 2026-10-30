#!/usr/bin/env python3
"""Reproduce all tables, figures, and numerical results for Old Partners, New Hierarchies.

Run from any working directory: python /path/to/EREH_Replication/run_all.py
No network connection is required after installing requirements.txt.
"""
from __future__ import annotations

import os
# Avoid oversubscribing small linear algebra problems. An explicit user setting wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import csv
import json
import math
import sys
import platform
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import scipy
import matplotlib
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from ppml_hdfe import fit_ppml_hdfe
from trade_data import (
    CORE19, FULL21, EXTERNAL, PortfolioRecord, TradeRecord,
    audit_panel, build_balanced_portfolio, build_full_internal,
    geography_matrix, load_panel, portfolio_arrays,
)
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
TAB = ROOT / "output" / "tables"


def subset_records(records: Sequence[PortfolioRecord], years: Iterable[int] | None = None,
                   internal_only: bool = False, excluded_reporters: set[str] | None = None):
    years_set = None if years is None else set(years)
    excluded_reporters = excluded_reporters or set()
    return [
        r for r in records
        if (years_set is None or r.year in years_set)
        and (not internal_only or (r.origin != EXTERNAL and r.destination != EXTERNAL))
        and r.reporter not in excluded_reporters
    ]


def arrays(records: Sequence[PortfolioRecord]):
    return portfolio_arrays(records)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None):
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sticky_gravity(records: Sequence[PortfolioRecord], years=range(1928, 1938),
                   portfolio_adjusted=True):
    sample = subset_records(records, years)
    # Attach the one-year lag by directed pair.
    lookup = {(r.pair, r.year): r.share for r in records}
    available = [r for r in sample if (r.pair, r.year - 1) in lookup]
    sample = [r for r in available if lookup[(r.pair, r.year - 1)] > 0.0]
    excluded_zero_lags = len(available) - len(sample)
    a = arrays(sample)
    lag = np.asarray([lookup[(r.pair, r.year - 1)] for r in sample], dtype=float)
    log_lag = np.log(lag)

    flexible = fit_ppml_hdfe(
        a["y"], fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
        cluster=a["pair"], tol=1e-9,
    )
    sticky_fe = [a["origin_year"], a["destination_year"]] if portfolio_adjusted else [a["destination_year"]]
    sticky = fit_ppml_hdfe(
        a["y"], x=log_lag, fixed_effects=sticky_fe, cluster=a["pair"],
        tol=1e-9, max_iter=10_000,
    )
    # Zero-total fixed-effect groups have zero fitted means at the likelihood
    # boundary. Keep their observed current zeros in the common blending sample.
    y = a["y"]
    f = np.where(flexible.keep, flexible.mu, 0.0)
    s = np.where(sticky.keep, sticky.mu, 0.0)
    delta = s - f
    denom = float(np.dot(delta, delta))
    alpha = 0.0 if denom <= 0 else float(np.clip(np.dot(y - f, delta) / denom, 0.0, 1.0))
    blend = alpha * s + (1.0 - alpha) * f

    return {
        "gamma": float(sticky.beta[0]),
        "gamma_se": float(sticky.se[0]),
        "alpha": alpha,
        "flexible_mse": float(np.mean((y - f) ** 2)),
        "sticky_mse": float(np.mean((y - s) ** 2)),
        "blended_mse": float(np.mean((y - blend) ** 2)),
        "alpha_identified": bool(denom > 0),
        "zero_lag_treatment": "condition on strictly positive recorded lag; current zeros retained",
        "excluded_zero_lag_observations": excluded_zero_lags,
        "gamma_estimation_nobs": sticky.nobs,
        "nobs": len(y),
        "portfolio_adjusted": portfolio_adjusted,
    }, (sample, a, lag, f, s, blend)


def sticky_by_period(records: Sequence[PortfolioRecord]):
    specs = [
        ("Full 1928-1937", range(1928, 1938)),
        ("Pre-break 1928-1931", range(1928, 1932)),
        ("Post-break 1932-1937", range(1932, 1938)),
        ("New Plan era 1934-1937", range(1934, 1938)),
    ]
    rows = []
    details = {}
    for label, years in specs:
        result, detail = sticky_gravity(records, years=years, portfolio_adjusted=True)
        result["period"] = label
        rows.append(result)
        details[label] = detail
    return rows, details


def standardized_pre_affinity(records: Sequence[PortfolioRecord], training_years=(1927, 1928, 1929)):
    """Exact log pre-period means, standardized across supported pairs per reporter."""
    by_pair = defaultdict(list)
    pair_reporter = {}
    for r in subset_records(records, training_years):
        by_pair[r.pair].append(r.share)
        pair_reporter[r.pair] = r.reporter
    means = {p: float(np.mean(v)) for p, v in by_pair.items() if np.sum(v) > 0.0}
    log_means = {p: math.log(v) for p, v in means.items()}
    by_reporter = defaultdict(list)
    for p, value in log_means.items():
        by_reporter[pair_reporter[p]].append(value)
    centers = {name: float(np.mean(v)) for name, v in by_reporter.items()}
    scales = {name: float(np.std(v)) or 1.0 for name, v in by_reporter.items()}
    return {p: (value - centers[pair_reporter[p]]) / scales[pair_reporter[p]]
            for p, value in log_means.items()}


def attenuation_event_study(records: Sequence[PortfolioRecord], reference_year=1929):
    pre_affinity = standardized_pre_affinity(records)
    sample = [r for r in subset_records(records, range(1928, 1938)) if r.pair in pre_affinity]
    a = arrays(sample)
    z = np.asarray([pre_affinity[r.pair] for r in sample])
    event_years = [y for y in range(1928, 1938) if y != reference_year]
    x = np.column_stack([z * (a["year"] == year) for year in event_years])
    model = fit_ppml_hdfe(
        a["y"], x=x,
        fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
        cluster=a["pair"], tol=1e-9, max_iter=10_000,
    )
    rows = []
    for year, beta, se in zip(event_years, model.beta, model.se):
        rows.append({
            "year": year, "estimate": float(beta), "se": float(se),
            "ci_low": float(beta - 1.96 * se), "ci_high": float(beta + 1.96 * se),
            "relative_percent_for_one_sd": float(100.0 * (math.exp(beta) - 1.0)),
        })
    # Candidate single breaks; same number of low-dimensional parameters, so log-likelihood ranks them.
    breaks = []
    for break_year in range(1930, 1936):
        xb = z * (a["year"] >= break_year)
        m = fit_ppml_hdfe(
            a["y"], x=xb,
            fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
            cluster=a["pair"], tol=1e-9,
        )
        breaks.append({
            "break_year": break_year, "estimate": float(m.beta[0]),
            "se": float(m.se[0]), "loglik": m.loglik, "deviance": m.deviance,
        })
    best = max(breaks, key=lambda x: x["loglik"])
    return rows, breaks, best, model


def germany_event_study(records: Sequence[PortfolioRecord], reference_year=1933):
    sample = subset_records(records, range(1928, 1938), internal_only=True)
    a = arrays(sample)
    balkans = {"Bulgaria", "Greece", "Hungary", "Romania", "Yugoslavia"}
    directions = {
        "Germany exports to Balkans": np.asarray([
            r.origin == "Germany" and r.destination in balkans for r in sample
        ], dtype=float),
        "Balkan exports to Germany": np.asarray([
            r.origin in balkans and r.destination == "Germany" for r in sample
        ], dtype=float),
    }
    event_years = [y for y in range(1928, 1938) if y != reference_year]
    output = []
    models = {}
    for label, dyad in directions.items():
        x = np.column_stack([dyad * (a["year"] == year) for year in event_years])
        model = fit_ppml_hdfe(
            a["y"], x=x,
            fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
            cluster=a["pair"], tol=1e-9, max_iter=10_000,
        )
        models[label] = model
        for year, beta, se in zip(event_years, model.beta, model.se):
            output.append({
                "direction": label, "year": year, "estimate": float(beta), "se": float(se),
                "ci_low": float(beta - 1.96 * se), "ci_high": float(beta + 1.96 * se),
                "percent": float(100.0 * (math.exp(beta) - 1.0)),
            })
    return output, models


def germany_pretrend_and_trend_adjustment(
    records: Sequence[PortfolioRecord], event_models: dict[str, object], reference_year=1933
):
    """Diagnose pre-trends and estimate a post-1934 shift net of a differential linear trend.

    The event coefficients are descriptive rather than a staggered-treatment design.  This
    diagnostic therefore jointly tests the 1928--1932 coefficients relative to 1933 and then
    estimates a dyad-group-specific linear trend together with a post-1934 level shift.
    """
    sample = subset_records(records, range(1928, 1938), internal_only=True)
    a = arrays(sample)
    balkans = {"Bulgaria", "Greece", "Hungary", "Romania", "Yugoslavia"}
    directions = {
        "Germany exports to Balkans": np.asarray([
            r.origin == "Germany" and r.destination in balkans for r in sample
        ], dtype=float),
        "Balkan exports to Germany": np.asarray([
            r.origin in balkans and r.destination == "Germany" for r in sample
        ], dtype=float),
    }
    event_years = [y for y in range(1928, 1938) if y != reference_year]
    pre_idx = [event_years.index(y) for y in (1928, 1929, 1930, 1931, 1932)]
    rows = []
    for label, dyad in directions.items():
        event = event_models[label]
        b = event.beta[pre_idx]
        v = event.vcov[np.ix_(pre_idx, pre_idx)]
        wald = float(b @ np.linalg.pinv(v, rcond=1e-12) @ b)
        pretrend_p = float(chi2.sf(wald, len(pre_idx)))

        time = (a["year"] - reference_year).astype(float)
        post = (a["year"] >= 1934).astype(float)
        x = np.column_stack([dyad * time, dyad * post])
        trend_model = fit_ppml_hdfe(
            a["y"], x=x,
            fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
            cluster=a["pair"], tol=1e-9, max_iter=10_000,
        )
        rows.append({
            "direction": label,
            "pretrend_wald": wald,
            "pretrend_df": len(pre_idx),
            "pretrend_p": pretrend_p,
            "linear_trend": float(trend_model.beta[0]),
            "linear_trend_se": float(trend_model.se[0]),
            "post1934_level_shift": float(trend_model.beta[1]),
            "post1934_level_shift_se": float(trend_model.se[1]),
            "post1934_implied_percent": float(100.0 * (math.exp(trend_model.beta[1]) - 1.0)),
        })
    return rows


def hub_placebo_model(records: Sequence[PortfolioRecord]):
    sample = subset_records(records, range(1928, 1938), internal_only=True)
    a = arrays(sample)
    balkans = {"Bulgaria", "Greece", "Hungary", "Romania", "Yugoslavia"}
    hubs = ["Germany", "United Kingdom", "France", "Italy"]
    labels = []
    regressors = []
    for hub in hubs:
        for direction in ("hub_to_balkans", "balkans_to_hub"):
            if direction == "hub_to_balkans":
                dyad = np.asarray([r.origin == hub and r.destination in balkans for r in sample], dtype=float)
            else:
                dyad = np.asarray([r.origin in balkans and r.destination == hub for r in sample], dtype=float)
            regressors.append(dyad * (a["year"] >= 1934))
            labels.append((hub, direction))
    x = np.column_stack(regressors)
    model = fit_ppml_hdfe(
        a["y"], x=x,
        fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
        cluster=a["pair"], tol=1e-9, max_iter=10_000,
    )
    rows = []
    for (hub, direction), beta, se in zip(labels, model.beta, model.se):
        rows.append({
            "hub": hub,
            "direction": "Hub exports to Balkans" if direction == "hub_to_balkans" else "Balkan exports to hub",
            "estimate": float(beta), "se": float(se),
            "ci_low": float(beta - 1.96 * se), "ci_high": float(beta + 1.96 * se),
            "percent": float(100.0 * (math.exp(beta) - 1.0)),
        })
    return rows, model


def full_unbalanced_attenuation(records: Sequence[PortfolioRecord]):
    pre_affinity = standardized_pre_affinity(records)
    sample = [r for r in subset_records(records, range(1928, 1938), internal_only=True)
              if r.pair in pre_affinity]
    a = arrays(sample)
    z = np.asarray([pre_affinity[r.pair] for r in sample])
    post = (a["year"] >= 1932).astype(float)
    x = z * post
    model = fit_ppml_hdfe(
        a["y"], x=x,
        fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
        cluster=a["pair"], tol=1e-9,
    )
    return {
        "estimate": float(model.beta[0]), "se": float(model.se[0]),
        "ci_low": float(model.beta[0] - 1.96 * model.se[0]),
        "ci_high": float(model.beta[0] + 1.96 * model.se[0]),
        "nobs": model.nobs,
    }


def external_share_trajectories(records: Sequence[PortfolioRecord]):
    rows = []
    for year in range(1927, 1938):
        annual = [r for r in records if r.year == year and r.external]
        for r in annual:
            rows.append({"year": year, "reporter": r.reporter, "external_share": r.share})
    return rows


def main():
    for folder in (DERIVED, TAB):
        folder.mkdir(parents=True, exist_ok=True)
    print("1/4  Constructing the analysis samples", flush=True)
    panel = load_panel(RAW / "panel_data.csv")
    audit = audit_panel(panel)
    (DERIVED / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(DERIVED / "reporter_year_closure.csv", audit["closure"])
    export_bal, export_anom = build_balanced_portfolio(panel, "export")
    import_bal, import_anom = build_balanced_portfolio(panel, "import")
    export_full = build_full_internal(panel, "export")
    import_full = build_full_internal(panel, "import")
    (DERIVED / "portfolio_anomalies.json").write_text(
        json.dumps({"export": export_anom, "import": import_anom}, indent=2), encoding="utf-8")
    for name, records in [
        ("balanced_export_portfolio.csv", export_bal),
        ("balanced_import_portfolio.csv", import_bal),
        ("full_export_internal.csv", export_full),
        ("full_import_internal.csv", import_full),
    ]:
        write_csv(DERIVED / name, [r.__dict__ | {
            "pair": r.pair, "origin_year": r.origin_year,
            "destination_year": r.destination_year} for r in records])
    geography = []
    for i in CORE19:
        for j in CORE19:
            if i != j:
                x = geography_matrix([i], [j])[0]
                geography.append({"origin": i, "destination": j,
                    "log_capital_distance": x[0], "interwar_border": int(x[1])})
    write_csv(DERIVED / "historical_geography.csv", geography)

    print("2/4  Estimating the appendix and regional specifications", flush=True)
    exp_sticky, _ = sticky_by_period(export_bal)
    imp_sticky, _ = sticky_by_period(import_bal)
    for r in exp_sticky: r["declaration"] = "Exporter"
    for r in imp_sticky: r["declaration"] = "Importer"
    write_csv(TAB / "lag_persistence.csv", exp_sticky + imp_sticky)

    exp_event, exp_breaks, exp_best, _ = attenuation_event_study(export_bal)
    imp_event, imp_breaks, imp_best, _ = attenuation_event_study(import_bal)
    for r in exp_event: r["declaration"] = "Exporter"
    for r in imp_event: r["declaration"] = "Importer"
    for r in exp_breaks: r["declaration"] = "Exporter"
    for r in imp_breaks: r["declaration"] = "Importer"
    write_csv(TAB / "annual_attenuation.csv", exp_event + imp_event)
    write_csv(TAB / "break_profiles.csv", exp_breaks + imp_breaks)

    exp_germany, exp_models = germany_event_study(export_full)
    imp_germany, imp_models = germany_event_study(import_full)
    for r in exp_germany: r["declaration"] = "Exporter"
    for r in imp_germany: r["declaration"] = "Importer"
    write_csv(TAB / "germany_event_profiles.csv", exp_germany + imp_germany)
    exp_diagnostics = germany_pretrend_and_trend_adjustment(export_full, exp_models)
    imp_diagnostics = germany_pretrend_and_trend_adjustment(import_full, imp_models)
    for r in exp_diagnostics: r["declaration"] = "Exporter"
    for r in imp_diagnostics: r["declaration"] = "Importer"
    write_csv(TAB / "germany_trend_diagnostics.csv", exp_diagnostics + imp_diagnostics)
    exp_hubs, _ = hub_placebo_model(export_full)
    imp_hubs, _ = hub_placebo_model(import_full)
    for r in exp_hubs: r["declaration"] = "Exporter"
    for r in imp_hubs: r["declaration"] = "Importer"
    write_csv(TAB / "regional_hub_comparisons.csv", exp_hubs + imp_hubs)
    exp_full_att = full_unbalanced_attenuation(export_full); exp_full_att["declaration"] = "Exporter"
    imp_full_att = full_unbalanced_attenuation(import_full); imp_full_att["declaration"] = "Importer"
    write_csv(TAB / "full_panel_attenuation.csv", [exp_full_att, imp_full_att])
    exp_external = external_share_trajectories(export_bal)
    imp_external = external_share_trajectories(import_bal)
    for r in exp_external: r["declaration"] = "Exporter"
    for r in imp_external: r["declaration"] = "Importer"
    write_csv(TAB / "external_share_trajectories.csv", exp_external + imp_external)

    print("3/4  Estimating the portfolio predictions and sensitivity analyses", flush=True)
    from portfolio_analysis import run_portfolio_analysis
    primary = run_portfolio_analysis(ROOT, export_bal, import_bal, export_full, import_full)
    summary = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "system": platform.system()},
        "audit": {"n_rows": audit["n_rows"], "countries": len(audit["countries"]),
                  "missing_export": audit["missing_export"], "missing_import": audit["missing_import"],
                  "anomalies": {"export": export_anom, "import": import_anom}},
        "sticky": {"export": exp_sticky, "import": imp_sticky},
        "attenuation": {"export": exp_event, "import": imp_event,
                        "export_best_break": exp_best, "import_best_break": imp_best},
        "germany": {"export_event": exp_germany, "import_event": imp_germany,
                    "export_hubs": exp_hubs, "import_hubs": imp_hubs,
                    "export_diagnostics": exp_diagnostics, "import_diagnostics": imp_diagnostics},
        "full_unbalanced": {"export": exp_full_att, "import": imp_full_att},
        "external_shares": {"export": exp_external, "import": imp_external},
        # Retain the established result key for compatibility with publication assets.
        "revised": primary,
        "inference_convention": {
            "coefficient_covariance": "directed-pair cluster sandwich, G/(G-1) multiplier",
            "critical_values": "normal approximations; no claim to exact small-treated-group inference",
            "germany_reference_year": 1933,
            "germany_joint_pretrend_years": list(range(1928, 1933)),
            "attenuation_reference_year": 1929,
            "attenuation_support": "positive 1927-1929 mean; exact log; within-reporter standardization",
            "lag_diagnostic_support": "strictly positive recorded lag; current zeros retained; no pseudo-count",
            "automatic_separation_scope": "exactly zero-total fixed-effect groups only",
        },
    }
    (ROOT / "output" / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("4/4  Building the publication tables and figures", flush=True)
    from build_manuscript_assets import build_assets
    build_assets(ROOT)
    print(json.dumps({"status": "success", "established_link_endpoint": [
        r for r in primary["established_prediction"] if r["year"] == 1937]}, indent=2))


if __name__ == "__main__":
    main()
