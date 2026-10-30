"""Established-link portfolio predictions and transparent comparison exercises.

The primary outcome remains a share of the reporting country's world trade.
Only the prediction support is restricted: an established cell has a positive
sum of recorded shares during 1927--1929.  No pseudo-count is used on that
support.  Positive outcomes outside it are accounted for separately, without
being reallocated to the External category.

These are conditional allocation exercises: evaluation-year margins of the
selected share matrix are observed and refitted, not forecast.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ppml_hdfe import fit_ppml_hdfe, poisson_deviance
from trade_data import EXTERNAL, PortfolioRecord, geography_matrix, portfolio_arrays

PRE = (1927, 1928, 1929)
YEARS = tuple(range(1930, 1938))


def _csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select(records, years=None, support=None, internal=False, exclude=()):
    ys = None if years is None else set(years)
    removed = set(exclude)
    return [r for r in records
            if (ys is None or r.year in ys)
            and (support is None or r.pair in support)
            and (not internal or not r.external)
            and r.origin not in removed and r.destination not in removed]


def _support(records, years=PRE):
    sums = defaultdict(float)
    for r in _select(records, years):
        sums[r.pair] += r.share
    return {pair for pair, total in sums.items() if total > 0.0}


def _fit_affinity(records, years=PRE, support=None, epsilon=0.0):
    sample = _select(records, years, support)
    a = portfolio_arrays(sample)
    model = fit_ppml_hdfe(
        a["y"] + epsilon,
        fixed_effects=[a["pair"], a["origin_year"], a["destination_year"]],
        tol=1e-11, max_iter=10000,
    )
    if not model.converged:
        raise RuntimeError("Pre-period affinity model did not converge")
    offsets = {str(p): float(v)
               for p, v in zip(model.fe_levels[0], model.fe_effects[0])}
    if set(a["pair"]) != set(offsets):
        raise RuntimeError("Training support contains a separated pair")
    return offsets


def _fit_geography(records, years=PRE, support=None):
    sample = _select(records, years, support, internal=True)
    a = portfolio_arrays(sample)
    model = fit_ppml_hdfe(
        a["y"], x=geography_matrix(a["origin"], a["destination"]),
        fixed_effects=[a["origin_year"], a["destination_year"]],
        cluster=a["pair"], tol=1e-11, max_iter=10000,
    )
    if not model.converged:
        raise RuntimeError("Geography model did not converge")
    return model


def _prediction(sample, affinity=None, geo_beta=None, vector=None):
    a = portfolio_arrays(sample)
    if vector is not None:
        offset = np.asarray(vector, dtype=float)
    elif affinity is not None:
        missing = set(a["pair"]) - set(affinity)
        if missing:
            raise RuntimeError(f"No pre-period prediction for pairs: {sorted(missing)}")
        offset = np.array([affinity[str(p)] for p in a["pair"]])
    elif geo_beta is not None:
        offset = geography_matrix(a["origin"], a["destination"]) @ geo_beta
    else:
        offset = np.zeros(len(sample))
    model = fit_ppml_hdfe(
        a["y"], fixed_effects=[a["origin"], a["destination"]],
        offset=offset, tol=1e-12, max_iter=10000,
    )
    if not model.converged:
        raise RuntimeError("Conditional allocation model did not converge")
    # A zero-total evaluation margin has a zero fitted mean, not a missing value.
    mu = np.where(model.keep, model.mu, 0.0)
    return a, mu


def _metrics(sample, historical, null):
    a = portfolio_arrays(sample)
    y = a["y"]
    d_hist = poisson_deviance(y, historical)
    d_null = poisson_deviance(y, null)
    reporters = sorted(set(a["reporter"]))
    conditional_tv = []
    max_margin_error = 0.0
    for reporter in reporters:
        mask = a["reporter"] == reporter
        total = float(y[mask].sum())
        max_margin_error = max(max_margin_error,
                               abs(total - float(historical[mask].sum())))
        if total > 0:
            conditional_tv.append(float(0.5 * np.abs(y[mask] - historical[mask]).sum() / total))
    return {
        "score": float(1.0 - d_hist / d_null),
        "historical_deviance": float(d_hist),
        "margins_deviance": float(d_null),
        "tv_world_share": float(0.5 * np.abs(y - historical).sum() / len(reporters)),
        "tv_conditional_on_support": float(np.mean(conditional_tv)),
        "supported_world_share_mean": float(y.sum() / len(reporters)),
        "reporter_margin_max_error": float(max_margin_error),
        "n_observations": len(sample),
        "n_reporters": len(reporters),
    }


def _series(records, support, training=PRE, years=YEARS, geography=False):
    affinity = _fit_affinity(records, training, support)
    geo = _fit_geography(records, training, support) if geography else None
    rows = []
    for year in years:
        sample = _select(records, [year], support)
        a, historical = _prediction(sample, affinity=affinity)
        _, null = _prediction(sample)
        row = {"year": int(year), **_metrics(sample, historical, null)}
        if geo is not None:
            _, predicted_geo = _prediction(sample, geo_beta=geo.beta)
            dg = poisson_deviance(a["y"], predicted_geo)
            row.update({
                "geography_deviance": float(dg),
                "geography_score": float(1.0 - dg / row["margins_deviance"]),
                "history_increment_over_geography": float(1.0 - row["historical_deviance"] / dg),
            })
        rows.append(row)
    return rows, affinity, geo


def _jackknife(full_rows, replications, key):
    """Conventional delete-country jackknife sensitivity intervals, not population CIs."""
    by_year = defaultdict(list)
    for rows in replications:
        for row in rows:
            by_year[row["year"]].append(row[key])
    estimates = {row["year"]: row[key] for row in full_rows}
    out = {}
    for year, values in by_year.items():
        v = np.asarray(values, dtype=float)
        n = len(v)
        center = float(v.mean())
        se = float(np.sqrt((n - 1) / n * np.sum((v - center) ** 2)))
        out[year] = {
            "jk_n_countries": n,
            "jk_se": se,
            "jk_interval_low": float(estimates[year] - 1.96 * se),
            "jk_interval_high": float(estimates[year] + 1.96 * se),
            "delete_country_min": float(v.min()),
            "delete_country_max": float(v.max()),
        }
    return out


def _entry_rows(records, support, declaration):
    aggregate, country = [], []
    pre_values = defaultdict(list)
    for r in _select(records, PRE):
        pre_values[r.pair].append(r.share)
    pre_mean = {k: float(np.mean(v)) for k, v in pre_values.items()}
    for year in YEARS:
        annual = _select(records, [year])
        reporters = sorted({r.reporter for r in annual})
        local = []
        for reporter in reporters:
            rr = [r for r in annual if r.reporter == reporter]
            entered = [r for r in rr if r.pair not in support and r.share > 0]
            exited = [r for r in rr if r.pair in support and r.share == 0]
            internal_share = float(sum(r.share for r in rr if not r.external))
            new_mass = float(sum(r.share for r in entered if not r.external))
            row = {
                "declaration": declaration, "year": year, "reporter": reporter,
                "newly_positive_world_share": new_mass,
                "internal_world_share": internal_share,
                "newly_positive_fraction_of_internal": new_mass / internal_share if internal_share > 0 else None,
                "newly_positive_recorded_cells": len(entered),
                "now_zero_established_cells": len(exited),
                "pre_share_of_now_zero_cells": float(sum(pre_mean[r.pair] for r in exited)),
            }
            local.append(row)
        country.extend(local)
        aggregate.append({
            "declaration": declaration, "year": year,
            "mean_newly_positive_world_share": float(np.mean([r["newly_positive_world_share"] for r in local])),
            "mean_newly_positive_fraction_of_internal": float(np.mean([r["newly_positive_fraction_of_internal"] for r in local if r["newly_positive_fraction_of_internal"] is not None])),
            "pooled_normalized_share_fraction_newly_positive_internal": float(sum(r["newly_positive_world_share"] for r in local) / sum(r["internal_world_share"] for r in local)),
            "max_newly_positive_world_share": float(max(r["newly_positive_world_share"] for r in local)),
            "newly_positive_recorded_cells": sum(r["newly_positive_recorded_cells"] for r in local),
            "now_zero_established_cells": sum(r["now_zero_established_cells"] for r in local),
            "mean_pre_share_of_now_zero_cells": float(np.mean([r["pre_share_of_now_zero_cells"] for r in local])),
            "n_reporters": len(reporters),
        })
    return aggregate, country


def _rolling(records, anchored_support, declaration):
    """Matched-support fixed-map and rolling-map allocation comparisons.

    At each target year the two maps are estimated and assessed on exactly the
    same pair support.  That support requires positive training total under both
    maps, so neither map requires an arbitrary prediction for a training zero.
    The short 1927--1929 pre-period is not an undisturbed historical control.
    """
    rows = []
    for target in range(1928, 1938):
        for window in (1, 3):
            training = tuple(range(target - window, target))
            if min(training) < 1927:
                continue
            rolling_support = _support(records, training)
            # Earlier years provide only a descriptive one-year baseline; the
            # anchored map would otherwise use contemporaneous/future outcomes.
            matched = rolling_support & anchored_support if target >= 1930 else rolling_support
            roll, _, _ = _series(records, matched, training, [target])
            rr = roll[0]
            row = {
                "declaration": declaration, "year": target,
                "rolling_training_years": "-".join(map(str, training)),
                "training_window_years": window,
                "rolling_score": rr["score"],
                "rolling_tv_world_share": rr["tv_world_share"],
                "supported_world_share_mean": rr["supported_world_share_mean"],
                "n_observations": rr["n_observations"],
                "n_reporters": rr["n_reporters"],
            }
            if target >= 1930:
                anchored, _, _ = _series(records, matched, PRE, [target])
                fixed = anchored[0]
                row.update({
                    "anchored_score_same_support": fixed["score"],
                    "rolling_minus_anchored_score": rr["score"] - fixed["score"],
                    "anchored_tv_world_share_same_support": fixed["tv_world_share"],
                })
            rows.append(row)
    return rows


def _canonical_residual(pair_records, affinity, geo_beta):
    origins = np.asarray([r.origin for r in pair_records], dtype=object)
    destinations = np.asarray([r.destination for r in pair_records], dtype=object)
    geography = geography_matrix(origins, destinations) @ geo_beta
    raw = np.array([affinity[r.pair] for r in pair_records]) - geography
    design = np.column_stack([
        (origins[:, None] == np.unique(origins)[None, :]).astype(float),
        (destinations[:, None] == np.unique(destinations)[None, :]).astype(float),
    ])
    residual = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    # An observationally equivalent pair FE normalization must produce exactly
    # the same canonical residual.  This tests the analytical gauge invariance.
    rng = np.random.default_rng(4819)
    shift = design @ rng.normal(size=design.shape[1])
    altered = raw + shift
    check = altered - design @ np.linalg.lstsq(design, altered, rcond=None)[0]
    error = float(np.max(np.abs(residual - check)))
    if error > 1e-9:
        raise AssertionError("Canonical affinity residual is not normalization invariant")
    return {r.pair: float(v) for r, v in zip(pair_records, residual)}, error


def _reassignment(records, support, declaration, draws=499):
    internal = _select(records, internal=True)
    valid = support & {r.pair for r in internal}
    affinity = _fit_affinity(internal, PRE, valid)
    geo = _fit_geography(internal, PRE, valid)
    unique = {r.pair: r for r in _select(internal, PRE, valid)}
    pairs = [unique[k] for k in sorted(unique)]
    residual, gauge_error = _canonical_residual(pairs, affinity, geo.beta)
    sample = _select(internal, [1937], valid)
    a = portfolio_arrays(sample)
    g = geography_matrix(a["origin"], a["destination"]) @ geo.beta
    u = np.array([residual[r.pair] for r in sample])
    _, actual = _prediction(sample, vector=g + u)
    _, original = _prediction(sample, affinity=affinity)
    reconstruction_error = float(np.max(np.abs(actual - original)))
    if reconstruction_error > 1e-8:
        raise AssertionError("Canonical decomposition changed inherited predictions")
    observed = poisson_deviance(a["y"], actual)
    reporter_groups = [np.flatnonzero(a["reporter"] == name) for name in np.unique(a["reporter"])]
    rng = np.random.default_rng(20261027 if declaration == "Exporter" else 20261028)
    deviances = []
    for _ in range(draws):
        reassigned = u.copy()
        for indices in reporter_groups:
            reassigned[indices] = rng.permutation(reassigned[indices])
        _, prediction = _prediction(sample, vector=g + reassigned)
        deviances.append(poisson_deviance(a["y"], prediction))
    dev = np.asarray(deviances)
    row = {
        "declaration": declaration, "year": 1937, "draws": draws,
        "observed_deviance": float(observed),
        "reassignment_mean": float(dev.mean()),
        "reassignment_p05": float(np.quantile(dev, .05)),
        "reassignment_median": float(np.median(dev)),
        "reassignment_p95": float(np.quantile(dev, .95)),
        "reassignments_with_deviance_at_most_observed": int(np.sum(dev <= observed)),
        "canonical_normalization_error": gauge_error,
        "canonical_prediction_error": reconstruction_error,
        "interpretation": "Descriptive geography-conditioned residual reassignment; not a randomization p-value or concentration-preserving test.",
    }
    detail = [{"declaration": declaration, "draw": i + 1, "deviance": float(d)}
              for i, d in enumerate(dev)]
    return row, detail


def _epsilon_sensitivity(records, declaration):
    """Retain the full recorded matrix to expose the arbitrary zero prediction."""
    rows = []
    internal = _select(records, internal=True)
    geo = _fit_geography(internal)
    for epsilon in (1e-6, 1e-8, 1e-10, 1e-12):
        full_affinity = _fit_affinity(records, epsilon=epsilon)
        int_affinity = _fit_affinity(internal, epsilon=epsilon)
        for year in YEARS:
            sample = _select(records, [year])
            _, hist = _prediction(sample, affinity=full_affinity)
            _, null = _prediction(sample)
            met = _metrics(sample, hist, null)
            ins = _select(internal, [year])
            a, ih = _prediction(ins, affinity=int_affinity)
            _, ig = _prediction(ins, geo_beta=geo.beta)
            dg = poisson_deviance(a["y"], ig)
            rows.append({
                "declaration": declaration, "year": year, "epsilon": epsilon,
                "full_recorded_map_score": met["score"],
                "full_recorded_tv_world_share": met["tv_world_share"],
                "internal_history_increment_over_geography": float(1.0 - poisson_deviance(a["y"], ih) / dg),
            })
    return rows


def _matched_comparison(export_records, import_records):
    # External has a different directed orientation in the two declarations;
    # only internal named-country links can be held literally identical.
    ex = _select(export_records, internal=True, exclude={"Spain"})
    im = _select(import_records, internal=True, exclude={"Spain"})
    support = _support(ex) & _support(im)
    ex_rows, _, _ = _series(ex, support)
    im_rows, _, _ = _series(im, support)
    rows = []
    for x, m in zip(ex_rows, im_rows):
        assert x["year"] == m["year"] and x["n_observations"] == m["n_observations"]
        rows.append({
            "year": x["year"], "exporter_score": x["score"],
            "importer_score": m["score"], "difference_exporter_minus_importer": x["score"] - m["score"],
            "n_common_dyads": x["n_observations"], "n_countries": x["n_reporters"],
        })
    countries = sorted({r.origin for r in ex} | {r.destination for r in ex})
    replicates, detail = [], []
    for country in countries:
        dx, _, _ = _series(_select(ex, exclude={country}), support)
        dm, _, _ = _series(_select(im, exclude={country}), support)
        rr = []
        for x, m in zip(dx, dm):
            rr.append({"year": x["year"], "difference_exporter_minus_importer": x["score"] - m["score"]})
            detail.append({"omitted_country": country, "year": x["year"],
                           "exporter_score": x["score"], "importer_score": m["score"],
                           "difference_exporter_minus_importer": x["score"] - m["score"]})
        replicates.append(rr)
    intervals = _jackknife(rows, replicates, "difference_exporter_minus_importer")
    for row in rows:
        row.update(intervals[row["year"]])
    return rows, detail


def _sample_robustness(records, full_records, declaration):
    baseline_support = _support(records)
    rows = []
    for label, training in (("1927-1929", PRE), ("1927-1928", (1927, 1928)),
                            ("1928-1929", (1928, 1929)), ("1928 only", (1928,))):
        # Compare the alternative map and the main map on the same support.
        valid = baseline_support & _support(records, training)
        alt, _, _ = _series(records, valid, training, [1937])
        anchor, _, _ = _series(records, valid, PRE, [1937])
        row = {"declaration": declaration, "specification": "Training " + label,
               **alt[0], "anchor_score_same_support": anchor[0]["score"]}
        rows.append(row)
    samples = [
        ("Exclude every Germany-related dyad", _select(records, exclude={"Germany"})),
        ("Exclude every Spain-related dyad", _select(records, exclude={"Spain"})),
        ("Nineteen-country internal established dyads", _select(records, internal=True)),
    ]
    if full_records is not None:
        samples.append(("Full twenty-one-country unbalanced internal established dyads", list(full_records)))
    for label, sample in samples:
        valid = _support(sample)
        result, _, _ = _series(sample, valid, PRE, [1937])
        rows.append({"declaration": declaration, "specification": label, **result[0]})
    return rows


def run_portfolio_analysis(root, export_bal, import_bal, export_full=None, import_full=None):
    """Generate the portfolio predictions and their supporting comparisons."""
    root = Path(root)
    output = root / "output"
    tables = output / "tables"
    all_primary, all_geo, all_entry, entry_country = [], [], [], []
    all_jk, geo_jk, all_rolling, all_epsilon, all_shuffle, shuffle_draws = [], [], [], [], [], []
    all_robustness = []
    methods = {}
    for declaration, records in (("Exporter", export_bal), ("Importer", import_bal)):
        support = _support(records)
        countries = sorted({r.reporter for r in _select(records, PRE)})
        primary, _, _ = _series(records, support)
        internal = _select(records, internal=True)
        internal_support = support & {r.pair for r in internal}
        geography, _, geo_model = _series(internal, internal_support, geography=True)
        primary_reps, geography_reps = [], []
        for country in countries:
            fewer = _select(records, exclude={country})
            p, _, _ = _series(fewer, support)
            g, _, _ = _series(_select(fewer, internal=True), internal_support, geography=True)
            primary_reps.append(p)
            geography_reps.append(g)
            all_jk.extend({"declaration": declaration, "omitted_country": country, **r} for r in p)
            geo_jk.extend({"declaration": declaration, "omitted_country": country, **r} for r in g)
        pint = _jackknife(primary, primary_reps, "score")
        gint = _jackknife(geography, geography_reps, "history_increment_over_geography")
        for row in primary:
            row.update({"declaration": declaration, **pint[row["year"]]})
        for row in geography:
            row.update({"declaration": declaration, **gint[row["year"]]})
        entry, country_entry = _entry_rows(records, support, declaration)
        all_primary.extend(primary)
        all_geo.extend(geography)
        all_entry.extend(entry)
        entry_country.extend(country_entry)
        all_rolling.extend(_rolling(records, support, declaration))
        all_epsilon.extend(_epsilon_sensitivity(records, declaration))
        full_records = export_full if declaration == "Exporter" else import_full
        all_robustness.extend(_sample_robustness(records, full_records, declaration))
        shuffled, draws = _reassignment(records, support, declaration)
        all_shuffle.append(shuffled)
        shuffle_draws.extend(draws)
        methods[declaration] = {
            "training_years": list(PRE), "established_pairs_including_external": len(support),
            "established_internal_pairs": len(internal_support),
            "preperiod_zero_pairs": len({r.pair for r in _select(records, PRE)}) - len(support),
            "geography_beta_on_established_internal_support": geo_model.beta.tolist(),
            "geography_se_on_established_internal_support": geo_model.se.tolist(),
        }
    matched, matched_detail = _matched_comparison(export_bal, import_bal)
    named_tables = {
        "established_link_prediction.csv": all_primary,
        "recorded_entry.csv": all_entry,
        "history_and_geography.csv": all_geo,
        "fixed_horizon_comparison.csv": all_rolling,
        "matched_declaration_comparison.csv": matched,
        "smoothing_sensitivity.csv": all_epsilon,
        "geography_conditioned_reassignment.csv": all_shuffle,
        "sample_robustness.csv": all_robustness,
        "entry_by_country.csv": entry_country,
        "country_deletions.csv": all_jk,
        "geography_country_deletions.csv": geo_jk,
        "matched_country_deletions.csv": matched_detail,
        "reassignment_draws.csv": shuffle_draws,
    }
    for name, rows in named_tables.items():
        _csv(tables / name, rows)
    summary = {
        "estimand": "Conditional prediction on positive-total1927-1929 recorded support, in original reporter world shares; current margins of that support refitted; no smoothing.",
        "newly_positive_caveat": "Previously zero recorded cells need not represent new economic links because source rounding and reporting thresholds can censor small flows.",
        "uncertainty": "Delete-country jackknife sensitivity: remove every incident named-country dyad, keep the fixed External category unchanged; intervals are point estimate plus/minus1.96 jackknife standard errors.",
        "matched_comparison": "Internal common18-country network excluding Spain; intersection of positive-pre supports of both declarations; same directed support but original differing reporting/normalization conventions.",
        "methods": methods, "established_prediction": all_primary,
        "newly_positive_cells": all_entry, "established_geography": all_geo,
        "fixed_horizon_comparison": all_rolling,
        "matched_declaration_differences": matched,
        "full_matrix_smoothing_sensitivity": all_epsilon,
        "geography_conditioned_reassignment": all_shuffle,
        "established_sample_robustness": all_robustness,
        "generated_tables": list(named_tables),
    }
    return summary


if __name__ == "__main__":
    from trade_data import build_balanced_portfolio, build_full_internal, load_panel
    root = Path(__file__).resolve().parent
    panel = load_panel(root / "data" / "raw" / "panel_data.csv")
    ex, _ = build_balanced_portfolio(panel, "export")
    im, _ = build_balanced_portfolio(panel, "import")
    summary = run_portfolio_analysis(root, ex, im,
                                 build_full_internal(panel, "export"),
                                 build_full_internal(panel, "import"))
    print(json.dumps({"methods": summary["methods"],
                      "endpoint": [r for r in summary["established_prediction"] if r["year"] == 1937],
                      "matched_endpoint": [r for r in summary["matched_declaration_differences"] if r["year"] == 1937]}, indent=2))
