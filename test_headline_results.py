#!/usr/bin/env python3
"""Regression and identity checks for the published statistical specification."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary_path = ROOT / "output" / "analysis_summary.json"
assert summary_path.exists(), "Run `python run_all.py` first."
data = json.loads(summary_path.read_text(encoding="utf-8"))
revised = data["revised"]


def endpoint(block, declaration):
    return next(r for r in revised[block] if r["declaration"] == declaration and r["year"] == 1937)


def close(actual, expected, tol=5e-8):
    assert math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol), (actual, expected)


close(endpoint("established_prediction", "Exporter")["score"], 0.5608560699433078)
close(endpoint("established_prediction", "Importer")["score"], 0.47747596300084116)
close(endpoint("established_geography", "Exporter")["history_increment_over_geography"], 0.362263390384561)
close(endpoint("established_geography", "Importer")["history_increment_over_geography"], 0.232043546044562)
for row in revised["established_prediction"]:
    close(row["score"], 1 - row["historical_deviance"] / row["margins_deviance"], tol=2e-12)
    assert row["reporter_margin_max_error"] < 1e-9
for row in revised["geography_conditioned_reassignment"]:
    assert row["canonical_normalization_error"] < 1e-9
    assert row["canonical_prediction_error"] < 1e-8
    assert "randomization_p_value" not in row

# Event designs include 1928 and omit only the designated reference.
for declaration in ("export", "import"):
    for direction in ("Germany exports to Balkans", "Balkan exports to Germany"):
        years = {r["year"] for r in data["germany"][declaration + "_event"] if r["direction"] == direction}
        assert years == set(range(1928, 1938)) - {1933}, years
    years = {r["year"] for r in data["attenuation"][declaration]}
    assert years == set(range(1928, 1938)) - {1929}, years
assert data["inference_convention"]["germany_joint_pretrend_years"] == [1928, 1929, 1930, 1931, 1932]

for name in revised["generated_tables"]:
    assert (ROOT / "output" / "tables" / name).is_file(), name
assert not (ROOT / "output" / "tables" / "Table_A4_Permutation_Benchmark.csv").exists()
print("PASS: primary estimates, accounting identities, normalization checks, and event references")

# Every published object is present with its actual manuscript numbering.
asset_root = ROOT / "output" / "manuscript_assets"
tables = list((asset_root / "tables").glob("*.tex")) + list((asset_root / "appendix_tables").glob("*.tex"))
assert len(tables) == 26
assert len(list((asset_root / "figures").glob("*.pdf"))) == 5
for path in tables:
    text = path.read_text(encoding="utf-8")
    assert r"\textit{Notes:}" in text and r"\textit{Source:}" in text, path.name
with (asset_root / "MANUSCRIPT_OBJECT_MAP.csv").open(newline="", encoding="utf-8") as f:
    mapped = list(csv.DictReader(f))
labels = {r["publication_label"] for r in mapped if r["kind"] in {"table", "figure"}}
expected = {f"Table {n}" for n in range(1, 4)} | {f"Table A{n}" for n in range(1, 24)} | {f"Figure {n}" for n in range(1, 6)}
assert labels == expected
print("PASS: all 26 tables, 5 figures, notes, sources, and publication labels")
