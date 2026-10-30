#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trade_data import load_panel, build_balanced_portfolio  # noqa: E402

panel = load_panel(ROOT / "data" / "raw" / "panel_data.csv")
exports, export_anom = build_balanced_portfolio(panel, "export")
imports, import_anom = build_balanced_portfolio(panel, "import")

assert len(panel) == 5040
assert len(exports) == 3952, len(exports)
assert len(imports) == 3971, len(imports)
assert len(export_anom) == 1, export_anom
assert export_anom[0]["type"] == "nonclosing"
assert export_anom[0]["reporter"] == "Spain"
assert export_anom[0]["year"] == 1937
assert math.isclose(float(export_anom[0]["internal_sum"]), 1.164, rel_tol=0, abs_tol=1e-12)
assert import_anom == []

# Shares and nominal values use matching reporter totals in the
# 1927--1937 estimation window. Archived 1938 observations are not estimated.
for row in panel:
    if row.year > 1937:
        continue
    if row.export_share is not None and row.export_value is not None:
        assert math.isclose(row.export_share, row.export_value / row.total_exports,
                            rel_tol=0, abs_tol=1e-10), (row.reporter, row.partner, row.year, "export")
    if row.import_share is not None and row.import_value is not None:
        assert math.isclose(row.import_share, row.import_value / row.total_imports,
                            rel_tol=0, abs_tol=1e-10), (row.reporter, row.partner, row.year, "import")

for records in (exports, imports):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for row in records:
        key = (row.reporter, row.year)
        sums[key] += row.share
        counts[key] += 1
    for key, total in sums.items():
        assert counts[key] == 19, (key, counts[key])
        assert math.isclose(total, 1.0, rel_tol=0, abs_tol=2e-10), (key, total)

print("PASS: raw dimensions, analysis-window value/share consistency, anomaly handling, and portfolio closure")
