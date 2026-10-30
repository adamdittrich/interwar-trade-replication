"""Data construction and audit helpers for Old Partners, New Hierarchies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import csv
import math
import numpy as np

CORE19 = [
    "Austria", "Belgium", "Czechoslovakia", "Denmark", "Finland",
    "France", "Germany", "Greece", "Hungary", "Italy", "Netherlands",
    "Norway", "Poland", "Romania", "Spain", "Sweden", "Switzerland",
    "USSR", "United Kingdom",
]
FULL21 = [
    "Austria", "Belgium", "Bulgaria", "Czechoslovakia", "Denmark",
    "Finland", "France", "Germany", "Greece", "Hungary", "Italy",
    "Netherlands", "Norway", "Poland", "Romania", "Spain", "Sweden",
    "Switzerland", "USSR", "United Kingdom", "Yugoslavia",
]
EXTERNAL = "External aggregate"

CAPITAL_COORDS = {
    "Austria": (48.2082, 16.3738),
    "Belgium": (50.8503, 4.3517),
    "Czechoslovakia": (50.0755, 14.4378),
    "Denmark": (55.6761, 12.5683),
    "Finland": (60.1699, 24.9384),
    "France": (48.8566, 2.3522),
    "Germany": (52.5200, 13.4050),
    "Greece": (37.9838, 23.7275),
    "Hungary": (47.4979, 19.0402),
    "Italy": (41.9028, 12.4964),
    "Netherlands": (52.3676, 4.9041),
    "Norway": (59.9139, 10.7522),
    "Poland": (52.2297, 21.0122),
    "Romania": (44.4268, 26.1025),
    "Spain": (40.4168, -3.7038),
    "Sweden": (59.3293, 18.0686),
    "Switzerland": (46.9480, 7.4474),
    "USSR": (55.7558, 37.6173),
    "United Kingdom": (51.5074, -0.1278),
}

# Interwar land borders among the balanced 19-country core. The list is deliberately
# explicit in the replication files so that no contemporary-border database is silently used.
BORDER_PAIRS = {
    frozenset(x) for x in [
        ("Germany", "Austria"), ("Germany", "Belgium"),
        ("Germany", "Denmark"), ("Germany", "France"),
        ("Germany", "Netherlands"), ("Germany", "Poland"),
        ("Germany", "Czechoslovakia"), ("Germany", "Switzerland"),
        ("Austria", "Czechoslovakia"), ("Austria", "Hungary"),
        ("Austria", "Italy"), ("Austria", "Switzerland"),
        ("Belgium", "France"), ("Belgium", "Netherlands"),
        ("Spain", "France"),
        ("Finland", "USSR"), ("Finland", "Norway"), ("Finland", "Sweden"),
        ("France", "Italy"), ("France", "Switzerland"),
        ("Hungary", "Czechoslovakia"), ("Hungary", "Romania"),
        ("Italy", "Switzerland"), ("Norway", "Sweden"),
        ("Poland", "Czechoslovakia"), ("Poland", "USSR"),
        ("Poland", "Romania"), ("Romania", "USSR"),
        ("Romania", "Czechoslovakia"),
    ]
}


@dataclass(frozen=True)
class TradeRecord:
    year: int
    reporter: str
    partner: str
    export_share: float | None
    import_share: float | None
    export_value: float | None
    import_value: float | None
    total_exports: float
    total_imports: float
    currency: str
    multiplier: float


@dataclass(frozen=True)
class PortfolioRecord:
    year: int
    origin: str
    destination: str
    reporter: str
    partner: str
    share: float
    external: bool
    declaration: str

    @property
    def pair(self) -> str:
        return f"{self.origin}->{self.destination}"

    @property
    def origin_year(self) -> str:
        return f"{self.origin}|{self.year}"

    @property
    def destination_year(self) -> str:
        return f"{self.destination}|{self.year}"


def _num(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_panel(path: str | Path) -> list[TradeRecord]:
    records: list[TradeRecord] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reporter = row["reporting_country"].strip()
            partner = row["importer"].strip()  # exporter always equals reporter in the source panel
            records.append(
                TradeRecord(
                    year=int(float(row["year"])),
                    reporter=reporter,
                    partner=partner,
                    export_share=_num(row["export_share_world"]),
                    import_share=_num(row["import_share_world"]),
                    export_value=_num(row["export_value_raw"]),
                    import_value=_num(row["import_value_raw"]),
                    total_exports=float(row["total_exports"]),
                    total_imports=float(row["total_imports"]),
                    currency=row["currency"].strip(),
                    multiplier=float(row["multiplier"]),
                )
            )
    return records


def index_panel(records: Sequence[TradeRecord]) -> dict[tuple[str, str, int], TradeRecord]:
    return {(r.reporter, r.partner, r.year): r for r in records}


def build_balanced_portfolio(
    records: Sequence[TradeRecord],
    declaration: str,
    years: Iterable[int] = range(1927, 1938),
    core: Sequence[str] = CORE19,
    external_name: str = EXTERNAL,
    drop_nonclosing: bool = True,
) -> tuple[list[PortfolioRecord], list[dict[str, object]]]:
    """Construct complete reporter portfolios with a stable outside-core residual.

    Export declarations are oriented reporter -> partner. Import declarations are oriented
    partner -> reporter so that both datasets describe the same directed economic relation from
    opposite statistical declarations.
    """
    if declaration not in {"export", "import"}:
        raise ValueError("declaration must be 'export' or 'import'")
    idx = index_panel(records)
    out: list[PortfolioRecord] = []
    anomalies: list[dict[str, object]] = []

    for year in years:
        for reporter in core:
            internal: list[tuple[str, float]] = []
            missing: list[str] = []
            for partner in core:
                if partner == reporter:
                    continue
                rec = idx[(reporter, partner, year)]
                value = rec.export_share if declaration == "export" else rec.import_share
                if value is None:
                    missing.append(partner)
                else:
                    internal.append((partner, float(value)))

            if missing:
                anomalies.append({
                    "type": "missing_internal",
                    "declaration": declaration,
                    "year": year,
                    "reporter": reporter,
                    "partners": "; ".join(missing),
                })
                continue

            internal_sum = float(sum(v for _, v in internal))
            residual = 1.0 - internal_sum
            if residual < -1e-8:
                anomalies.append({
                    "type": "nonclosing",
                    "declaration": declaration,
                    "year": year,
                    "reporter": reporter,
                    "internal_sum": internal_sum,
                    "residual": residual,
                })
                if drop_nonclosing:
                    continue
            if -1e-8 <= residual < 0:
                residual = 0.0

            if declaration == "export":
                for partner, share in internal:
                    out.append(PortfolioRecord(year, reporter, partner, reporter, partner, share, False, declaration))
                out.append(PortfolioRecord(year, reporter, external_name, reporter, external_name, residual, True, declaration))
            else:
                for partner, share in internal:
                    out.append(PortfolioRecord(year, partner, reporter, reporter, partner, share, False, declaration))
                out.append(PortfolioRecord(year, external_name, reporter, reporter, external_name, residual, True, declaration))

    return out, anomalies


def build_full_internal(
    records: Sequence[TradeRecord], declaration: str, years: Iterable[int] = range(1927, 1938)
) -> list[PortfolioRecord]:
    """Create the full 21-country internal panel, retaining missingness as missing/omitted."""
    allowed_years = set(years)
    out: list[PortfolioRecord] = []
    for rec in records:
        if rec.year not in allowed_years:
            continue
        value = rec.export_share if declaration == "export" else rec.import_share
        if value is None:
            continue
        if declaration == "export":
            # Spain's 1937 export account is arithmetically non-closing (116.4 percent).
            if rec.reporter == "Spain" and rec.year == 1937:
                continue
            origin, destination = rec.reporter, rec.partner
        else:
            origin, destination = rec.partner, rec.reporter
        out.append(
            PortfolioRecord(rec.year, origin, destination, rec.reporter, rec.partner,
                            float(value), False, declaration)
        )
    return out


def portfolio_arrays(records: Sequence[PortfolioRecord]):
    return {
        "y": np.asarray([r.share for r in records], dtype=float),
        "year": np.asarray([r.year for r in records], dtype=int),
        "origin": np.asarray([r.origin for r in records], dtype=object),
        "destination": np.asarray([r.destination for r in records], dtype=object),
        "reporter": np.asarray([r.reporter for r in records], dtype=object),
        "partner": np.asarray([r.partner for r in records], dtype=object),
        "pair": np.asarray([r.pair for r in records], dtype=object),
        "origin_year": np.asarray([r.origin_year for r in records], dtype=object),
        "destination_year": np.asarray([r.destination_year for r in records], dtype=object),
        "external": np.asarray([r.external for r in records], dtype=bool),
    }


def great_circle_km(a: str, b: str) -> float:
    lat1, lon1 = map(math.radians, CAPITAL_COORDS[a])
    lat2, lon2 = map(math.radians, CAPITAL_COORDS[b])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(h))


def geography_matrix(origins: Sequence[str], destinations: Sequence[str]) -> np.ndarray:
    log_distance = []
    border = []
    for origin, destination in zip(origins, destinations):
        log_distance.append(math.log(great_circle_km(origin, destination)))
        border.append(float(frozenset((origin, destination)) in BORDER_PAIRS))
    return np.column_stack([log_distance, border])


def audit_panel(records: Sequence[TradeRecord]) -> dict[str, object]:
    years = sorted({r.year for r in records})
    countries = sorted({r.reporter for r in records})
    missing_export = sum(r.export_share is None for r in records)
    missing_import = sum(r.import_share is None for r in records)
    missing_by_partner: dict[str, int] = {}
    for r in records:
        if r.export_share is None or r.import_share is None:
            missing_by_partner[r.partner] = missing_by_partner.get(r.partner, 0) + 1

    closure = []
    for year in years:
        for reporter in countries:
            rr = [r for r in records if r.year == year and r.reporter == reporter]
            ex = sum(r.export_share or 0.0 for r in rr)
            im = sum(r.import_share or 0.0 for r in rr)
            closure.append({
                "year": year,
                "reporter": reporter,
                "export_observed_sum": ex,
                "import_observed_sum": im,
                "missing_partners": sum(r.export_share is None for r in rr),
            })

    return {
        "n_rows": len(records),
        "years": years,
        "countries": countries,
        "missing_export": missing_export,
        "missing_import": missing_import,
        "missing_by_partner": dict(sorted(missing_by_partner.items(), key=lambda kv: -kv[1])),
        "closure": closure,
    }
