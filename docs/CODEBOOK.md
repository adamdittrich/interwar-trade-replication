# Data and output codebook

## Input and historical sources

`data/raw/panel_data.csv` is the final machine-readable transcription used by the analysis, not a set of digitized source-page images. It has 5,040 records (21 reporting countries × 20 partners × 12 years, 1927–1938) and 13 fields. Estimation is restricted to 1927–1937. The source grid and values are retained unchanged in this compact release.

Principal publications:

- League of Nations (1930), *Memorandum on International Trade and Balances of Payments, 1927–1929*. Geneva.
- League of Nations (1937), *International Trade Statistics, 1936*. Geneva.
- League of Nations (1939), *International Trade Statistics, 1938*. Geneva.

Where successive publications report a revised value, the transcription convention selects the later value. Some shares are reported directly; others are calculated from a bilateral nominal value and the corresponding reporter world total. Some nominal values are derived from reported shares. Agreement between those fields is therefore not independent measurement. The deposited input supports replication from the completed transcription; the original printed tables are not reproduced in this archive.

## Variables

| Variable | Type | Definition |
|---|---:|---|
| `year` | integer | Calendar year, 1927–1938. |
| `exporter` | string | Reporting country in the source workbook. It always equals `reporting_country`; retained to preserve the original schema. |
| `importer` | string | Named partner country. The analysis code reads this field as `partner`. |
| `import_value_raw` | numeric/missing | Reporting country’s imports from the partner in source currency and scale. The original protocol uses special trade except for Norway, which uses general trade. Some values are reconstructed from reported shares. |
| `import_share_world` | numeric/missing | Share of the reporter's world imports. Usually calculated as `import_value_raw / total_imports`; in some records the share is entered first and the nominal value is derived from it. |
| `export_value_raw` | numeric/missing | Reporting country’s exports to the partner in source currency and scale. The original protocol uses special trade except for Norway. Derived nominal values use the reported share and the corresponding world export total. |
| `export_share_world` | numeric/missing | Share of the reporter's world exports. Usually calculated as `export_value_raw / total_exports`; some nominal values are derived from the recorded share. |
| `currency` | string | Currency used in the source table for the reporting country-year. |
| `multiplier` | numeric | Source-table scale (for example, thousands or millions). |
| `total_imports` | numeric | Reporting country’s total imports in the same currency/scale. |
| `total_exports` | numeric | Reporting country’s total exports in the same currency/scale. |
| `reporting_country` | string | Country whose customs/statistical publication supplies both declarations. |
| `pair_id` | string | Original reporter–partner identifier. |


## Measurement and sample conventions

Blank fields are missing and are not filled from mirror reports. Numerical zeros are retained and can include negligible flows below the source's reporting threshold. The input does not distinguish every printed zero from a below-threshold dash; a later positive entry therefore need not mark the beginning of an economic relationship.

The trade concept is special trade except for Norway's general-trade series. National customs territories, valuation, and coverage need not be identical. Shares remove source currency units but do not harmonize these differences. Nominal values retain their row's source currency and multiplier and should not be pooled as common-currency levels. The archived 1938 rows are outside the study's maintained sample and are not used by its estimators.

The reporters are Austria, Belgium, Bulgaria, Czechoslovakia, Denmark, Finland, France, Germany, Greece, Hungary, Italy, the Netherlands, Norway, Poland, Romania, Spain, Sweden, Switzerland, the United Kingdom, the USSR, and Yugoslavia. `exporter` is the reporting country in the input, including for import declarations; it should not be read as the economic origin of every observation.

The complete core excludes Bulgaria and Yugoslavia from separately identified partner columns. Its fixed `External aggregate` includes both countries and every country outside the core. Each complete reporter-year contains 18 core partners and this residual. Spain's 1937 export shares sum to 1.164, so its entire export reporter-year is excluded. Its import account is retained. The full observed 21-country internal sample retains missing observations as missing and is used for regional and sample comparisons.

Export declarations are oriented reporter → partner; import declarations are oriented partner → reporter. The two systems are never pooled as independent observations of the same flow. An established relationship has a positive recorded total over 1927–1929. Later zeros remain observations, and restricting the support does not renormalize the reporter's original world shares.

Geographic inputs are the fixed capital coordinates and explicit interwar land-border pairs in `src/trade_data.py`. The code computes great-circle capital distance and its natural logarithm. The derived geography file contains `origin`, `destination`, `log_capital_distance`, and `interwar_border`. This is the paper's stated distance-and-border benchmark, not a comprehensive measure of transport costs.

## Generated analysis samples

`python run_all.py` creates `data/derived/`. These files are intermediates, not additional inputs.

`balanced_export_portfolio.csv`, `balanced_import_portfolio.csv`, `full_export_internal.csv`, and `full_import_internal.csv` contain:

| Field | Meaning |
|---|---|
| `year` | Calendar year, 1927–1937. |
| `origin`, `destination` | Economic direction of the trade flow. |
| `reporter`, `partner` | Reporting country and named partner or external aggregate. |
| `share` | Fraction of the reporter's world exports or world imports. |
| `external` | Whether the record is the outside-core residual. |
| `declaration` | `export` or `import`. |
| `pair` | Directed `origin->destination` fixed-effect identifier. |
| `origin_year`, `destination_year` | Country–year fixed-effect identifiers. |

The pipeline also writes sample counts, reporter-year sums, and the geography inputs. Those generated records document the actual estimation samples.

## Numerical outputs

`output/analysis_summary.json` contains the portfolio results under the stable key `revised`, alongside `sticky`, `attenuation`, `germany`, `full_unbalanced`, `external_shares`, and `inference_convention`. `runtime` records software versions. The stored field name `revised` is a schema identifier, not a separate run.

All CSV results are in `output/tables/`:

| Files | Results |
|---|---|
| `established_link_prediction.csv`, `recorded_entry.csv` | Annual inherited-map fit and trade recorded in previously zero cells. |
| `history_and_geography.csv`, `fixed_horizon_comparison.csv` | Geographic and recent-map benchmarks. |
| `matched_declaration_comparison.csv`, `sample_robustness.csv` | Matched exporter/importer comparison and sample alternatives. |
| `smoothing_sensitivity.csv`, `geography_conditioned_reassignment.csv` | Appendix zero-treatment and reassignment comparisons. |
| `country_deletions.csv`, `geography_country_deletions.csv`, `matched_country_deletions.csv` | Country-deletion calculations underlying sensitivity ranges. |
| `entry_by_country.csv`, `reassignment_draws.csv` | Reporter-level entry shares and individual reassignment draws. |
| `annual_attenuation.csv`, `break_profiles.csv`, `full_panel_attenuation.csv` | Supporting attenuation and candidate-break specifications. |
| `germany_event_profiles.csv`, `germany_trend_diagnostics.csv`, `regional_hub_comparisons.csv` | Regional event profiles, pre-period restrictions, and hub comparisons. |
| `lag_persistence.csv` | Positive-lag predictive specifications. |
| `external_share_trajectories.csv` | Reporter-year outside-core shares. |

Shares, scores, and total-variation fields in CSV/JSON are fractions unless the field explicitly names a percentage. Table and figure assets apply their displayed scaling. Deviances are unscaled loss values. `estimate` and `se` are coefficients and standard errors; event-study percentage transformations are labeled separately. Country-deletion ranges describe sensitivity, not confidence intervals for a causal effect.

`MANUSCRIPT_OBJECT_MAP.csv` gives the actual manuscript number, generated file, and source block for every table and figure. `NUMERICAL_PROSE_MAP.csv` maps numerical text macros. These prevent confusion between publication numbering and filenames.
