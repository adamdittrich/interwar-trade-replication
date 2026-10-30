# Old Partners, New Hierarchies — replication materials

**Adam Dittrich**  
*Trade Reallocation in Interwar Europe, 1927–1937*

This archive contains the analysis input, executable code, and outputs for every table and figure in the article and online appendix. It is a computational replication package; the manuscript, editorial correspondence, and document-production archive are separate.

## Reproduce the results

Use Python 3.12 or 3.13. In the extracted `EREH_Replication` folder, run:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

An isolated Python environment is recommended. On Windows, `py -3` can be used instead of `python`. The pinned dependencies are NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. The package has been run with Python 3.13.5 on Linux. No internet connection, spreadsheet application, proprietary software, or account credentials are required after the dependencies are installed.

The complete run rebuilds the results from `data/raw/panel_data.csv`, writes intermediate samples to `data/derived/`, and finishes with `"status": "success"`. Intermediate samples are generated rather than duplicated in the download. Supplied output files may be removed before running; they are not estimation inputs.

## Where to look

| Path | Contents |
|---|---|
| `data/raw/panel_data.csv` | Final machine-readable input: 5,040 reporter–partner–year records. |
| `run_all.py`, `portfolio_analysis.py`, `src/` | Sample construction, estimators, and all reported specifications. |
| `build_manuscript_assets.py` | Publication table and figure generator. |
| `output/manuscript_assets/tables/` | The 3 main-paper tables, in LaTeX. |
| `output/manuscript_assets/appendix_tables/` | The 23 appendix tables, in LaTeX. |
| `output/manuscript_assets/figures/` | The 5 main-paper figures, as vector PDFs. |
| `output/tables/` | Full-precision estimates and the supporting sensitivity calculations, as CSV. |
| `output/analysis_summary.json` | Complete retained result blocks and software versions. |
| `output/manuscript_assets/MANUSCRIPT_OBJECT_MAP.csv` | Each published table/figure number, its output file, and its numerical source. |
| `docs/CODEBOOK.md` | Variables, historical sources, and sample conventions. |
| `docs/COMPUTATIONAL_DETAILS.md` | Estimands, algorithms, numerical tolerances, and inference. |

Generated numerical-text snippets and their map are also retained under `output/manuscript_assets/`. Table source attributions and notes are included in the LaTeX files. Figure captions, notes, and source attributions are in `docs/FIGURE_NOTES.md`. Numbering in the article and appendix is given by the object map, not by CSV filenames.

The LaTeX files are the exact table assets used in the paper, not standalone documents. Reproducing the statistical outputs requires Python only. They can be included in a LaTeX document using `booktabs`, `threeparttable`, `tabularx`, `longtable`, `pdflscape`, `float`, `natbib`, and the paper's `\External` abbreviation and `\parencite` alias (`\citep`); cited entries are in `docs/references.bib`.

## Scope and sources

The observations are a final transcription of League of Nations trade statistics. The main study uses 1927–1937; 1938 is retained in the input but excluded from estimation. Exporter and importer declarations are estimated separately. The analysis predicts bilateral allocations conditional on current country margins; its prediction scores are not shares of trade caused by historical relationships.

The code retains the complete set of main-paper and appendix specifications, including sample sensitivity, recorded entry, alternative zero conventions, rolling maps, and regional comparisons. Input definitions and sample exclusions are documented in the codebook.

## Optional checks

```bash
python tests/run_tests.py
```

The checks use the same dependencies as the analysis. To rebuild only the presentation assets from existing results:

```bash
python build_manuscript_assets.py
```

The input checksum and file inventory are in `SHA256SUMS.txt`. The estimation input and the numerical contents of the paper's tables and figures are unchanged from the accompanying submission.

## Citation and rights

Citation metadata is in `CITATION.cff`; code and data rights are specified separately in `LICENSE.md`. The local package identifier is `3.0.3-minimal`, not an ICPSR release number. Cite the version-specific repository record assigned when this package is publicly deposited. The associated study is openICPSR 247132; this package does not assert that an earlier public release contains these files.
