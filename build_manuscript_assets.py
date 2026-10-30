#!/usr/bin/env python3
"""Generate every numerical publication asset from the analysis outputs.

The authoritative manuscript files are in output/manuscript_assets.  The object
map identifies the exact input and result block for each file and each prose
macro.  No result is read from a previously typeset table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

DECLARATIONS = ("Exporter", "Importer")
PUBLICATION_LABELS = {'tables/table1_data.tex': 'Table 1', 'tables/table2_established.tex': 'Table 2', 'tables/table3_germany.tex': 'Table 3', 'appendix_tables/app_data_comparison.tex': 'Table A1', 'appendix_tables/app_sample.tex': 'Table A2', 'appendix_tables/app_primary.tex': 'Table A3', 'appendix_tables/app_newly_positive.tex': 'Table A4', 'appendix_tables/app_geography_coefficients.tex': 'Table A5', 'appendix_tables/app_geography.tex': 'Table A6', 'appendix_tables/app_rolling_one.tex': 'Table A7', 'appendix_tables/app_rolling_three.tex': 'Table A8', 'appendix_tables/app_matched.tex': 'Table A9', 'appendix_tables/app_robustness.tex': 'Table A10', 'appendix_tables/app_smoothing_export.tex': 'Table A11', 'appendix_tables/app_smoothing_import.tex': 'Table A12', 'appendix_tables/app_reassignment.tex': 'Table A13', 'appendix_tables/app_attenuation.tex': 'Table A14', 'appendix_tables/app_breaks.tex': 'Table A15', 'appendix_tables/app_full_unbalanced.tex': 'Table A16', 'appendix_tables/app_lag.tex': 'Table A17', 'appendix_tables/app_germany_export.tex': 'Table A18', 'appendix_tables/app_germany_import.tex': 'Table A19', 'appendix_tables/app_germany_diagnostics.tex': 'Table A20', 'appendix_tables/app_hubs.tex': 'Table A21', 'appendix_tables/app_external_export.tex': 'Table A22', 'appendix_tables/app_external_import.tex': 'Table A23', 'figures/main_1_established_map.pdf': 'Figure 1', 'figures/main_2_geography.pdf': 'Figure 2', 'figures/main_3_fixed_horizon.pdf': 'Figure 3', 'figures/main_4_germany.pdf': 'Figure 4', 'figures/main_5_external.pdf': 'Figure 5'}
COLORS = {"Exporter": "#173E70", "Importer": "#B5582C"}
MARKERS = {"Exporter": "o", "Importer": "s"}


def esc(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def number(value, digits=2, scale=1):
    if value is None or value == "":
        return r"\textemdash"
    value = float(value) * scale
    if not math.isfinite(value):
        raise ValueError(f"Non-finite publication value: {value}")
    return f"{value:.{digits}f}"


def pct(value, digits=1):
    return number(value, digits, 100)


def integer(value):
    return f"{int(value):,}"


def interval(low, high, scale=100, digits=1):
    return f"[{number(low,digits,scale)}, {number(high,digits,scale)}]"


def pvalue(value):
    return r"$<0.001$" if float(value) < .001 else number(value, 3)


def cell_se(row, estimate="estimate", se="se"):
    return f"{number(row[estimate],3)} ({number(row[se],3)})"


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty object map: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_assets(root: Path) -> None:
    root = Path(root).resolve()
    summary_path = root / "output/analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    revised = summary["revised"]
    import sys
    sys.path.insert(0, str(root / "src"))
    from trade_data import load_panel, audit_panel, build_full_internal
    panel = load_panel(root / "data/raw/panel_data.csv")
    audit = audit_panel(panel)
    out = root / "output/manuscript_assets"
    if out.exists():
        shutil.rmtree(out)
    for folder in ("tables", "appendix_tables", "figures"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    objects = []
    macro_values = {}

    def register(path, kind, source, locator, use):
        objects.append({
            "object": path.name, "publication_label": PUBLICATION_LABELS.get(str(path.relative_to(out)), "Numerical text"), "kind": kind,
            "generated_file": str(path.relative_to(out)),
            "authoritative_source": source,
            "result_locator": locator,
            "manuscript_use": use,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    def write_asset(relative, content, kind, source, locator, use):
        path = out / relative
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        register(path, kind, source, locator, use)
        return path

    def macro(key, value, locator):
        if key in macro_values:
            raise ValueError(f"Duplicate macro: {key}")
        macro_values[key] = (value, locator)

    def publication_source(relative):
        league = r"Author's calculations from League of Nations trade statistics \parencite{league1930,league1936,league1938}."
        if relative.endswith("app_data_comparison.tex"):
            return (r"Author's compilation from \textcite{gowa2013}, \textcite{dedinger2017}, "
                    r"\textcite{federico2019}, and League of Nations trade statistics "
                    r"\parencite{league1930,league1936,league1938}.")
        if any(x in relative for x in ("app_geography", "app_reassignment")):
            return league + " Geography inputs are defined in Appendix A3.1."
        return league

    def table(relative, caption, header, rows, columns, note, source, locator,
              long=False, landscape=False, font=r"\small", tabularx=False,
              label=None, extra_head=None, placement="!htbp"):
        # Every current appendix table fits on one page; keep notes and source attached.
        if long and len(rows) <= 40:
            long = False
            placement = "!htbp"
        label = label or "tab:" + Path(relative).stem
        ncols = len(rows[0]) if rows else len(header)
        if any(len(row) != ncols for row in rows):
            raise ValueError(f"Inconsistent columns: {relative}")
        heading = " & ".join(header) + r" \\"
        if extra_head:
            heading = extra_head + "\n" + heading
        body = "\n".join(" & ".join(map(str,row)) + r" \\" for row in rows)
        beginning = r"\clearpage\begin{landscape}" + "\n" if landscape else ""
        ending = "\n" + r"\end{landscape}\clearpage" if landscape else ""
        if long:
            content = beginning + r"{"+font+r"\setlength{\tabcolsep}{4pt}" + "\n"
            content += r"\begin{longtable}{"+columns+"}\n"
            content += r"\caption{"+caption+r"}\label{"+label+r"}\\"+"\n"
            content += r"\toprule"+"\n"+heading+"\n"+r"\midrule\endfirsthead"+"\n"
            content += r"\multicolumn{"+str(ncols)+r"}{l}{\textit{Table \thetable\ continued}}\\"+"\n"
            content += r"\toprule"+"\n"+heading+"\n"+r"\midrule\endhead"+"\n"
            content += r"\midrule\multicolumn{"+str(ncols)+r"}{r}{\footnotesize Continued on next page}\\\endfoot"+"\n"
            content += r"\bottomrule\endlastfoot"+"\n"+body+"\n"+r"\end{longtable}"+"\n"
            content += r"{\footnotesize\noindent\textit{Notes:} "+note+r"\par\smallskip\noindent\textit{Source:} "+publication_source(relative)+r"\par}"+"\n}"+ending
        else:
            content = beginning+r"\begin{table}["+placement+r"]\centering"+"\n"
            content += r"\caption{"+caption+r"}\label{"+label+"}\n"
            content += r"\begin{threeparttable}"+font+r"\setlength{\tabcolsep}{4pt}"+"\n"
            if tabularx:
                content += r"\begin{tabularx}{\textwidth}{"+columns+"}\n"
            else:
                content += r"\begin{tabular}{"+columns+"}\n"
            content += r"\toprule"+"\n"+heading+"\n"+r"\midrule"+"\n"+body+"\n"+r"\bottomrule"+"\n"
            content += r"\end{tabularx}" if tabularx else r"\end{tabular}"
            content += "\n"+r"\begin{tablenotes}[flushleft]\footnotesize\item \textit{Notes:} "+note+"\n"
            content += r"\item \textit{Source:} "+publication_source(relative)+"\n"
            content += r"\end{tablenotes}\end{threeparttable}\end{table}"+ending
        return write_asset(relative, content, "table", source, locator,
                           "Online appendix" if relative.startswith("appendix") else "Main manuscript")

    def get(block, declaration, year=1937):
        return next(r for r in revised[block]
                    if r["declaration"] == declaration and r["year"] == year)

    def legacy_event(declaration, direction):
        return next(r for r in summary["germany"][declaration+"_event"]
                    if r["direction"] == direction and r["year"] == 1937)

    for decl, short, legacy in (("Exporter","x","export"),("Importer","m","import")):
        for year in (1930,1937):
            macro(f"score_{short}_{year}", pct(get("established_prediction",decl,year)["score"]),
                  f"revised.established_prediction[declaration={decl},year={year}].score")
        for name, field, digits, block in [
            ("tv","tv_world_share",1,"established_prediction"),
            ("new_"+short+"_world","mean_newly_positive_world_share",3,"newly_positive_cells"),
            ("new_"+short+"_internal","mean_newly_positive_fraction_of_internal",3,"newly_positive_cells"),
            ("geo","history_increment_over_geography",1,"established_geography"),
        ]:
            key = f"{name}_{short}_1937" if name in ("tv","geo") else f"{name}_1937"
            macro(key,pct(get(block,decl)[field],digits),f"revised.{block}[declaration={decl},year=1937].{field}")
        atten = next(r for r in summary["attenuation"][legacy] if r["year"]==1937)
        macro(f"attenuation_{short}_1937", "$"+number(atten["estimate"],3)+"$",
              f"attenuation.{legacy}[year=1937].estimate")
        for direction, ds in (("Germany exports to Balkans","out"),("Balkan exports to Germany","in")):
            event=legacy_event(legacy,direction)
            macro(f"germany_{short}_{ds}_1937",number(event["estimate"],3),
                  f"germany.{legacy}_event[direction={direction},year=1937].estimate")
            diag=next(r for r in summary["germany"][legacy+"_diagnostics"] if r["direction"]==direction)
            macro(f"pretrend_{short}_{ds}",pvalue(diag["pretrend_p"]),
                  f"germany.{legacy}_diagnostics[direction={direction}].pretrend_p")
        for country in ("France","Germany"):
            for year in (1928,1937):
                e=next(r for r in summary["external_shares"][legacy] if r["reporter"]==country and r["year"]==year)
                macro(f"external_{country.lower()}_{short}_{year}",pct(e["external_share"]),
                      f"external_shares.{legacy}[reporter={country},year={year}].external_share")
    matched=next(r for r in revised["matched_declaration_differences"] if r["year"]==1937)
    macro("matched_difference_1937",pct(matched["difference_exporter_minus_importer"]),
          "revised.matched_declaration_differences[year=1937].difference_exporter_minus_importer")
    macro("matched_interval_1937",r"\("+interval(matched["jk_interval_low"],matched["jk_interval_high"])+r"\)",
          "revised.matched_declaration_differences[year=1937].jk_interval_low,jk_interval_high")
    macro_text="% Generated by build_manuscript_assets.py; do not edit numerical values.\n"
    for key,(value,locator) in sorted(macro_values.items()):
        macro_text += r"\expandafter\def\csname result@"+key+r"\endcsname{"+value+"}\n"
    macros_path=write_asset("results_macros.tex",macro_text,"prose macros",
                           "output/analysis_summary.json","See NUMERICAL_PROSE_MAP.csv","Main manuscript and appendix")
    write_csv(out/"NUMERICAL_PROSE_MAP.csv",[
        {"macro":key,"typeset_value":value,"authoritative_source":"output/analysis_summary.json","result_locator":locator}
        for key,(value,locator) in sorted(macro_values.items())
    ])

    ex=get("established_prediction","Exporter")
    im=get("established_prediction","Importer")
    abstract=(
        "Research on interwar European trade emphasizes the persistence of commercial relationships despite the Great Depression. We distinguish continuity in trading partners from stability in their relative importance. Using League of Nations export and import declarations for 1927--1937, we compare gravity-based predictions from pre-crisis and annually updated trade maps, conditional on current country margins. The pre-crisis map loses substantial predictive power but remains informative beyond distance and borders. Recent maps predict much better. Reallocation occurs overwhelmingly within established relationships, with Germany gaining regional weight. Strong annual continuity thus coexisted with cumulative reordering: Europe's trade hierarchy changed without wholesale replacement of partners."
    )
    abstract_count=len(abstract.split())
    if abstract_count != 100:
        raise ValueError(f"Abstract must have 100 whitespace-delimited words, found {abstract_count}")
    write_asset("abstract_text.tex",abstract,"abstract","output/analysis_summary.json",
                "revised.established_prediction; revised.fixed_horizon_comparison; revised.established_geography; revised.newly_positive_cells; germany; literature framing in manuscript","Main abstract (100 words)")

    rolling = {(r["declaration"],r["training_window_years"]):r
               for r in revised["fixed_horizon_comparison"] if r["year"]==1937}
    horizon=(
        f"In 1937, the one-year rolling score is {pct(rolling['Exporter',1]['rolling_score'])}\\% "
        f"in exporter declarations and {pct(rolling['Importer',1]['rolling_score'])}\\% in importer declarations; "
        f"the same-support anchored scores are {pct(rolling['Exporter',1]['anchored_score_same_support'])}\\% "
        f"and {pct(rolling['Importer',1]['anchored_score_same_support'])}\\%. "
        f"The three-year rolling scores are {pct(rolling['Exporter',3]['rolling_score'])}\\% and "
        f"{pct(rolling['Importer',3]['rolling_score'])}\\%, compared with respective same-support anchors "
        f"of {pct(rolling['Exporter',3]['anchored_score_same_support'])}\\% and "
        f"{pct(rolling['Importer',3]['anchored_score_same_support'])}\\%."
    )
    write_asset("horizon_results_text.tex",horizon,"numerical prose","output/analysis_summary.json",
                "revised.fixed_horizon_comparison[year=1937]","Results: fixed horizon")
    robust={(r["declaration"],r["specification"]):r for r in revised["established_sample_robustness"]}
    spec_g="Exclude every Germany-related dyad"
    spec_s="Exclude every Spain-related dyad"
    spec_i="Nineteen-country internal established dyads"
    spec_f="Full twenty-one-country unbalanced internal established dyads"
    robust_text=(
        f"Removing every Germany-related dyad leaves 1937 historical scores of "
        f"{pct(robust['Exporter',spec_g]['score'])}\\% and {pct(robust['Importer',spec_g]['score'])}\\%; "
        f"removing every Spain-related dyad gives {pct(robust['Exporter',spec_s]['score'])}\\% and "
        f"{pct(robust['Importer',spec_s]['score'])}\\%. "
        "The wider reordering therefore survives both exclusions. "
        f"On established internal links alone, the nineteen-country scores are "
        f"{pct(robust['Exporter',spec_i]['score'])}\\% and {pct(robust['Importer',spec_i]['score'])}\\%. "
        f"The observed twenty-one-country internal panel gives {pct(robust['Exporter',spec_f]['score'])}\\% "
        f"and {pct(robust['Importer',spec_f]['score'])}\\%, reversing the declaration ranking. "
        "That incomplete internal panel measures a different portfolio boundary, but it reinforces the case "
        "for treating the exporter--importer ordering as sample-dependent. Alternative pre-period training "
        "windows retain positive historical predictive content; their exact support and comparable anchored "
        "scores are reported in the appendix."
    )
    write_asset("robustness_results_text.tex",robust_text,"numerical prose","output/analysis_summary.json",
                "revised.established_sample_robustness","Results: sample sensitivity")

    # Main tables.
    data_rows=[
        ["Source grid",f"{integer(audit['n_rows'])} rows; {len(audit['countries'])} reporters; 1927--1938. "
          "The study uses 1927--1937."],
        ["Measurement","Exporter and importer declarations estimated separately in reporter world-share units."],
        ["Trade concept",r"Special trade where available; Norway uses general trade."],
        ["Core boundary",r"Nineteen named countries plus a fixed \External{} residual, containing Bulgaria, Yugoslavia and every other non-core country."],
        ["Missing and zero",f"{integer(audit['missing_export'])} missing export and {integer(audit['missing_import'])} missing import entries in the full source grid. Blanks remain missing; recorded zeros remain zero."],
        ["Established support",f"{revised['methods']['Exporter']['established_pairs_including_external']} positive-pre-period cells per declaration, including {revised['methods']['Exporter']['established_internal_pairs']} internal cells; later zeros retained."],
        ["Closure exclusion",r"Spain's 1937 export account sums to 116.4\%; its entire export reporter-year is omitted. The import account is retained."],
    ]
    table("tables/table1_data.tex","Data architecture and analysis boundaries",
          ["Component","Construction"],data_rows,r"@{}p{.25\textwidth}X@{}",
          r"The core's \External{} boundary is fixed across years. Established support requires a positive recorded total in 1927--1929; selected outcomes retain their original world-share denominators.",
          "data/raw/panel_data.csv; output/analysis_summary.json",
          "source counts; revised.methods",tabularx=True,label="tab:data")

    rows=[]
    for year in (1930,1932,1934,1936,1937):
        e=get("established_prediction","Exporter",year);i=get("established_prediction","Importer",year)
        rows.append([year,pct(e["score"]),interval(e["delete_country_min"],e["delete_country_max"]),
                     pct(e["tv_world_share"]),pct(i["score"]),
                     interval(i["delete_country_min"],i["delete_country_max"]),pct(i["tv_world_share"])])
    table("tables/table2_established.tex","Predictive survival of the pre-crisis hierarchy",
          ["Year",r"$P_t$ (\%)","Deletion range","TV (pp)",r"$P_t$ (\%)","Deletion range","TV (pp)"],
          rows,r"@{}lrrrrrr@{}",
          r"$P_t=1-D_H/D_M$ on established support including \External{}. Deletion ranges are actual minima and maxima after "
          r"removing each named country and every incident dyad, holding \External{} fixed. They are sensitivity ranges, not "
          r"confidence intervals. TV is mean half absolute allocation error in world-portfolio percentage points. "
          r"Spain's export reporter-year is omitted in 1937.",
          "output/analysis_summary.json","revised.established_prediction",label="tab:established",
          font=r"\small",extra_head=r"& \multicolumn{3}{c}{Exporter declarations} & \multicolumn{3}{c}{Importer declarations}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    # Compact main Germany table, with clear separation between hub association and trend diagnostics.
    hubs={(r["declaration"],r["hub"],r["direction"]):r
          for leg in ("export","import") for r in summary["germany"][leg+"_hubs"]}
    hubrows=[]
    for hub in ("Germany","United Kingdom","France","Italy"):
        hubrows.append([hub]+[cell_se(hubs[d,hub,direction])
                       for d in DECLARATIONS for direction in ("Hub exports to Balkans","Balkan exports to hub")])
    diagrows=[]
    for leg,d in (("export","Exporter"),("import","Importer")):
        for r in summary["germany"][leg+"_diagnostics"]:
            direct=r"Germany $\to$ region" if r["direction"].startswith("Germany") else r"Region $\to$ Germany"
            diagrows.append([d,direct,pvalue(r["pretrend_p"]),
                            cell_se(r,"post1934_level_shift","post1934_level_shift_se")])
    gtext=r"""\begin{table}[!htbp]\centering
\caption{Regional hub associations and Germany's timing diagnostics}\label{tab:germany}
\begin{threeparttable}\small\setlength{\tabcolsep}{4pt}
\textit{Panel A. Joint post-1934 hub interactions}\\[4pt]
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{Exporter declarations} & \multicolumn{2}{c}{Importer declarations}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Hub & Hub $\to$ region & Region $\to$ hub & Hub $\to$ region & Region $\to$ hub\\
\midrule
"""
    gtext+="\n".join(" & ".join(r)+r" \\" for r in hubrows)
    gtext+=r"""
\bottomrule\end{tabular}
\par\medskip\textit{Panel B. Germany: earlier movements and trend-adjusted level shifts}\\[4pt]
\begin{tabular}{@{}llrr@{}}
\toprule
Declaration & Direction & Pre-period $p$ & Post-1934 level shift\\\midrule
"""
    gtext+="\n".join(" & ".join(r)+r" \\" for r in diagrows)
    gtext+=r"""
\bottomrule\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Notes:} Coefficients with directed-pair clustered standard errors in parentheses. All models include directed-pair,
origin-year and destination-year effects on the observed twenty-one-country internal panel, 1928--1937.
Panel A estimates all eight hub--direction interactions jointly. Panel B's pre-period test jointly evaluates
1928--1932 event coefficients against 1933; its level shift is from a separate group-specific linear-trend plus
post-1934 indicator model. The region is Bulgaria, Greece, Hungary, Romania and Yugoslavia. These are relative
associations; conventional clustered inference does not establish a policy effect with five regional dyads per direction.
\item \textit{Source:} Author's calculations from League of Nations trade statistics \parencite{league1930,league1936,league1938}.
\end{tablenotes}\end{threeparttable}\end{table}
"""
    write_asset("tables/table3_germany.tex",gtext,"table","output/analysis_summary.json",
                "germany.*_hubs; germany.*_diagnostics","Main manuscript: Germany")

    # All publication figures are regenerated directly from final result arrays.
    plt.rcParams.update({
        "font.family":"DejaVu Serif","font.size":10,"axes.titlesize":11,"axes.labelsize":10,
        "legend.fontsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#555555","axes.linewidth":.7,
        "grid.color":"#D6D8DB","grid.linewidth":.55,"pdf.fonttype":42,"ps.fonttype":42,
        "savefig.facecolor":"white","axes.facecolor":"white",
    })

    def style(ax, years=range(1930,1938), percent=False):
        ax.grid(axis="y",alpha=.8)
        ax.set_axisbelow(True)
        ax.set_xticks(list(years))
        ax.tick_params(length=3)
        ax.margins(x=.035)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    def savefigure(fig,name,locator):
        fig.savefig(out/"figures"/(name+".pdf"),bbox_inches="tight",
                    metadata={"Title":name.replace("_"," "),"Author":"","CreationDate":None,"ModDate":None})
        register(out/"figures"/(name+".pdf"),"figure","output/analysis_summary.json",locator,"Main manuscript")
        plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(7.2,6.2),gridspec_kw={"height_ratios":[2.1,1]},layout="constrained")
    for decl in DECLARATIONS:
        rows=[r for r in revised["established_prediction"] if r["declaration"]==decl]
        years=[r["year"] for r in rows]
        axes[0].plot(years,[100*r["score"] for r in rows],color=COLORS[decl],marker=MARKERS[decl],lw=1.8,ms=4,label=decl)
        axes[0].fill_between(years,[100*r["delete_country_min"] for r in rows],
                             [100*r["delete_country_max"] for r in rows],color=COLORS[decl],alpha=.12,lw=0)
        entry=[r for r in revised["newly_positive_cells"] if r["declaration"]==decl]
        axes[1].plot(years,[100*r["mean_newly_positive_world_share"] for r in entry],
                     color=COLORS[decl],marker=MARKERS[decl],lw=1.8,ms=4)
    axes[0].set_title("A. Predictive fit of the 1927–1929 hierarchy",loc="left",pad=10)
    axes[0].set_ylabel("Deviance reduction (%)");axes[0].set_ylim(0,100)
    axes[0].legend(frameon=False,ncol=2,loc="lower left")
    axes[1].set_title("B. Portfolio mass in newly positive internal cells",loc="left",pad=10)
    axes[1].set_ylabel("World portfolio (%)");axes[1].set_ylim(bottom=0)
    for ax in axes: style(ax)
    savefigure(fig,"main_1_established_map","revised.established_prediction; revised.newly_positive_cells")

    fig,ax=plt.subplots(figsize=(7.2,4.5),layout="constrained")
    for decl in DECLARATIONS:
        rows=[r for r in revised["established_geography"] if r["declaration"]==decl]
        years=[r["year"] for r in rows]
        ax.plot(years,[100*r["history_increment_over_geography"] for r in rows],
                color=COLORS[decl],marker=MARKERS[decl],lw=1.9,ms=4,label=decl)
        ax.fill_between(years,[100*r["delete_country_min"] for r in rows],
                        [100*r["delete_country_max"] for r in rows],color=COLORS[decl],alpha=.12,lw=0)
    style(ax);ax.axhline(0,color="#777777",lw=.7)
    ax.set_ylabel("Historical improvement over geography (%)")
    ax.set_ylim(0,100);ax.legend(frameon=False,ncol=2,loc="upper right")
    savefigure(fig,"main_2_geography","revised.established_geography")

    fig,axes=plt.subplots(1,2,figsize=(7.4,4.5),sharey=True,layout="constrained")
    horizon_colors={1:"#173E70",3:"#B5582C"}
    for ax,decl in zip(axes,DECLARATIONS):
        for horizon in (1,3):
            rows=[r for r in revised["fixed_horizon_comparison"]
                  if r["declaration"]==decl and r["training_window_years"]==horizon]
            ax.plot([r["year"] for r in rows],[100*r["rolling_score"] for r in rows],
                    color=horizon_colors[horizon],lw=1.7,marker="o" if horizon==1 else "s",ms=3,
                    label=f"{horizon}-year rolling")
            anchor=[r for r in rows if "anchored_score_same_support" in r]
            ax.plot([r["year"] for r in anchor],[100*r["anchored_score_same_support"] for r in anchor],
                    color=horizon_colors[horizon],lw=1.2,ls="--",alpha=.85,
                    label=f"Anchor: {horizon}-year support")
        ax.set_title(decl+" declarations",loc="left")
        style(ax,range(1928,1938,2));ax.set_ylim(0,101)
        ax.axvline(1930,color="#AAAAAA",lw=.6,ls=":")
        ax.set_xlabel("Evaluation year")
    axes[0].set_ylabel("Deviance reduction (%)")
    axes[1].legend(frameon=False,loc="lower left",fontsize=10)
    savefigure(fig,"main_3_fixed_horizon","revised.fixed_horizon_comparison")

    fig,axes=plt.subplots(1,2,figsize=(7.4,4.5),sharey=True,layout="constrained")
    directions=("Germany exports to Balkans","Balkan exports to Germany")
    for ax,direction,title in zip(axes,directions,("Germany → region","Region → Germany")):
        for decl,leg,offset in (("Exporter","export",-.07),("Importer","import",.07)):
            rows=[r for r in summary["germany"][leg+"_event"] if r["direction"]==direction]
            rows=sorted(rows+[{"year":1933,"estimate":0.,"se":0.}],key=lambda r:r["year"])
            x=np.array([r["year"] for r in rows],float)
            y=np.array([r["estimate"] for r in rows])
            se=np.array([r["se"] for r in rows])
            ax.errorbar(x+offset,y,yerr=1.96*se,color=COLORS[decl],marker=MARKERS[decl],
                        lw=1.4,elinewidth=.7,capsize=2,ms=3,label=decl)
        style(ax,range(1928,1938,2));ax.axhline(0,color="#777777",lw=.75)
        ax.axvline(1933,color="#AAAAAA",lw=.6,ls=":")
        ax.set_title(title,loc="left");ax.set_xlabel("Year")
    axes[0].set_ylabel("Relative interaction (log points; 1933 = 0)")
    axes[1].legend(frameon=False,loc="upper left")
    savefigure(fig,"main_4_germany","germany.export_event; germany.import_event")

    fig,axes=plt.subplots(1,2,figsize=(7.4,4.5),sharey=True,layout="constrained")
    countries=("United Kingdom","France","Germany","Czechoslovakia")
    cc={"United Kingdom":"#173E70","France":"#B5582C","Germany":"#347566","Czechoslovakia":"#765585"}
    for ax,leg,decl in zip(axes,("export","import"),DECLARATIONS):
        for country in countries:
            rows=[r for r in summary["external_shares"][leg] if r["reporter"]==country]
            ax.plot([r["year"] for r in rows],[100*r["external_share"] for r in rows],
                    color=cc[country],lw=1.7,marker=None,label=country)
        style(ax,range(1927,1938,2));ax.set_title(decl+" declarations",loc="left")
        ax.set_xlabel("Year");ax.set_ylim(0,85)
    axes[0].set_ylabel("Outside-core share of world portfolio (%)")
    axes[1].legend(frameon=False,loc="lower left",fontsize=10)
    savefigure(fig,"main_5_external","external_shares.export; external_shares.import")


    # Online appendix: complete environments with captions, notes and repeated headers.
    table("appendix_tables/app_data_comparison.tex",
          "Data resources and the purpose of the present reconstruction",
          ["Resource","Coverage and organization","Implication for this paper"],[
            [r"Gowa--Hicks \parencite{gowa2013}",
             "A broader interwar country panel used to study politics, institutions and trade.",
             "Provides the broader institutional comparison. The present design retains the two customs declarations as parallel reporter portfolios."],
            [r"RICardo \parencite{dedinger2017}",
             "Long-run bilateral trade records, with reporter, partner, flow direction and source-related metadata across a much wider archive.",
             "Provides broader historical coverage. The contribution here is a fixed core boundary and an explicit, common portfolio-construction protocol."],
            [r"Federico--Tena \parencite{federico2019}",
             "Long-run aggregate trade series and a synthesis of world trade through 1938.",
             "Provides aggregate context. This paper addresses the distribution across partners conditional on current margins."],
            ["Present reconstruction",
             f"{integer(audit['n_rows'])} reporter--partner--year rows, 21 countries, 1927--1938; 1927--1937 estimation; separate export/import declarations and world totals.",
             "Supports closed core-plus-external portfolios, recorded-support accounting and comparable conditional predictions. It omits product and firm detail and separates no destinations within the external residual."],
          ],r"@{}p{.23\textwidth}p{.34\textwidth}X@{}",
          "The comparison concerns coverage and measurement design. Sources can overlap where datasets draw on the same League publications. "
          "The present reconstruction combines separate customs declarations with reporter world totals and a fixed European core.",
          "data/raw/panel_data.csv; docs/references.bib","coverage counts and cited resource descriptions",
          tabularx=True)

    samples=[]
    for decl,leg in (("Exporter","export"),("Importer","import")):
        methods=revised["methods"][decl]
        primary=[r for r in revised["established_prediction"] if r["declaration"]==decl]
        geo=[r for r in revised["established_geography"] if r["declaration"]==decl]
        full=[r.__dict__ for r in build_full_internal(panel,leg)]
        samples.extend([
            [decl,"Core plus external: training, 1927--1929",integer(methods["established_pairs_including_external"]),integer(3*methods["established_pairs_including_external"])],
            [decl,"Core plus external: evaluation, 1930--1937",r"\textemdash",integer(sum(r["n_observations"] for r in primary))],
            [decl,"Internal geography support: training",integer(methods["established_internal_pairs"]),integer(3*methods["established_internal_pairs"])],
            [decl,"Internal geography support: evaluation",r"\textemdash",integer(sum(r["n_observations"] for r in geo))],
            [decl,"Observed full internal panel, 1927--1937",r"\textemdash",integer(sum(1927<=int(r["year"])<=1937 for r in full))],
        ])
    table("appendix_tables/app_sample.tex","Maintained analysis samples",
          ["Declaration","Sample","Training cells","Observations"],samples,
          r"@{}lp{.45\textwidth}rr@{}",
          r"Training cells are distinct directed cells with a positive total during 1927--1929; every available training year, including recorded zeros, enters. "
          r"Evaluation counts include later zeros and omit Spain's non-closing 1937 export reporter-year. Full-panel counts describe available internal records before model-specific support restrictions. "
          r"The paired declaration comparison uses 294 common internal dyads among eighteen countries, excluding Spain and \External{}.",
          "output/analysis_summary.json; data/raw/panel_data.csv",
          "revised.methods; revised.established_prediction; revised.established_geography; observed full panel",
          long=True)

    rows=[]
    for r in revised["established_prediction"]:
        rows.append([r["declaration"][:3],r["year"],pct(r["score"]),
                     interval(r["delete_country_min"],r["delete_country_max"]),
                     interval(r["jk_interval_low"],r["jk_interval_high"]),
                     pct(r["tv_world_share"]),pct(r["tv_conditional_on_support"]),integer(r["n_observations"])])
    table("appendix_tables/app_primary.tex","Established-link prediction: annual results and country sensitivity",
          ["Dec.","Year",r"$P_t$ (\%)",r"\shortstack{Deletion\\range}",r"\shortstack{Jackknife\\interval}",
           r"\shortstack{World\\TV (pp)}",r"\shortstack{Supported\\TV (pp)}","Cells"],rows,
          r"@{}llrrrrrr@{}",
          r"Exp and Imp denote exporter and importer declarations. The support is fixed by positive total recorded trade in 1927--1929, including \External{}. "
          r"Entries are $100P_t=100(1-D_H/D_M)$. Deletion ranges show actual minima and maxima after all-incident-dyad country removal. "
          r"Jackknife intervals are full-sample estimate $\pm1.96$ times the conventional delete-country jackknife standard error; they are sensitivity summaries, can exceed 100\%, and are not clipped. "
          r"World TV uses original world-share units; supported TV divides each reporter's half absolute error by that reporter's supported observed mass.",
          "output/analysis_summary.json","revised.established_prediction",long=True,font=r"\footnotesize")

    rows=[]
    for r in revised["newly_positive_cells"]:
        rows.append([r["declaration"][:3],r["year"],integer(r["newly_positive_recorded_cells"]),
                     pct(r["mean_newly_positive_world_share"],3),
                     pct(r["mean_newly_positive_fraction_of_internal"],3),
                     integer(r["now_zero_established_cells"]),
                     pct(r["mean_pre_share_of_now_zero_cells"],3),integer(r["n_reporters"])])
    table("appendix_tables/app_newly_positive.tex","Recorded entry, exit and their portfolio mass",
          ["Dec.","Year",r"\shortstack{Newly positive\\cells}",r"\shortstack{World\\share (\%)}",
           r"\shortstack{Internal\\share (\%)}",r"\shortstack{Now-zero\\established}",r"\shortstack{Their pre-period\\world share (\%)}","Reporters"],
          rows,r"@{}llrrrrrr@{}",
          r"Newly positive cells were recorded as zero throughout 1927--1929. World and internal shares are means of reporter-level shares. "
          r"Now-zero established cells had a positive pre-period total; the next column gives their mean pre-period world-share mass per current reporter. "
          r"These are recorded-country-cell measures, not firm entry or exit. The initial core has 331 established internal cells out of 342 possible cells per declaration; \External{} is already positive for every reporter.",
          "output/analysis_summary.json","revised.newly_positive_cells",long=True,font=r"\footnotesize")

    rows=[]
    for r in revised["established_geography"]:
        rows.append([r["declaration"][:3],r["year"],pct(r["score"]),pct(r["geography_score"]),
                     pct(r["history_increment_over_geography"]),
                     interval(r["delete_country_min"],r["delete_country_max"]),
                     interval(r["jk_interval_low"],r["jk_interval_high"]),integer(r["n_observations"])])
    table("appendix_tables/app_geography.tex","Historical and geographic predictions on established internal links",
          ["Dec.","Year",r"\shortstack{History\\$P_t$ (\%)}",r"\shortstack{Geography\\score (\%)}",
           r"$G_t$ (\%)",r"\shortstack{$G_t$ deletion\\range}",r"\shortstack{$G_t$ jackknife\\interval}","Cells"],
          rows,r"@{}llrrrrrr@{}",
          r"History and geography scores are each measured relative to the identical current-margins benchmark. "
          r"Entries in the improvement column are $100G_t=100(1-D_H/D_{Geo})$. Geography consists of pre-period coefficients on log capital distance and the interwar border indicator. "
          r"Both predictors refit current margins on the same established internal cells; \External{} is excluded. "
          r"The country-deletion range and jackknife sensitivity interval apply to $G_t$, not the two margins-relative scores.",
          "output/analysis_summary.json","revised.established_geography",long=True,font=r"\footnotesize")

    rows=[]
    for decl in DECLARATIONS:
        m=revised["methods"][decl]
        for k,term in enumerate(("Log capital distance","Interwar common border")):
            rows.append([decl,term,number(m["geography_beta_on_established_internal_support"][k],3),
                         number(m["geography_se_on_established_internal_support"][k],3)])
    table("appendix_tables/app_geography_coefficients.tex","Pre-period geographic coefficients",
          ["Declaration","Covariate","Coefficient","Cluster SE"],rows,r"@{}llrr@{}",
          r"PPML on established internal links during 1927--1929, with origin-year and destination-year effects. "
          r"The coefficients are frozen during later prediction. Standard errors cluster by directed pair and use $G/(G-1)$. "
          r"The sample comprises internal country pairs with positive recorded trade in at least one pre-period year.",
          "output/analysis_summary.json","revised.methods.*.geography_beta_on_established_internal_support; geography_se_on_established_internal_support")

    for length,name in ((1,"one"),(3,"three")):
        rows=[]
        for r in revised["fixed_horizon_comparison"]:
            if r["training_window_years"]!=length:continue
            rows.append([r["declaration"][:3],r["year"],pct(r["rolling_score"]),
                         pct(r.get("anchored_score_same_support")) if "anchored_score_same_support" in r else r"\textemdash",
                         pct(r.get("rolling_minus_anchored_score")) if "rolling_minus_anchored_score" in r else r"\textemdash",
                         pct(r["rolling_tv_world_share"]),pct(r["supported_world_share_mean"],2),integer(r["n_observations"])])
        table(f"appendix_tables/app_rolling_{name}.tex",
              f"Rolling prediction with a {length}-year training window",
              ["Dec.","Year",r"\shortstack{Rolling\\score (\%)}",r"\shortstack{Anchor\\score (\%)}",
               r"\shortstack{Difference\\(pp)}",r"\shortstack{Rolling\\TV (pp)}",
               r"\shortstack{Supported\\mass (\%)}","Cells"],rows,r"@{}llrrrrrr@{}",
              r"Every rolling map predicts the year immediately after its training window. From 1930 onward, rolling and 1927--1929 anchored maps "
              r"are fitted and evaluated on the intersection of their positive training supports, with identical current margins and benchmark observations. "
              r"The displayed difference is rolling minus anchored score. Pre-1930 rows have no out-of-period 1927--1929 anchor. "
              r"Outcomes retain their world-share denominators; supported mass is the mean reporter total on the selected support.",
              "output/analysis_summary.json",f"revised.fixed_horizon_comparison[training_window_years={length}]",
              long=True,font=r"\footnotesize")

    rows=[]
    for r in revised["matched_declaration_differences"]:
        rows.append([r["year"],pct(r["exporter_score"]),pct(r["importer_score"]),
                     pct(r["difference_exporter_minus_importer"]),
                     interval(r["delete_country_min"],r["delete_country_max"]),
                     interval(r["jk_interval_low"],r["jk_interval_high"])])
    table("appendix_tables/app_matched.tex","Paired declaration comparison on common internal support",
          ["Year",r"$P^X_t$ (\%)",r"$P^M_t$ (\%)","Gap (pp)",
           r"\shortstack{Gap deletion\\range (pp)}",r"\shortstack{Gap jackknife\\interval (pp)}"],
          rows,r"@{}lrrrrr@{}",
          r"The common network has eighteen countries and 294 directed established dyads, excluding Spain and \External{}. "
          r"Scores are relative to each declaration's margins-only predictor on identical directed support. "
          r"The gap is $P^X_t-P^M_t$, not the difference between geography increments. Country deletions are paired. "
          r"The original export and import reporter denominators remain different; matching cells does not reconcile customs measurement.",
          "output/analysis_summary.json","revised.matched_declaration_differences",long=True)

    spec_labels={
        "Training 1927-1929":"Training 1927--1929",
        "Training 1927-1928":"Training 1927--1928",
        "Training 1928-1929":"Training 1928--1929",
        "Training 1928 only":"Training 1928 only",
        spec_g:"Delete all Germany-related dyads",
        spec_s:"Delete all Spain-related dyads",
        spec_i:"Nineteen-country internal links",
        spec_f:"Twenty-one-country observed internal links",
    }
    rows=[]
    for r in revised["established_sample_robustness"]:
        rows.append([r["declaration"][:3],spec_labels[r["specification"]],pct(r["score"]),
                     pct(r["anchor_score_same_support"]) if "anchor_score_same_support" in r else r"\textemdash",
                     pct(r["tv_world_share"]),pct(r["supported_world_share_mean"],1),integer(r["n_observations"])])
    table("appendix_tables/app_robustness.tex","Training windows and sample boundaries: 1937",
          ["Dec.","Specification",r"\shortstack{Score\\(\%)}",r"\shortstack{Anchor\\(\%)}",r"\shortstack{TV\\(pp)}",
           r"\shortstack{Supported\\mass (\%)}","Cells"],rows,
          r"@{}lp{.35\textwidth}rrrrr@{}",
          r"Every row uses positive total recorded trade in its stated training window and retains original world-share outcomes. "
          r"The alternative training rows also report the 1927--1929 anchor re-estimated on the same support. "
          r"Country exclusions remove all incident observations; \External{} keeps its original definition. "
          r"Internal-only and unbalanced twenty-one-country results have different observed portfolio boundaries. "
          r"Missing observations are not set to zero or imputed.",
          "output/analysis_summary.json","revised.established_sample_robustness",long=False,font=r"\footnotesize")

    for decl,leg in (("Exporter","export"),("Importer","import")):
        rows=[]
        selected=sorted([r for r in revised["full_matrix_smoothing_sensitivity"] if r["declaration"]==decl],
                        key=lambda r:(r["epsilon"],r["year"]),reverse=False)
        for r in selected:
            exponent=int(round(math.log10(r["epsilon"])))
            rows.append([r["year"],r"$10^{"+str(exponent)+r"}$",pct(r["full_recorded_map_score"]),
                         pct(r["full_recorded_tv_world_share"]),
                         pct(r["internal_history_increment_over_geography"])])
        table(f"appendix_tables/app_smoothing_{leg}.tex",
              f"Full-matrix zero-replacement sensitivity: {decl.lower()} declarations",
              ["Year",r"$\epsilon$",r"\shortstack{Full-matrix\\score (\%)}",
               r"\shortstack{World TV\\(pp)}",r"\shortstack{Internal history gain\\over geography (\%)}"],
              rows,r"@{}llrrr@{}",
              r"The full recorded training matrix is evaluated using four positive training additions, with observed evaluation shares unchanged. "
              r"These are supplementary zero-treatment diagnostics. The main established-support estimates use no $\epsilon$ and report the "
              r"mass on previously zero cells separately. The internal geography comparison excludes \External{} and uses the full recorded internal support.",
              "output/analysis_summary.json",f"revised.full_matrix_smoothing_sensitivity[declaration={decl}]",
              long=True)

    rows=[]
    for r in revised["geography_conditioned_reassignment"]:
        rows.append([r["declaration"],integer(r["draws"]),number(r["observed_deviance"],3),
                     number(r["reassignment_mean"],3),number(r["reassignment_p05"],3),
                     number(r["reassignment_median"],3),number(r["reassignment_p95"],3),
                     integer(r["reassignments_with_deviance_at_most_observed"])])
    table("appendix_tables/app_reassignment.tex","Geography-conditioned residual reassignment in 1937",
          ["Declaration","Draws","Observed","Mean","5th","Median","95th",r"\shortstack{At most\\observed}"],
          rows,r"@{}lrrrrrrr@{}",
          r"Entries summarize Poisson deviance after holding the geography component on the actual dyad, "
          r"reassigning the canonically normalized historical residual within reporters and refitting current margins. "
          r"Percentile columns refer to the descriptive reassignment distribution. The final column is a count, not a randomization $p$-value. "
          r"Residual normalization is invariant to additive origin and destination pair-effect shifts; the original historical prediction is reconstructed before reassignment. "
          r"The experiment does not preserve observed share concentration or represent randomized historical treatment.",
          "output/analysis_summary.json","revised.geography_conditioned_reassignment",font=r"\footnotesize")

    rows=[]
    for year in range(1928,1938):
        row=[year]
        for leg in ("export","import"):
            if year==1929:
                row += ["0.000",r"\textemdash",r"\textemdash"]
            else:
                r=next(r for r in summary["attenuation"][leg] if r["year"]==year)
                row += [number(r["estimate"],3),number(r["se"],3),
                        interval(r["ci_low"],r["ci_high"],1,3)]
        rows.append(row)
    table("appendix_tables/app_attenuation.tex","Annual changes in the relative weight of pre-period affinity",
          ["Year","Coefficient","SE","95\\% interval","Coefficient","SE","95\\% interval"],
          rows,r"@{}lrrrrrr@{}",
          r"The affinity measure is the exact log of the mean 1927--1929 share, standardized across established pairs within reporter. "
          r"All year interactions from 1928--1937 enter except 1929, the sole reference. "
          r"Models include directed-pair, origin-year and destination-year effects; current recorded zeros remain. "
          r"Standard errors cluster by directed pair. These profiles are descriptive and may reflect mean reversion in a pre-period-outcome-based regressor.",
          "output/analysis_summary.json","attenuation.export; attenuation.import",long=False,placement="H",
          extra_head=r"& \multicolumn{3}{c}{Exporter declarations} & \multicolumn{3}{c}{Importer declarations}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    rows=[]
    for r in load_csv(root/"output/tables/break_profiles.csv"):
        rows.append([r["declaration"],r["break_year"],number(r["estimate"],3),number(r["se"],3),number(r["deviance"],4)])
    table("appendix_tables/app_breaks.tex","Descriptive one-step break profiles",
          ["Declaration","Step begins","Coefficient","Cluster SE","Deviance"],
          rows,r"@{}llrrr@{}",
          r"Each candidate interacts standardized exact-log pre-period affinity with an indicator beginning in the stated year, "
          r"on positive-pre-period support over 1928--1937. Pair, origin-year and destination-year effects are included. "
          r"The lowest deviance selects a descriptive fitted step; comparison across candidates is not a test identifying a historical treatment date.",
          "output/tables/break_profiles.csv","all candidate years and declarations",long=False,placement="H")

    for leg,decl in (("export","Exporter"),("import","Importer")):
        rows=[]
        for year in range(1928,1938):
            row=[year]
            for direction in directions:
                if year==1933: row += ["0.000",r"\textemdash",r"\textemdash"]
                else:
                    r=next(r for r in summary["germany"][leg+"_event"]
                           if r["year"]==year and r["direction"]==direction)
                    row += [number(r["estimate"],3),number(r["se"],3),interval(r["ci_low"],r["ci_high"],1,3)]
            rows.append(row)
        table(f"appendix_tables/app_germany_{leg}.tex",
              f"Annual Germany--regional interactions: {decl.lower()} declarations",
              ["Year","Coefficient","SE","95\\% interval","Coefficient","SE","95\\% interval"],rows,
              r"@{}lrrrrrr@{}",
              r"The observed full internal panel, 1928--1937, with directed-pair, origin-year and destination-year effects. "
              r"The sole omitted year is 1933; the normalized reference has no estimated standard error. "
              r"The regional group is Bulgaria, Greece, Hungary, Romania and Yugoslavia. Standard errors cluster by directed pair. "
              r"The intervals are conventional normal approximations and do not establish a clean New Plan policy effect.",
              "output/analysis_summary.json",f"germany.{leg}_event",long=True,
              extra_head=r"& \multicolumn{3}{c}{Germany $\to$ region} & \multicolumn{3}{c}{Region $\to$ Germany}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    rows=[]
    for leg,decl in (("export","Exporter"),("import","Importer")):
        for r in summary["germany"][leg+"_diagnostics"]:
            direct=r"Germany $\to$ region" if r["direction"].startswith("Germany") else r"Region $\to$ Germany"
            rows.append([decl[:3],direct,number(r["pretrend_wald"],3),pvalue(r["pretrend_p"]),
                         cell_se(r,"linear_trend","linear_trend_se"),
                         cell_se(r,"post1934_level_shift","post1934_level_shift_se")])
    table("appendix_tables/app_germany_diagnostics.tex","Germany: all-pre-period tests and a trend-adjusted level shift",
          ["Dec.","Direction",r"Wald $\chi^2_5$",r"$p$",r"\shortstack{Linear trend\\(SE)}",r"\shortstack{Post-1934 shift\\(SE)}"],
          rows,r"@{}llrrrr@{}",
          r"The joint Wald test covers every 1928--1932 interaction against the sole 1933 reference. "
          r"The trend and level-shift estimates come from separate direction-specific specifications including a dyad-group linear trend "
          r"and a post-1934 indicator, alongside all three fixed-effect families. A level indicator does not estimate a post-1934 slope acceleration. "
          r"Failure to reject a pre-period restriction is not evidence that parallel trends have been established.",
          "output/analysis_summary.json","germany.export_diagnostics; germany.import_diagnostics",
          font=r"\footnotesize")

    rows=[]
    for leg,decl in (("export","Exporter"),("import","Importer")):
        for r in summary["germany"][leg+"_hubs"]:
            direct=r"Hub $\to$ region" if r["direction"].startswith("Hub") else r"Region $\to$ hub"
            rows.append([decl[:3],r["hub"],direct,number(r["estimate"],3),number(r["se"],3),
                         interval(r["ci_low"],r["ci_high"],1,3),number(r["percent"],1)])
    table("appendix_tables/app_hubs.tex","Joint post-1934 regional associations for four hubs",
          ["Dec.","Hub","Direction","Coefficient","SE","95\\% interval",r"\shortstack{Implied\\change (\%)}"],
          rows,r"@{}lllr rrr@{}",
          r"Eight post-1934 hub--direction interactions are estimated jointly, with directed-pair, origin-year and destination-year effects "
          r"over 1928--1937. Implied changes are $100[\exp(\widehat\beta)-1]$. "
          r"Alternative hubs provide historically relevant comparisons, not untreated controls. Standard errors cluster by directed pair.",
          "output/analysis_summary.json","germany.export_hubs; germany.import_hubs",long=True,font=r"\footnotesize")

    rows=[]
    for leg,decl in (("export","Exporter"),("import","Importer")):
        r=summary["full_unbalanced"][leg]
        rows.append([decl,number(r["estimate"],3),number(r["se"],3),
                     interval(r["ci_low"],r["ci_high"],1,3),integer(r["nobs"])])
    table("appendix_tables/app_full_unbalanced.tex","Full-panel affinity sensitivity with a fixed 1932 split",
          ["Declaration","Coefficient","SE","95\\% interval",r"$N$"],rows,r"@{}lrrrr@{}",
          r"Observed twenty-one-country internal panel, 1928--1937, restricted to a positive recorded mean in 1927--1929. "
          r"The regressor is standardized exact-log pre-period affinity interacted with an indicator for 1932--1937. "
          r"Models include directed-pair, origin-year and destination-year effects. "
          r"This is a fixed-split sensitivity specification, not an annual 1937 coefficient or a fitted structural break. "
          r"Standard errors cluster by directed pair.",
          "output/analysis_summary.json","full_unbalanced.export; full_unbalanced.import")

    periods={"Full 1928-1937":"1928--1937","Pre-break 1928-1931":"1928--1931",
             "Post-break 1932-1937":"1932--1937","New Plan era 1934-1937":"1934--1937"}
    rows=[]
    for leg,decl in (("export","Exporter"),("import","Importer")):
        for r in summary["sticky"][leg]:
            rows.append([decl[:3],periods[r["period"]],cell_se(r,"gamma","gamma_se"),number(r["alpha"],3),
                         number(r["flexible_mse"],3,1e4),number(r["sticky_mse"],3,1e4),
                         number(r["blended_mse"],3,1e4),integer(r["nobs"]),integer(r["excluded_zero_lag_observations"])])
    table("appendix_tables/app_lag.tex","Positive-lag persistence and the descriptive fitted-prediction blend",
          ["Dec.","Period",r"\shortstack{Lag elasticity\\(cluster SE)}",r"$\alpha$",
           r"\shortstack{Flexible MSE\\$\times10^4$}",r"\shortstack{Lag MSE\\$\times10^4$}",
           r"\shortstack{Blend MSE\\$\times10^4$}",r"$N$",r"\shortstack{Excluded\\zero lags}"],
          rows,r"@{}llrrrrrrr@{}",
          r"The lag coefficient is estimated on the exact log of a strictly positive recorded lagged share, with origin-year and destination-year effects. "
          r"Current zeros remain. The flexible prediction uses directed-pair and both country-year effects on the same sample. "
          r"The bounded $\alpha$ minimizes in-sample squared error of $\alpha\widehat x^{lag}+(1-\alpha)\widehat x^{flex}$. "
          r"It is a descriptive fit weight rather than a structural share of sticky contracts. All displayed blend weights have positive identification denominators. "
          r"Period labels are calendar intervals and do not assert identified break dates.",
          "output/analysis_summary.json","sticky.export; sticky.import",long=False,landscape=False,font=r"\small")

    for leg,decl in (("export","Exporter"),("import","Importer")):
        ext=summary["external_shares"][leg]
        rows=[]
        for country in sorted({r["reporter"] for r in ext}):
            lookup={r["year"]:r["external_share"] for r in ext if r["reporter"]==country}
            rows.append([country]+[pct(lookup[y],1) if y in lookup else r"\textemdash" for y in range(1927,1938)])
        table(f"appendix_tables/app_external_{leg}.tex",
              f"Outside-core world-portfolio shares: {decl.lower()} declarations",
              ["Reporter"]+list(map(str,range(1927,1938))),rows,r"@{}l"+"r"*11+r"@{}",
              r"Percent of each reporter's world exports or imports assigned to the fixed \External{} residual. "
              r"It contains Bulgaria, Yugoslavia and all other countries outside the nineteen-country core. "
              r"Spain's 1937 export entry is omitted because the internal account exceeds its world total; its import entry remains valid. "
              r"The aggregate does not identify the distribution among outside partners.",
              "output/analysis_summary.json",f"external_shares.{leg}",long=True,landscape=True,font=r"\small")

    # Explicit object inventory and deterministic build record.
    write_csv(out/"MANUSCRIPT_OBJECT_MAP.csv",objects)
    build_record={
        "abstract_word_count":abstract_count,
        "numerical_macros":len(macro_values),
        "main_tables":sum(r["generated_file"].startswith("tables/") for r in objects),
        "appendix_tables":sum(r["generated_file"].startswith("appendix_tables/") for r in objects),
        "figure_pdfs":sum(r["kind"]=="figure" for r in objects),
        "analysis_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "panel_csv_sha256":hashlib.sha256((root/"data/raw/panel_data.csv").read_bytes()).hexdigest(),
        "output_file_count":sum(p.is_file() for p in out.rglob("*")),
        "all_values_finite":True,
    }
    print(json.dumps({"publication_assets":"success",**build_record},indent=2))


def sync_to_latex(root: Path, destination: Path):
    root=Path(root).resolve()
    destination=Path(destination).resolve()
    source=root/"output/manuscript_assets"
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("tables","appendix_tables","figures"):
        target=destination/name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source/name,target)
    for path in source.glob("*.tex"):
        shutil.copy2(path,destination/path.name)
    for name in ("MANUSCRIPT_OBJECT_MAP.csv","NUMERICAL_PROSE_MAP.csv"):
        shutil.copy2(source/name,destination/name)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex-dir",type=Path,help="Optional LaTeX destination to synchronize after generating assets.")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    build_assets(root)
    if args.latex_dir:
        sync_to_latex(root,args.latex_dir)

