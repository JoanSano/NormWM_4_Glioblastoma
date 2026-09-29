#!/usr/bin/env python3
"""Assemble the pooled clinical + imaging database used by every downstream analysis.

For each cohort the script reads the two tables produced by the per-cohort pipeline,

    <MAIN_DIR>/<cohort folder>/<TDMaps folder>/demographics-TDMaps_streamTH-<s>.csv
    <MAIN_DIR>/<cohort folder>/<TDMaps folder>/morphology-tissues.csv

harmonises their heterogeneous clinical schemas into a common set of covariates
(age, sex, extent of resection, MGMT, KPS, overall survival and censoring status),
concatenates the cohorts, estimates and removes the site effect on survival times,
and writes the resulting `data-clinical_*` table in CSV and TSV form together with
a `keys-maps.json` describing the categorical encodings.

Inclusion criteria are applied per cohort (IDH-wildtype, grade IV, no previous
treatment, known survival), so the assembled sample is smaller than the public
releases.

The site effect
---------------
`site` is a two-group partition of the selected cohorts: group 0 is the reference,
whose survival times are left untouched, and group 1's are multiplied by
exp(log HR). Which cohorts form the reference is an input, not a property of the
method -- it defaults to the `site` field of COHORTS (UCSF alone, the cohort whose
survival was recorded differently) and `--site-reference` names any other group.
The resolved partition is recorded in the provenance JSON, so a table always says
which grouping produced it.

By default the effect is estimated from the `site` indicator alone and the survival
times of group 1 are multiplied by exp(log HR). That crude effect might
be, however, not purely an artefact of how survival was recorded: the cohorts also 
differ in case-mix, and MGMT methylation in particular is far commoner in UCSF-PDGM 
than in the pooled remainder. `--adjust-covariates` therefore lets the site model 
condition on any of age, sex, EOR, MGMT and KPS.

The adjusted coefficient is estimated on the subjects reporting every chosen
covariate and then applied to every subject, so the assembled table never
shrinks -- missingness is severe and cohort-structured (EOR is unrecorded for all
of TCGA, MGMT for all of RHUH, KPS for all of UCSF and LUMIERE), and a
complete-case *table* would cost most of the sample. How the correction was
obtained is recorded in `<stem>_site-correction.json` next to the table.

The evidence behind that choice is reported on every run -- covariate balance and
missingness between the site groups, a same-sample adjustment ladder, a
reverse-Kaplan-Meier follow-up comparison and an administrative-truncation check.
`--ladder-covariates` and `--truncate-months` shape it; nothing turns it off,
because a correction whose justification is optional is a correction nobody
checks.

Every figure and table the run produces, and the recommendation it ends with, are
collected into a single self-contained `<stem>_report.html` next to the table, with
the figures embedded rather than linked. Open it in a browser, or print it to PDF
from there.

The terminal stays quiet: a run announces where it is writing, then prints the paths
it produced and the recommended survival column. The line-by-line record goes to
`createDatabase_log.txt` beside the table -- always written, renamed with `--log`,
and mirrored to the terminal by `--verbose`. The report names that file rather than
embedding it, so it stays a report rather than a log with pictures.

Examples
--------
    # As before: crude site correction
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_5-cohorts_Tissues \
                             --idh WT --grade IV --stream-th 0 --format pdf

    # Adjust the site effect for case-mix
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_5-cohorts_Tissues \
                             --adjust-covariates age sex eor mgmt

    # Pool a subset of the cohorts
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_2-cohorts --cohorts UCSF UPENN

    # Group the cohorts some other way: two cohorts as the reference
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_5-cohorts_Tissues \
                             --site-reference UCSF LUMIERE

    # Assemble the table with no correction at all
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_2-cohorts --cohorts UCSF UPENN \
                             --site-reference UCSF UPENN
"""

import argparse
import base64
import contextlib
import html
import io
import itertools
import json
import os
import sys
from datetime import datetime

import matplotlib

# Figures are written to disk; only open a window when explicitly asked for.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils.lowess import lowess
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from tqdm import tqdm

daysXmonth = 365 / 12
daysXweek = 7

# ---------------------------------------------------------------------------
# Columns kept in the pooled table
# ---------------------------------------------------------------------------
CLINICAL_COLUMNS = [
    "ID", "age", "sex", "eor", "mgmt", "kps (preop)", "OS (days)", "status",
]
MORPHOLOGY_COLUMNS = [
    "Whole tumor size (voxels)", "Core size (voxels)",
    "Non-enhancing size (voxels)", "Enhancing size (voxels)",
]
TRACT_DENSITY_COLUMNS = [
    "Whole TDMap", "Whole lesion TDMap",
    "Core TDMap", "Core lesion TDMap",
    "Non-enhancing TDMap", "Non-enhancing lesion TDMap",
    "Enhancing TDMap", "Enhancing lesion TDMap",
    "Core+Enhancing TDMap", "Core+Enhancing lesion TDMap",
]
COVARIATES_OF_INTEREST = CLINICAL_COLUMNS + MORPHOLOGY_COLUMNS + TRACT_DENSITY_COLUMNS

# Categorical encodings shared by all cohorts (the source labels differ, the codes do not)
MAP_EOR = {"biopsy": 0, "STR": 1, "GTR": 2, "Not Available": np.nan}
KEYS_MAPS = {
    "Sex": {"Male": 0, "Female": 1},
    "Extent of Resection": {
        "Biopsy": 0, "Subtotal (<90%)": 1, "Gross total (>= 90%)": 2,
        "Not Available": "np.nan",
    },
    "MGMT Promoter": {
        "Unmethylated/Negative": 0, "Intermediate": 1, "Methylated/Positive": 2,
        "Not Available": "np.nan",
    },
}

# ---------------------------------------------------------------------------
# Cohorts. `site` is the DEFAULT partition the site-effect correction acts on:
# 0 is the reference group, whose survival times are left untouched, and 1 the
# group whose times the fitted coefficient rescales. It encodes how survival was
# recorded, which for these five cohorts separates UCSF from the rest -- but that
# is a property of this particular selection, not of the method, so it is a
# default rather than a law. --site-reference names a different reference group,
# and the resolved partition travels with the output in the provenance JSON.
# ---------------------------------------------------------------------------
COHORTS = {
    "UCSF": dict(
        id=0, site=0, color="tab:green",
        folder="Glioblastoma_UCSF-PDGM_v3-20230111", subdir="TDMaps_Grade-{grade}",
    ),
    "UPENN": dict(
        id=1, site=1, color="tab:purple",
        folder="Glioblastoma_UPENN-GBM_v2-20221024", subdir="TDMaps_IDH1-{idh}",
    ),
    "TCGA": dict(
        id=2, site=1, color="tab:blue",
        folder="Glioblastoma_TCGA-GBM_v1-20170717", subdir="TDMaps_IDH1-{idh}",
    ),
    "RHUH": dict(
        id=3, site=1, color="tab:red",
        folder="Glioblastoma_RHUH-GBM_v2-29102025", subdir="TDMaps_IDH1-{idh}",
    ),
    "LUMIERE": dict(
        id=4, site=1, color="darkkhaki",
        folder="Glioblastoma_LUMIERE-GBM_v1-13122022", subdir="TDMaps_IDH1-{idh}",
    ),
}

# ---------------------------------------------------------------------------
# Clinical covariates the site-effect model may adjust for. `scale` divides a
# continuous covariate before it enters the design; it is 1.0 throughout, so every
# hazard ratio reads per native unit (one year, one KPS point). `levels` names the
# dummy contrasts of a categorical one against its lowest-coded reference level.
# ---------------------------------------------------------------------------
ADJUSTMENT_COVARIATES = {
    "age":  dict(column="age",         kind="continuous",  scale=1.0,  label="Age (per year)"),
    "sex":  dict(column="sex",         kind="categorical", scale=1.0,  label="Sex",
                 levels={0: "Male", 1: "Female"}),
    "eor":  dict(column="eor",         kind="categorical", scale=1.0,  label="EOR",
                 levels={0: "Biopsy", 1: "Subtotal (<90%)", 2: "Gross total (>=90%)"}),
    "mgmt": dict(column="mgmt",        kind="categorical", scale=1.0,  label="MGMT",
                 levels={0: "Unmethylated", 1: "Intermediate", 2: "Methylated"}),
    "kps":  dict(column="kps (preop)", kind="continuous",  scale=1.0,  label="KPS (per point)"),
}

# Fixed entry order, so a ladder is comparable between runs whatever order the
# covariates were typed on the command line.
ADJUSTMENT_ORDER = ["age", "sex", "eor", "mgmt", "kps"]

# Ladder walked by the diagnostics when no adjustment was requested. KPS is left
# out: UCSF and LUMIERE record none at all, so including it costs 100% of two
# cohorts and turns a complete-case model into a UPENN-only one.
DEFAULT_LADDER = ["age", "sex", "eor", "mgmt"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cohort_paths(main_dir, name, idh, grade, stream_th):
    """Locations of the per-cohort pipeline outputs.

    Args:
        main_dir: Root directory holding one folder per cohort.
        name: Cohort key into COHORTS ("UCSF", "UPENN", "TCGA", "RHUH", "LUMIERE").
        idh: IDH status the per-cohort pipeline was run with; fills the {idh} slot
            of that cohort's `subdir` template.
        grade: WHO grade the pipeline was run with; fills the {grade} slot, used by
            UCSF-PDGM only.
        stream_th: Minimum streamline density the indices were extracted at; part
            of the tract-density filename, not a filter applied here.

    Returns a dict with "td" and "morph", plus "eor" for LUMIERE, whose extent of
    resection is recomputed from the longitudinal scans rather than distributed.
    """
    meta = COHORTS[name]
    base = f"{main_dir}/{meta['folder']}/{meta['subdir'].format(idh=idh, grade=grade)}"
    paths = {
        "td": f"{base}/demographics-TDMaps_streamTH-{stream_th}.csv",
        "morph": f"{base}/morphology-tissues.csv",
    }
    if name == "LUMIERE":
        # Extent of resection is derived from the longitudinal scans (EOR.sh)
        paths["eor"] = f"{main_dir}/{meta['folder']}/data/LUMIERE_Extent-of-Resection.csv"
    return paths


def merge_td_morphology(td, morph):
    """Merge the tract-density and morphology tables on their shared clinical columns.

    Args:
        td: Tract-density table for one cohort.
        morph: Morphology table for the same cohort, already filtered the same way.

    The join keys are whatever columns the two tables have in common, so both must
    have been through the same inclusion filter or the merge silently drops rows.
    """
    shared = [c for c in td.columns if c in set(morph.columns)]
    return pd.merge(td, morph, on=shared)


def report_censoring(name, data):
    """Print how many subjects a cohort contributes and what fraction is censored.

    Args:
        name: Cohort name to print.
        data: Harmonised table with a 0/1 `status` column (1 = event observed).
    """
    n = data["status"].value_counts().sum()
    censored = (data["status"] == 0).sum()
    print(f"{name}: {n} subjects -- percentage of censoring: {round(100 * censored / n, 2)}%")


def as_structured(event, time):
    """Right-censored survival data in the structured-array form scikit-survival expects.

    Args:
        event: Per-subject event indicator, truthy where the event was observed.
        time: Per-subject follow-up time, in the same order as `event`.
    """
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", "bool"), ("time", "float")],
    )


def km_curve(data, duration_col, status_col):
    """Kaplan-Meier estimate with log-log confidence bands, anchored at (0, 1).

    Args:
        data: Table of subjects to estimate from; rows with a missing duration or
            status must already have been dropped.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Returns (time, survival_prob, conf_int) with a leading (0, 1) point inserted,
    so every curve starts at full survival rather than at the first event.
    """
    time, survival_prob, conf_int = kaplan_meier_estimator(
        data[status_col] == 1, data[duration_col], conf_type="log-log"
    )
    time = np.insert(time, 0, 0)
    survival_prob = np.insert(survival_prob, 0, 1)
    conf_int = np.insert(conf_int, 0, 1, axis=1)
    return time, survival_prob, conf_int


class Report:
    """Collects the run's figures, tables and log into one self-contained HTML file.

    Blocks are appended in the order they are produced, so the report reads in the
    order the analysis ran. Figures are embedded as base64 PNGs rather than linked:
    a report that stops rendering once the results directory is reorganised is not
    worth writing, and the .pdf/.svg files stay on disk for the manuscript.
    """

    def __init__(self):
        """Start an empty report. Takes no arguments."""
        self.blocks = []

    def heading(self, text, level=2):
        """Append a heading.

        Args:
            text: Heading text.
            level: HTML heading level, 2 for a section and 3 for a subsection.
        """
        self.blocks.append(("heading", (level, text)))

    def paragraph(self, text):
        """Append a paragraph of explanatory prose.

        Args:
            text: Plain text; escaped, so it may contain < and &.
        """
        self.blocks.append(("paragraph", text))

    def table(self, frame, caption=None, float_format=None, formatters=None):
        """Append a table.

        Args:
            frame: DataFrame to render; nothing is appended if it is empty.
            caption: Line printed above the table.
            float_format: Callable applied to every float cell.
            formatters: {column: callable} taking precedence over `float_format`,
                so a p-value column can keep its own notation.
        """
        if frame is None or frame.empty:
            return
        self.blocks.append(("table", (frame.copy(), caption, float_format, formatters)))

    def figure(self, fig, caption=None):
        """Append a figure, captured as it stands.

        Args:
            fig: Matplotlib figure. It is rendered to PNG immediately, so this
                must be called before the figure is closed.
            caption: Line printed under the figure.
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        self.blocks.append(
            ("figure", (base64.b64encode(buffer.getvalue()).decode("ascii"), caption)))

    def callout(self, text):
        """Append the one sentence a reader must not miss.

        Args:
            text: The verdict, in plain words.
        """
        self.blocks.append(("callout", text))

    def code(self, text, caption=None):
        """Append a block of preformatted text: a log, or a JSON document.

        Args:
            text: The text, kept verbatim.
            caption: Line printed above it.
        """
        self.blocks.append(("code", (text, caption)))

    def render(self, title, subtitle=""):
        """The complete HTML document as a string.

        Args:
            title: Document title and top heading.
            subtitle: Line under the title, typically the command that was run.
        """
        parts = [_REPORT_HEAD.format(title=html.escape(title))]
        parts.append(f"<h1>{html.escape(title)}</h1>")
        if subtitle:
            parts.append(f'<p class="subtitle">{html.escape(subtitle)}</p>')
        for kind, payload in self.blocks:
            if kind == "heading":
                level, text = payload
                parts.append(f"<h{level}>{html.escape(text)}</h{level}>")
            elif kind == "paragraph":
                parts.append(f"<p>{html.escape(payload)}</p>")
            elif kind == "callout":
                parts.append(f'<p class="verdict">{html.escape(payload)}</p>')
            elif kind == "code":
                text, caption = payload
                if caption:
                    parts.append(f'<p class="caption">{html.escape(caption)}</p>')
                parts.append(f"<pre>{html.escape(text)}</pre>")
            elif kind == "figure":
                encoded, caption = payload
                parts.append('<figure>'
                             f'<img src="data:image/png;base64,{encoded}" alt="'
                             f'{html.escape(caption or "figure")}">')
                if caption:
                    parts.append(f"<figcaption>{html.escape(caption)}</figcaption>")
                parts.append("</figure>")
            elif kind == "table":
                frame, caption, float_format, formatters = payload
                if caption:
                    parts.append(f'<p class="caption">{html.escape(caption)}</p>')
                parts.append('<div class="scroll">' + frame.to_html(
                    index=False, na_rep="n/a", border=0, escape=True,
                    float_format=float_format, formatters=formatters or {},
                ) + "</div>")
        parts.append("</main></body></html>")
        return "\n".join(parts)

    def write(self, path, title, subtitle=""):
        """Write the report and say where it went.

        Args:
            path: File to write.
            title: Document title and top heading.
            subtitle: Line under the title.
        """
        with open(path, "w") as handle:
            handle.write(self.render(title, subtitle))
        print(f"Report written to {path}")
        return path


# One per run. A module-level collector rather than a parameter threaded through
# every plotting helper: `save_figure` is the single point every figure passes
# through, so registering there catches all of them without touching signatures.
REPORT = Report()

_REPORT_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark;
           --ink: #1a1a1a; --bg: #ffffff; --muted: #5a5a5a;
           --rule: #d9d9d9; --band: #f5f5f5; --accent: #5b2d8e; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #e8e8e8; --bg: #16181c; --muted: #a0a0a0;
             --rule: #33363d; --band: #1e2127; --accent: #c4a7e7; }}
  }}
  html {{ background: var(--bg); }}
  body {{ margin: 0; padding: 0 16px 64px; color: var(--ink); background: var(--bg);
          font: 15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  main, h1, p.subtitle {{ max-width: 980px; margin-left: auto; margin-right: auto; }}
  h1 {{ font-size: 1.7rem; padding-top: 32px; margin-bottom: 4px; }}
  h2 {{ font-size: 1.25rem; margin-top: 40px; padding-top: 12px;
        border-top: 2px solid var(--accent); }}
  h3 {{ font-size: 1.05rem; margin-top: 28px; color: var(--muted); }}
  p.subtitle {{ color: var(--muted); margin-top: 0; font-family: ui-monospace,
                SFMono-Regular, Menlo, monospace; font-size: 0.85rem;
                word-break: break-all; }}
  p.caption {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 6px; }}
  p.verdict {{ border-left: 4px solid var(--accent); background: var(--band);
               padding: 12px 16px; margin: 16px 0; font-size: 1.02rem; }}
  figure {{ margin: 20px 0; }}
  img {{ max-width: 100%; height: auto; display: block; background: #fff;
         border: 1px solid var(--rule); border-radius: 6px; padding: 8px; }}
  figcaption {{ color: var(--muted); font-size: 0.85rem; margin-top: 8px; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums;
           font-size: 0.87rem; margin-bottom: 8px; }}
  th, td {{ padding: 5px 12px; text-align: right; white-space: nowrap;
            border-bottom: 1px solid var(--rule); }}
  th {{ text-align: right; font-weight: 600; }}
  td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
  tbody tr:nth-child(even) {{ background: var(--band); }}
  pre {{ background: var(--band); border: 1px solid var(--rule); border-radius: 6px;
         padding: 12px; overflow-x: auto; font-size: 0.8rem; line-height: 1.45; }}
</style></head><body><main>"""


def at_risk_and_censored(data, months, duration_col, status_col):
    """Numbers under a Kaplan-Meier curve: still at risk, and censored before then.

    Args:
        data: Subjects of one curve; the durations must already be whatever the
            curve plots, so a rescaled panel passes its rescaled times.
        months: Time points the row is printed at.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Returns (at_risk, censored), two lists aligned on `months`. `at_risk` counts
    the subjects whose follow-up reaches the month; `censored` is cumulative --
    everyone right-censored strictly before it -- so the pair reads "how many are
    still being watched (and how many stopped being watched)".

    The convention is deliberately duplicated from the
    Tract-Density_Components-Survival repository rather than imported: the two
    repositories stay independent on purpose, and the tables under their
    Kaplan-Meier curves have to be read the same way.
    """
    at_risk, censored = [], []
    for month in months:
        limit = month * daysXmonth
        at_risk.append(int((data[duration_col] >= limit).sum()))
        censored.append(int(((data[status_col] == 0)
                             & (data[duration_col] < limit)).sum()))
    return at_risk, censored


def draw_at_risk_table(ax, months, rows, colors, top=-0.07, step_y=0.06, fontsize=9.5):
    """Draw the "No. at risk (right-censored)" block under a Kaplan-Meier curve.

    Args:
        ax: Axes to draw on. Its limits and ticks must already be set, because the
            columns are spaced against the final data-to-pixel transform.
        months: Time points the counts were computed at.
        rows: One (at_risk, censored) pair per curve, as `at_risk_and_censored`
            returns them, in the order the curves were drawn.
        colors: One color per row, matching the curves.
        top: y of the first row, in data coordinates.
        step_y: vertical distance between rows, in data coordinates.
        fontsize: point size of the counts.

    Columns are thinned to whatever fits: "367 (144)" is roughly twice the width
    of the bare count this used to print, and eleven of them do not fit across a
    six-inch axis at a legible size. Every printed column is centred on its tick,
    so a thinned row still lines up with the axis it describes.
    """
    labels = [[f"{a} ({c})" for a, c in zip(*row)] for row in rows]
    stride = 1
    if len(months) > 1:
        spacing = abs(ax.transData.transform((months[1], 0))[0]
                      - ax.transData.transform((months[0], 0))[0])
        widest = max((len(text) for row in labels for text in row), default=0)
        # 0.6 em per character is the usual approximation for a proportional face
        needed = widest * fontsize * 0.6 * ax.figure.dpi / 72.0
        if spacing:
            stride = max(1, int(np.ceil(needed / spacing)))
    for k, row in enumerate(labels):
        for i in range(0, len(months), stride):
            ax.text(months[i], top - step_y * k, row[i], transform=ax.transData,
                    fontsize=fontsize, verticalalignment="top",
                    horizontalalignment="center", color=colors[k])
    ax.text(ax.get_xlim()[0], -0.01, "No. at risk (right-censored)",
            transform=ax.transData, fontsize=10, verticalalignment="top",
            color="black", fontweight="bold")
    ax.hlines(0, ax.get_xlim()[0], months[-1] + 5, color="black", linewidth=0.5)


def save_figure(fig, results, stem, formats):
    """Write one figure to `results`/OS-stats/ once per requested format.

    Args:
        fig: Matplotlib figure to write.
        results: Results directory; its OS-stats/ subdirectory must already exist.
        stem: File name without extension.
        formats: Extensions to write, e.g. ("pdf", "svg").

    The figure is also captured for the HTML report, which is why this is the only
    place figures are written: a figure saved past it would be missing there.
    """
    for fmt in formats:
        fig.savefig(f"{results}/OS-stats/{stem}.{fmt}", dpi=200, format=fmt)
    REPORT.figure(fig, caption=stem)


def eor_to_category(eor_series, threshold=90.0):
    """Convert continuous EOR (%) to GTR/STR/NaN, preserving missing values.

    Args:
        eor_series: Extent of resection as a percentage.
        threshold: Percentage at or above which a resection counts as gross total.

    A subject with no recorded percentage stays NaN rather than falling into the
    lower category, which `pd.cut` would otherwise do silently.
    """
    return pd.cut(
        eor_series,
        bins=[-np.inf, threshold, np.inf],
        labels=["STR", "GTR"],
        right=False,  # [threshold, inf) -> GTR, i.e. >= threshold
    ).astype(object).where(eor_series.notna(), other=np.nan)


def resolve_site_codes(names, reference=None):
    """Assign each selected cohort to the 0/1 group the correction acts on.

    Args:
        names: Cohort names pooled in this run.
        reference: Cohort names forming the reference group (site 0), or None to
            take the `site` field of COHORTS.

    Returns {cohort name: 0 or 1}. Which group is the reference is not cosmetic:
    site 0 keeps its survival times untouched and site 1's are rescaled by
    exp(logHR), so naming the reference chooses which group is being corrected
    *onto*. A reference covering every selected cohort leaves one group, which is
    the supported way of assembling the table with no correction at all.
    """
    if reference is None:
        return {n: COHORTS[n]["site"] for n in names}
    reference = set(reference)
    return {n: (0 if n in reference else 1) for n in names}


def is_default_partition(site_of):
    """Whether a resolved partition agrees with the `site` field of COHORTS.

    Args:
        site_of: {cohort name: 0 or 1}, as `resolve_site_codes` returns.

    Only the default partition may reuse the legacy "OTHERS" label, so a custom
    run cannot overwrite the figures of the canonical one.
    """
    return all(site == COHORTS[name]["site"] for name, site in site_of.items())


def build_site_labels(names, site_of=None):
    """Name each site group from the cohorts actually pooled into it.

    Args:
        names: Cohort names being pooled in this run, in any order.
        site_of: {cohort name: 0 or 1} from `resolve_site_codes`, or None for the
            default partition.

    The legacy name "OTHERS" is kept when a group holds every cohort of that site
    -- which is what the full run pools, and what every figure already on disk is
    named after. Any narrower selection is named after the cohorts in it, because
    a figure labelled "OTHERS" while only two of the four were pooled is both
    wrong and liable to overwrite the full run's figure. A partition that is not
    the default one never earns the legacy name, for the same reason: it would
    put a differently grouped run under the canonical run's file names.
    """
    site_of = resolve_site_codes(names) if site_of is None else site_of
    legacy = is_default_partition(site_of)
    labels = {}
    for site in sorted({site_of[n] for n in names}):
        selected = [n for n in names if site_of[n] == site]
        everything = [n for n in COHORTS if COHORTS[n]["site"] == site]
        if len(selected) == 1:
            labels[site] = selected[0]
        elif legacy and set(selected) == set(everything):
            labels[site] = "OTHERS"
        else:
            labels[site] = "+".join(sorted(selected, key=lambda n: COHORTS[n]["id"]))
    return labels


def km_median(time, survival_prob):
    """First time at which a Kaplan-Meier estimate reaches 0.5; nan if it never does.

    Args:
        time: Times of a Kaplan-Meier curve, ascending.
        survival_prob: Survival probability at each of those times.

    Used for both median survival and -- on the curve of the flipped event
    indicator -- median potential follow-up, so the two are read off by exactly
    the same estimator as every curve this script plots.
    """
    below = np.flatnonzero(np.asarray(survival_prob) <= 0.5)
    return float(np.asarray(time)[below[0]]) if below.size else np.nan


def default_output_stem(names, prefix="data-clinical_TD-tissues"):
    """Base name of the assembled table for a given cohort selection.

    Args:
        names: Cohort names being pooled in this run.
        prefix: Leading part of the file name, before the cohort description.

    `<prefix>_<N>-cohorts` is ambiguous the moment --cohorts is used: {UCSF, TCGA}
    and {UPENN, RHUH} are both "2-cohorts" and would overwrite each other -- and
    each other's keys-maps.json and figures -- in the same results directory. The
    legacy name is therefore kept only when the selection is the first N cohorts
    by ID, which is what every existing *_4-cohorts.csv and *_5-cohorts.csv on
    disk is, and any other selection is named after the cohorts in it.
    """
    ordered = sorted(names, key=lambda n: COHORTS[n]["id"])
    ids = [COHORTS[n]["id"] for n in ordered]
    if ids == list(range(len(ids))):
        return f"{prefix}_{len(ids)}-cohorts"
    return f"{prefix}_{'-'.join(ordered)}"


# ---------------------------------------------------------------------------
# Per-cohort harmonisation
# ---------------------------------------------------------------------------
def load_lumiere(paths):
    """LUMIERE: IDH-wildtype with known survival; every subject experienced an event.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs plus the recomputed extent-of-resection table.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
        df = df.loc[df["IDH (WT: wild type)"].str.upper() == "WT"]
        return df.loc[df["Survival time (weeks)"] != "na"]

    data = merge_td_morphology(
        include(pd.read_csv(paths["td"])), include(pd.read_csv(paths["morph"]))
    )
    data["status"] = 1

    # Survival is recorded in weeks; the remaining cohorts use days
    data["OS (days)"] = data["Survival time (weeks)"].astype(int) * daysXweek
    data = data.rename(columns={
        "Patient": "ID",
        "Age at surgery (years)": "age",
        "Sex": "sex",
        "MGMT qualitative": "mgmt",
    })

    # EOR is not distributed with the cohort; it is recomputed from the pre/post-op
    # segmentations, and is missing for the subjects without a post-operative scan
    eor_lookup = (
        pd.read_csv(paths["eor"])
        .set_index("Patient")["Extent of Resection (%)"]
        .replace("na", np.nan)
        .astype(np.float64)
    )
    data["eor"] = eor_to_category(data["ID"].map(eor_lookup))
    data["kps (preop)"] = np.nan

    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    # Intermediate methylation is not reported for this cohort
    data["mgmt"] = data["mgmt"].map({"not methylated": 0, "methylated": 2, "na": np.nan})
    data["eor"] = data["eor"].map(MAP_EOR)
    return data[COVARIATES_OF_INTEREST].copy()


def load_rhuh(paths):
    """RHUH: IDH-wildtype, treatment-naive, with a known censoring status.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
        df = df.loc[df["IDH status"] == "wt"]
        df = df.loc[df["Previous treatment"] == "no"]
        return df.loc[df["Right Censored"].fillna("unknown") != "unknown"]

    data = merge_td_morphology(
        include(pd.read_csv(paths["td"])), include(pd.read_csv(paths["morph"]))
    )

    # 'Right Censored' is the opposite of 'status' (see the original reference)
    data = data.rename(columns={"Right Censored": "status"})
    data["status"] = data["status"].str.lower().map({"no": 1, "yes": 0})

    # Unify the resection percentages with the categories used by the other cohorts
    gtr = data["Extent of resection [EOR]  %"] >= 90
    data.loc[gtr, "EOR"] = "GTR"
    data.loc[~gtr, "EOR"] = "STR"

    data = data.rename(columns={
        "Patient ID": "ID",
        "Age": "age",
        "Sex": "sex",
        "EOR": "eor",
        "Overall survival [OS] (days)": "OS (days)",
        "Preoperative KPS": "kps (preop)",
    })
    data["mgmt"] = np.nan  # MGMT promoter status is not reported for this cohort

    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["eor"] = data["eor"].map(MAP_EOR)
    return data[COVARIATES_OF_INTEREST].copy()


def load_tcga(paths):
    """TCGA-GBM: IDH-wildtype subset of the pre-operative collection.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
        return df.loc[df["IDH.status"] == "WT"]

    data = merge_td_morphology(
        include(pd.read_csv(paths["td"])), include(pd.read_csv(paths["morph"]))
    )

    # Survival is recorded in months; the remaining cohorts use days
    data["OS (days)"] = data["Survival..months."] * daysXmonth
    data = data.rename(columns={
        "patient": "ID",
        "Age..years.at.diagnosis.": "age",
        "Gender": "sex",
        "MGMT.promoter.status": "mgmt",
        "Vital.status..1.dead.": "status",
        "Karnofsky.Performance.Score": "kps (preop)",
    })
    data["eor"] = np.nan  # extent of resection is not reported for this cohort

    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    data["mgmt"] = data["mgmt"].map(
        {"Unmethylated": 0, "Indeterminate": 1, "Methylated": 2, "Not Available": np.nan}
    )
    return data[COVARIATES_OF_INTEREST].copy()


def load_ucsf(paths):
    """UCSF-PDGM: IDH-wildtype glioblastomas (WHO 2021) with known survival.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
        df = df.loc[df["Final pathologic diagnosis (WHO 2021)"] == "Glioblastoma  IDH-wildtype"]
        return df.loc[df["OS"].fillna("unknown") != "unknown"]

    data = merge_td_morphology(
        include(pd.read_csv(paths["td"])), include(pd.read_csv(paths["morph"]))
    )

    data = data.rename(columns={
        "Age at MRI": "age",
        "Sex": "sex",
        "MGMT status": "mgmt",
        "EOR": "eor",
        "OS": "OS (days)",
        "1-dead 0-alive": "status",
    })
    data["kps (preop)"] = np.nan  # KPS is not reported for this cohort

    data["sex"] = data["sex"].map({"M": 0, "F": 1})
    data["eor"] = data["eor"].map(MAP_EOR)
    data["mgmt"] = data["mgmt"].map(
        {"negative": 0, "indeterminate": 1, "positive": 2, "Not Available": np.nan}
    )
    return data[COVARIATES_OF_INTEREST].copy()


def load_upenn(paths):
    """UPENN-GBM: de novo glioblastoma; the pipeline tables are already IDH-wildtype.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    data = merge_td_morphology(pd.read_csv(paths["td"]), pd.read_csv(paths["morph"]))

    data = data.rename(columns={
        "Age_at_scan_years": "age",
        "Gender": "sex",
        "MGMT": "mgmt",
        "GTR_over90percent": "eor",
        "Survival_from_surgery_days_UPDATED": "OS (days)",
        "Survival_Status": "status",
        "KPS": "kps (preop)",
    })

    data["sex"] = data["sex"].map({"M": 0, "F": 1})
    # Resection is only reported as above/below the 90% threshold, i.e. STR vs GTR
    data["eor"] = data["eor"].map({"N": 1, "Y": 2, "Not Available": np.nan})
    data["mgmt"] = data["mgmt"].map(
        {"Unmethylated": 0, "Indeterminate": 1, "Methylated": 2, "Not Available": np.nan}
    )
    data["kps (preop)"] = data["kps (preop)"].replace("Not Available", np.nan)
    return data[COVARIATES_OF_INTEREST].copy()


LOADERS = {
    "UCSF": load_ucsf,
    "UPENN": load_upenn,
    "TCGA": load_tcga,
    "RHUH": load_rhuh,
    "LUMIERE": load_lumiere,
}


# ---------------------------------------------------------------------------
# Survival across cohorts
# ---------------------------------------------------------------------------
def plot_cohort_survival(
        full_data,
        cohort_ids,
        name_cohort,
        colors,
        RESULTS,
        stem,
        title,
        duration_col="OS (days)",
        status_col="status",
        covariate_col="cohort",
        months=range(0, 110, 10),
        formats=("pdf", "svg"),
        show_plot=True,
    ):
    """Kaplan-Meier curves of every cohort, with pairwise log-rank tests.

    Args:
        full_data: Pooled table of every selected cohort.
        cohort_ids: Codes to draw, one curve each, in plotting order.
        name_cohort: {code: name} used for the legend and the printed tests.
        colors: One matplotlib color per entry of `cohort_ids`, same order.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        title: Figure title.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        covariate_col: Column holding the grouping variable ("cohort" or "site").
        months: Time points (months) of the numbers-at-risk row.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.

    The omnibus log-rank is annotated on the figure and skipped when a single
    cohort is selected; the pairwise tests are printed, not plotted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    OS_STATS = []
    GROUP_STATS = []
    nums = np.empty(shape=(len(cohort_ids),), dtype=object)
    for i, cohort in enumerate(cohort_ids):
        diag = full_data[full_data[covariate_col] == cohort]
        diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])]

        nums[i] = at_risk_and_censored(diag, months, duration_col, status_col)

        time, survival_prob, conf_int = km_curve(diag, duration_col, status_col)
        ax.step(time / daysXmonth, survival_prob, where="post", color=colors[i],
                label=f"Cohort: {name_cohort[cohort]}", linewidth=2)
        ax.fill_between(time / daysXmonth, conf_int[0], conf_int[1], alpha=0.10,
                        step="post", color=colors[i])
        for t in diag.loc[diag[status_col] == 0, duration_col].values:  # Censoring times
            ax.plot(time[time == t] / daysXmonth, survival_prob[time == t], "|", color=colors[i])

        OS_STATS.extend([(st, ovs) for st, ovs in zip(diag[status_col] == 1, diag[duration_col].values)])
        GROUP_STATS.extend([i + 1 for _ in diag[duration_col].values])

    OS_STATS = as_structured([e for e, _ in OS_STATS], [t for _, t in OS_STATS])
    # A log-rank test needs something to compare against: a single selected cohort
    # would otherwise abort the whole run inside sksurv.
    if len(set(GROUP_STATS)) >= 2:
        chisquared, p_val, stats, covariance = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
        ax.text(0.70, 0.90, r"$\chi^2 =$" + f"{round(chisquared, 4)}, \np = {round(p_val, 4)}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val <= 0.05 else "black")
    else:
        ax.text(0.70, 0.90, "single cohort\nno log-rank test",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.1), color="black")

    ax.set_ylim([-(0.085 + 0.06 * len(cohort_ids)), 1.1])
    ax.set_xlim([-5, 75])
    ax.set_xticks(range(0, months[-1] + 10, 10))
    ax.set_xticklabels(range(0, months[-1] + 10, 10))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, months[-1])
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Overall survival", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    # After the ticks: set_xticks widens the limits past set_xlim when a tick sits
    # outside them, and the table is spaced against the limits that end up drawn
    draw_at_risk_table(ax, list(months), [nums[k] for k in range(len(cohort_ids))],
                       colors)
    fig.suptitle(title, fontweight="bold")
    save_figure(fig, RESULTS, stem, formats)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Pairwise log-rank tests between cohorts
    GROUP_STATS = np.array(GROUP_STATS)
    for i, j in itertools.combinations(range(len(cohort_ids)), 2):
        pair = (GROUP_STATS == i + 1) | (GROUP_STATS == j + 1)
        chisquared, p_val = compare_survival(
            OS_STATS[pair], list(GROUP_STATS[pair]), return_stats=False
        )
        message = (f"Cohorts {name_cohort[cohort_ids[i]]} and {name_cohort[cohort_ids[j]]}: "
                   f"chi-squared of {chisquared.round(4)} with p-value of {p_val.round(4)} "
                   f"(two-sided log-rank test)")
        print(f"ATENTION!!\n------>\t {message}" if p_val <= 0.05 else message)




def inspect_survival_diffs_in_paired_cohorts(
        full_data,
        cohorts,
        RESULTS,
        name_cohort,
        colors,
        N_cohorts,
        duration_col="OS (days)",
        status_col="status",
        covariate_col="cohort",
        months=range(0, 110, 10),
        n_perms=1000,
        eps_=0.01,
        formats=("pdf", "svg"),
        show_plot=True,
        logHR_override=None,
        adjust_covariates=(),
    ):
    """Quantify the survival difference between two groups, and correct for it.

    Args:
        full_data: Pooled table holding both groups.
        cohorts: The two group codes to compare; sorted on entry, and the second
            one is the group whose survival times the right-hand panel rescales.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        name_cohort: {code: name} used for the title, legend and file name.
        colors: One matplotlib color per group, in the order of sorted `cohorts`.
        N_cohorts: Group sizes shown in the legend, in the same order.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        covariate_col: Column holding the grouping variable ("cohort" or "site").
        months: Time points (months) of the numbers-at-risk row.
        n_perms: Permutations of the C-index test on the crude group effect.
        eps_: Offset added inside the logs of the cloglog panel, so that a
            survival of 1 or a time of 0 stays plottable.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.
        logHR_override: Coefficient to rescale the right-hand panel by, in place
            of the crude one fitted here.
        adjust_covariates: Covariates `logHR_override` was adjusted for.

    Left panels show the complementary log-log survival curves and the scaled
    Schoenfeld residuals of the Grambsch-Therneau test (i.e. whether the difference
    is a proportional-hazards one); the right panel shows the effect of rescaling
    the survival times of the second group by the fitted hazard ratio.

    Returns the log hazard ratio of the second group relative to the first. It is
    contingent on the {0, 1} recoding of the two groups, not on the original IDs.

    `logHR_override` rescales the right-hand panel by a coefficient estimated
    elsewhere -- the covariate-adjusted site effect. Without it the figure would
    advertise a correction other than the one written to the data file. The
    returned value is always the crude coefficient, whatever is plotted.

    `adjust_covariates` names the covariates that coefficient was adjusted for, and
    they enter the Grambsch-Therneau fit as well, so the proportional-hazards panel
    diagnoses the model actually being applied. Proportional hazards is a property
    of a model, not of a pair of groups: a site term can satisfy it crudely and
    violate it once case-mix is held fixed, or the reverse, so diagnosing the crude
    fit while applying the adjusted coefficient vouches for the wrong model. Pass
    it only alongside the `logHR_override` it belongs to -- the two describe one
    model, and splitting them puts the figure back in the state this avoids.

    The cloglog curves and the log-rank beside them stay marginal whatever is
    adjusted for: a Kaplan-Meier curve has no covariates to hold fixed, and the
    annotation says so. The adjusted inference on the site term is printed by the
    caller, which fits that model.
    """
    cohorts = sorted(cohorts)
    logHR_applied = logHR_override

    print("+" * 40)
    print(f"{name_cohort[cohorts[0]]} vs. {name_cohort[cohorts[1]]}")

    # Pre-correction
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[:, 1])
    ax3 = plt.subplot(gs[1, 0])

    OS_STATS = []
    GROUP_STATS = []
    for i, cohort in enumerate(cohorts):
        diag = full_data[full_data[covariate_col] == cohort]
        diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])]

        time, survival_prob, conf_int = km_curve(diag, duration_col, status_col)
        ax1.step(np.log(eps_ + time / daysXmonth), np.log(eps_ - np.log(eps_ + survival_prob)),
                 where="post", color=colors[i],
                 label=f"Cohort: {name_cohort[cohort]} (n={N_cohorts[i]})")
        ax1.fill_between(np.log(eps_ + time / daysXmonth),
                         np.log(eps_ - np.log(eps_ + conf_int[0])),
                         np.log(eps_ - np.log(eps_ + conf_int[1])),
                         alpha=0.15, step="post", color=colors[i])
        for t in diag.loc[diag[status_col] == 0, duration_col].values:  # Censoring times
            ax1.plot(np.log(eps_ + time[time == t] / daysXmonth),
                     np.log(eps_ - np.log(eps_ + survival_prob[time == t])), "|", color=colors[i])

        OS_STATS.extend([(st, ovs) for st, ovs in zip(diag[status_col] == 1, diag[duration_col].values)])
        GROUP_STATS.extend([i + 1 for _ in diag[duration_col].values])

    OS_STATS = as_structured([e for e, _ in OS_STATS], [t for _, t in OS_STATS])
    chisquared, p_val, stats, covariance = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
    ax1.text(0.05, 0.90,
             r"$\chi^2 =$" + f"{round(chisquared, 4)}, \np = {round(p_val, 4)}"
             + "\n(log-rank, unadjusted)",
             transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val <= 0.05 else "black")
    ax1.set_xlabel("log [ time ]", fontsize=12)
    ax1.set_ylabel("log [ -log [ Survival ] ]", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False)

    # Fit the model to estimate the site effect ---> the covariate should be {0,1},
    # not the original cohort IDs
    map4cox = {cohorts[0]: 0, cohorts[1]: 1}
    mask = (
        ~np.isnan(full_data[status_col]) & ~np.isnan(full_data[duration_col])
        & full_data[covariate_col].isin(cohorts)
    )
    model_data = full_data.loc[mask]
    X = model_data[covariate_col].map(map4cox).values.reshape(-1, 1)
    y = as_structured(model_data[status_col] == 1, model_data[duration_col])

    Cmodel = CoxPHSurvivalAnalysis(n_iter=200)
    Cmodel.fit(X, y)
    c_index = Cmodel.score(X, y)
    pop = []
    for _ in tqdm(range(n_perms)):
        perm_y = np.random.permutation(y)
        p_Cmodel = CoxPHSurvivalAnalysis()
        p_Cmodel.fit(X, perm_y)
        pop.append(p_Cmodel.score(X, perm_y))
    p_value = np.mean(np.array(pop) >= c_index)
    print(f"C-index = {c_index} (p={p_value})")
    print(f"Log HR = {Cmodel.coef_}")

    # Plot Schoenfeld residuals. The covariate is remapped to {0, 1} here as well:
    # fitting on the raw IDs would put this model on a different scale from the one
    # above (for cohorts [0, 2] its coefficient would be half the Log HR printed
    # below), so the residuals would diagnose a fit other than the reported one.
    # Reusing `mask` also keeps both models on exactly the same rows.
    #
    # `adjust_covariates` enter here too, so what is diagnosed is the model whose
    # coefficient the right-hand panel applies. The price is the complete-case
    # restriction the adjusted fit already pays: this panel then describes fewer
    # subjects than the curves beside it, which its axis label reports.
    data = full_data.loc[mask, [covariate_col, duration_col, status_col]].copy()
    data[covariate_col] = data[covariate_col].map(map4cox)
    n_curves = len(data)
    ph_covariates = list(adjust_covariates)
    if ph_covariates:
        design, _, design_notes = build_site_design(full_data.loc[mask], ph_covariates)
        for note in design_notes:
            print(f"  PH diagnostic -- {note}")
        data = data.join(design).dropna()

    lifelines_model, reason = fit_cox(data, duration_col, status_col)
    if lifelines_model is None and ph_covariates:
        # An adjusted design can be singular where site alone is not; a diagnostic
        # of the crude fit, clearly labelled, beats no diagnostic at all
        print(f"  PH diagnostic -- adjusted fit failed ({reason}); "
              f"falling back to site alone.")
        ph_covariates = []
        data = full_data.loc[mask, [covariate_col, duration_col, status_col]].copy()
        data[covariate_col] = data[covariate_col].map(map4cox)
        lifelines_model, reason = fit_cox(data, duration_col, status_col)

    if lifelines_model is None:
        print(f"\nWARNING: the Grambsch-Therneau diagnostic could not be fitted ({reason}).")
        ax3.set_xlabel(f"rank-transformed time\n(diagnostic unavailable:\n{reason})", fontsize=8)
    else:
        test = proportional_hazard_test(lifelines_model, data, time_transform="all")  # Grambsch-Therneau
        print("\nResults from the Grambsch-Therneau test "
              f"({'adjusted for ' + ', '.join(ph_covariates) if ph_covariates else 'unadjusted'}):")
        print(test.summary)
        # By name, never positionally: with the adjustment covariates in the design
        # the first row of the summary is whichever term lifelines ordered first,
        # which is not necessarily site
        summary = test.summary
        if isinstance(summary.index, pd.MultiIndex):
            level = 0 if covariate_col in summary.index.get_level_values(0) else 1
            site_rows = summary.xs(covariate_col, level=level)
        else:
            site_rows = summary.loc[[covariate_col]]
        rank_stat = float(site_rows["test_statistic"].iloc[0])
        p_val_GT = float(site_rows["p"].iloc[0])
        schoenfeld_residuals = lifelines_model.compute_residuals(data, kind="scaled_schoenfeld")
        site_schoenfeld_residuals = schoenfeld_residuals[covariate_col].values
        tt = data.loc[data[status_col] == 1, duration_col].rank()
        ax3.scatter(tt[site_schoenfeld_residuals > 0], site_schoenfeld_residuals[site_schoenfeld_residuals > 0],
                    color=colors[1], alpha=0.25)
        ax3.scatter(tt[site_schoenfeld_residuals < 0], site_schoenfeld_residuals[site_schoenfeld_residuals < 0],
                    color=colors[0], alpha=0.25)
        y_lowess = lowess(tt.values, site_schoenfeld_residuals)
        ax3.scatter(tt.values, y_lowess, color="k", alpha=1.0, s=1)
        for _ in range(10):  # Bootstrapped lowess fits
            ix = sorted(np.random.choice(tt.shape[0], tt.shape[0]))
            tt_ = tt.values[ix]
            y_lowess = lowess(tt_, site_schoenfeld_residuals[ix])
            ax3.scatter(tt_, y_lowess, color="gray", alpha=0.10, s=2, marker="+")
        kind_ph = ("adjusted for " + ", ".join(ph_covariates)) if ph_covariates else "unadjusted"
        ax3.set_xlabel(f"rank-transformed time\n(GT={rank_stat:.4f}; p={p_val_GT:.4f})\n"
                       f"{kind_ph}; n={len(data)} of {n_curves}", fontsize=8)
    ax3.set_ylabel("scaled-Schoenfeld Residuals", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)

    # Post-correction
    if logHR_applied is None:
        logHR_applied = Cmodel.coef_[0]
    OS_STATS_deSITE = []
    GROUP_STATS = []
    nums = np.empty(shape=(2,), dtype=object)
    for i, cohort in enumerate(cohorts):
        diag = full_data[full_data[covariate_col] == cohort]
        diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])].copy()
        diag[duration_col] = diag[duration_col] * np.exp(logHR_applied * i)
        # Computed on the rescaled times, so the row describes the curve drawn
        nums[i] = at_risk_and_censored(diag, months, duration_col, status_col)

        time, survival_prob, conf_int = km_curve(diag, duration_col, status_col)
        kind = "adjusted" if logHR_override is not None else "crude"
        label = (f"{name_cohort[cohort]} corrected\n(log HR = {logHR_applied:.4f}, {kind})"
                 if i == 1 else f"{name_cohort[cohort]}")
        ax2.step(time / daysXmonth, survival_prob, where="post", color=colors[i], label=label)
        ax2.fill_between(time / daysXmonth, conf_int[0], conf_int[1], alpha=0.15,
                         step="post", color=colors[i])
        for t in diag.loc[diag[status_col] == 0, duration_col].values:  # Censoring times
            ax2.plot(time[time == t] / daysXmonth, survival_prob[time == t], "|", color=colors[i])

        OS_STATS_deSITE.extend([(st, ovs) for st, ovs in zip(diag[status_col] == 1, diag[duration_col].values)])
        GROUP_STATS.extend([i + 1 for _ in diag[duration_col].values])

    OS_STATS_deSITE = as_structured([e for e, _ in OS_STATS_deSITE], [t for _, t in OS_STATS_deSITE])
    chisquared, p_val, stats, covariance = compare_survival(OS_STATS_deSITE, GROUP_STATS, return_stats=True)
    ax2.text(0.70, 0.90, r"$\chi^2 =$" + f"{round(chisquared, 4)}, \np = {round(p_val, 4)}",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val <= 0.05 else "black")

    # Overlay the uncorrected curve of the rescaled cohort
    diag = full_data[full_data[covariate_col] == cohorts[1]]
    diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])]
    time, survival_prob, conf_int = km_curve(diag, duration_col, status_col)
    ax2.step(time / daysXmonth, survival_prob, where="post", color="gray", alpha=0.5,
             label=f"{name_cohort[cohorts[1]]} uncorrected")
    ax2.fill_between(time / daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="gray")
    for t in diag.loc[diag[status_col] == 0, duration_col].values:  # Censoring times
        ax2.plot(time[time == t] / daysXmonth, survival_prob[time == t], "|", color="gray")

    ax2.set_ylim([-0.2, 1])
    ax2.set_xlim([-5, 75])
    ax2.set_xticks(range(0, months[-1] + 10, 10))
    ax2.set_xticklabels(range(0, months[-1] + 10, 10))
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.spines["left"].set_bounds(0, 1)
    ax2.spines["bottom"].set_bounds(0, months[-1])
    ax2.set_xlabel("Time (months)", fontsize=12)
    ax2.set_ylabel("Overall survival", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False)
    draw_at_risk_table(ax2, list(months), [nums[0], nums[1]], colors)

    fig.suptitle(f"{name_cohort[cohorts[0]]} vs. {name_cohort[cohorts[1]]}", fontweight="bold")
    fig.tight_layout()
    # An adjusted run gets its own file, so the crude figure of an earlier run is
    # never silently overwritten by one drawn with a different coefficient.
    suffix = "" if logHR_override is None else "_adjusted"
    save_figure(fig, RESULTS,
                f"Site-effects_Survival-times_{name_cohort[cohorts[0]]}-{name_cohort[cohorts[1]]}{suffix}",
                formats)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print("+" * 40)
    return Cmodel.coef_[0]


# ---------------------------------------------------------------------------
# Reporting vocabulary
#
# Deliberately duplicated rather than imported: this script sits upstream of the
# analysis pipeline and the two live in separate repositories, so neither may
# import the other. The names match the pipeline's on purpose, so both logs read
# alike; the duplication is the price of keeping the repositories independent.
# ---------------------------------------------------------------------------
def fmt_p(p):
    """A p-value at a fixed width, without pretending to precision it lacks.

    Args:
        p: The p-value, or None/NaN when the test could not be run.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "     n/a"
    return "  <0.001" if p < 0.001 else f"{p:8.3f}"


def section(title):
    """Print a top-level banner.

    Args:
        title: Text of the banner.
    """
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    """Print a second-level banner.

    Args:
        title: Text of the banner.
    """
    print(f"\n-- {title} " + "-" * max(0, 74 - len(title)))


# ---------------------------------------------------------------------------
# Site effect: estimation, adjustment and diagnostics
# ---------------------------------------------------------------------------
def fit_cox(frame, duration_col, status_col, strata=None):
    """Fit one Cox model, returning (fitter, None) or (None, reason).

    Args:
        frame: Design matrix plus duration and status; every other column is
            taken as a covariate, so it must hold nothing else.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        strata: Columns to stratify on, or None for an unstratified fit.

    Separation, collinearity and too-few-events come back as a string rather than
    an exception, so one unfittable rung does not abort a whole ladder.
    """
    try:
        model = CoxPHFitter()
        model.fit(frame, duration_col=duration_col, event_col=status_col, strata=strata)
        return model, None
    except Exception as exc:  # convergence, singular design, empty strata, ...
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def build_site_design(data, covariates):
    """Design matrix of the adjustment covariates, aligned on `data.index`.

    Args:
        data: Table to read the covariate columns from.
        covariates: Keys of ADJUSTMENT_COVARIATES to build; entered in
            ADJUSTMENT_ORDER whatever order they arrive in, and any key whose
            column is absent, all-missing or constant is skipped with a note.

    Continuous covariates are divided by their `scale`, which is 1.0 throughout, so
    the hazard ratio reads per native unit; categorical ones are expanded into
    indicators against their lowest observed level, and a subject with no MGMT
    result gets NaN in every MGMT indicator rather than being silently coded as
    the reference.

    NaNs are left in place. The caller drops those rows, so the model is fitted on
    complete cases while the coefficient is applied to everyone.

    Returns (design, blocks, notes), where `blocks` is a list of
    {"key", "label", "columns"} in ADJUSTMENT_ORDER -- the unit the ladder walks.
    """
    design = pd.DataFrame(index=data.index)
    blocks, notes = [], []

    for key in [k for k in ADJUSTMENT_ORDER if k in covariates]:
        spec = ADJUSTMENT_COVARIATES[key]
        column = spec["column"]
        if column not in data.columns:
            notes.append(f"{spec['label']}: column '{column}' absent from the table -- skipped.")
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() == 0:
            notes.append(f"{spec['label']}: not recorded for any selected subject -- skipped.")
            continue

        if spec["kind"] == "continuous":
            name = spec["label"]
            design[name] = values / spec["scale"]
            if design[name].nunique(dropna=True) < 2:
                notes.append(f"{spec['label']}: constant -- skipped.")
                design = design.drop(columns=[name])
                continue
            blocks.append(dict(key=key, label=spec["label"], columns=[name]))
            continue

        # Categorical: indicators against the lowest observed level
        levels = sorted(values.dropna().unique())
        if len(levels) < 2:
            only = spec["levels"].get(levels[0], levels[0]) if levels else "none"
            notes.append(f"{spec['label']}: only one level observed ({only}) -- skipped.")
            continue
        reference, contrasts = levels[0], levels[1:]
        columns = []
        for level in contrasts:
            name = f"{spec['label']}: {spec['levels'].get(level, level)}"
            # NaN in every indicator, so an unrecorded subject is dropped by the
            # caller's dropna() rather than being absorbed into the reference level
            design[name] = np.where(values.isna(), np.nan, (values == level).astype(float))
            columns.append(name)
        notes.append(
            f"{spec['label']}: reference level is "
            f"{spec['levels'].get(reference, reference)}."
        )
        blocks.append(dict(key=key, label=spec["label"], columns=columns))

    return design, blocks, notes


def estimate_site_logHR(data, covariates=(), duration_col="OS (days)",
                        status_col="status", site_col="site", strata_col=None):
    """Log hazard ratio of site=1 relative to site=0, adjusted for `covariates`.

    Args:
        data: Table holding the site column, the survival columns and whatever
            `covariates` names.
        covariates: Keys of ADJUSTMENT_COVARIATES to adjust for; empty for the
            crude effect.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        site_col: Column holding the 0/1 group indicator whose coefficient is
            returned. It need not be called "site" -- the pairwise figures pass a
            per-pair indicator -- but it must be coded 0/1, since the coefficient
            is applied downstream as exp(logHR * code).
        strata_col: Column to stratify the baseline hazard on, or None.

    Fitted with lifelines rather than scikit-survival because the adjusted model
    needs a named coefficient, a standard error and a confidence interval, and
    because `strata` has no CoxPHSurvivalAnalysis equivalent. The sign convention
    matches -- a positive coefficient is a higher hazard, i.e. shorter survival,
    for site=1 -- so the correction applied downstream keeps its direction.

    The estimate is made on whoever reports every covariate in the model, and the
    caller applies it to every subject: that is what keeps the assembled table the
    same size whatever is adjusted for. The transfer assumes the site effect is
    the same in complete and incomplete cases. Missingness here is emphatically
    not random -- eor is unrecorded for all of TCGA, mgmt for all of RHUH, kps for
    all of UCSF and LUMIERE -- so the assumption is doing real work, and the
    per-group missingness columns of the balance table are the evidence a reader
    weighs it against.

    Returns a dict: logHR, HR, se, ci95 (on the LOG-HR scale), p, n, events,
    n_total, ph_p, covariates,
    columns, notes, reason. `reason` is non-empty (and logHR None) when the model
    could not be fit.
    """
    out = dict(logHR=None, HR=None, se=None, ci95=(None, None), p=None,
               n=0, events=0, n_total=len(data), ph_p=None,
               covariates=list(covariates), columns=[], notes=[], reason=None)

    duration = pd.to_numeric(data[duration_col], errors="coerce")
    status = pd.to_numeric(data[status_col], errors="coerce")
    # Cox needs a strictly positive duration; a recorded zero survives a NaN check
    usable = duration.notna() & status.notna() & (duration > 0)
    dropped = int((~usable).sum())
    if dropped:
        out["notes"].append(f"{dropped} subject(s) without a usable survival time excluded.")

    design, blocks, notes = build_site_design(data, covariates)
    out["notes"].extend(notes)
    out["columns"] = list(design.columns)

    frame = pd.concat([design, data[[site_col]], duration.rename(duration_col),
                       status.rename(status_col)], axis=1).loc[usable]
    if strata_col is not None:
        frame = frame.join(data[[strata_col]])
    before = len(frame)
    frame = frame.dropna()
    if before - len(frame):
        out["notes"].append(
            f"{before - len(frame)} subject(s) dropped from the model for an "
            f"incomplete covariate (the coefficient is still applied to all "
            f"{out['n_total']})."
        )

    out["n"], out["events"] = len(frame), int(frame[status_col].sum())
    if frame[site_col].nunique() < 2:
        out["reason"] = "only one site group survives the complete-case restriction"
        return out
    if out["events"] < 2:
        out["reason"] = f"too few events ({out['events']}) to fit"
        return out

    model, reason = fit_cox(frame, duration_col, status_col,
                            strata=[strata_col] if strata_col else None)
    if model is None:
        out["reason"] = reason
        return out

    # By name, never positionally: the design has more than one column now
    summary = model.summary.loc[site_col]
    out.update(
        logHR=float(model.params_[site_col]),
        HR=float(summary["exp(coef)"]),
        se=float(summary["se(coef)"]),
        ci95=(float(summary["coef lower 95%"]), float(summary["coef upper 95%"])),
        p=float(summary["p"]),
    )
    # Re-checked for every adjusted model: nothing guarantees the adjusted site
    # term satisfies proportional hazards because the crude one did
    try:
        ph = proportional_hazard_test(model, frame, time_transform="rank")
        out["ph_p"] = float(ph.summary.loc[site_col, "p"])
    except Exception:
        out["ph_p"] = None
    return out


def adjustment_ladder(data, covariates, duration_col="OS (days)", status_col="status",
                      site_col="site"):
    """Crude -> progressively adjusted site log-HR, every rung on ONE sample.

    Args:
        data: Table to fit every rung on.
        covariates: Keys of ADJUSTMENT_COVARIATES the ladder walks through, one
            rung each, entered in ADJUSTMENT_ORDER.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        site_col: Column holding the 0/1 site indicator.

    Each rung is fitted on the subjects reporting *every* covariate in the full
    ladder, so the rows differ only in what is adjusted for and never in who is in
    the model. That fixed sample is what makes the shrinkage attributable to
    case-mix rather than to a change of population.

    This is the opposite invariant from the complete-case ladder in the analysis
    pipeline's cohort description, which deliberately lets the sample shrink down
    each rung to show what each covariate *costs*. The two answer different
    questions and both are worth having; a "crude (all subjects)" row is reported
    alongside so the cost of restricting to complete cases stays visible.
    """
    rows = []
    full = estimate_site_logHR(data, (), duration_col, status_col, site_col)
    rows.append(("crude (all subjects)", full))

    # The sample every remaining rung is held to
    design, _, _ = build_site_design(data, covariates)
    duration = pd.to_numeric(data[duration_col], errors="coerce")
    status = pd.to_numeric(data[status_col], errors="coerce")
    complete = (design.notna().all(axis=1) & duration.notna() & status.notna()
                & (duration > 0))
    sample = data.loc[complete]

    rows.append(("crude (complete-case sample)",
                 estimate_site_logHR(sample, (), duration_col, status_col, site_col)))

    entered = []
    for key in [k for k in ADJUSTMENT_ORDER if k in covariates]:
        entered.append(key)
        rows.append((f"+ {' + '.join(entered)}",
                     estimate_site_logHR(sample, entered, duration_col, status_col, site_col)))

    baseline = rows[1][1]["logHR"]
    table = []
    for label, res in rows:
        removed = (np.nan if not baseline or res["logHR"] is None
                   else 100.0 * (1.0 - res["logHR"] / baseline))
        table.append(dict(
            model=label, n=res["n"], events=res["events"],
            logHR=res["logHR"], HR=res["HR"],
            logHR_ci_low=res["ci95"][0], logHR_ci_high=res["ci95"][1], p=res["p"],
            pct_of_crude_removed=removed, ph_p=res["ph_p"], reason=res["reason"],
        ))
    return pd.DataFrame(table)


def standardized_mean_difference(values, group):
    """Absolute SMD between the two levels of a {0, 1} group indicator.

    Args:
        values: Covariate to compare, continuous or a 0/1 indicator.
        group: The 0/1 group indicator, aligned on the same index.

    (m1 - m0) / sqrt((s0^2 + s1^2) / 2); for an indicator the same formula with
    s^2 = p(1-p), which is the binary case of the same pooled-variance definition.
    Unlike a p-value it does not shrink as n grows, so it measures imbalance rather
    than the power to detect it. |SMD| above 0.10 is the conventional threshold for
    an imbalance worth adjusting for.
    """
    a = pd.to_numeric(values[group == 0], errors="coerce").dropna()
    b = pd.to_numeric(values[group == 1], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return np.nan if pooled == 0 else abs(b.mean() - a.mean()) / pooled


def site_balance_table(data, site_col="site", covariates=ADJUSTMENT_ORDER):
    """Distribution of every covariate across the two site groups.

    Args:
        data: Table holding the site column and the covariate columns.
        site_col: Column holding the 0/1 site indicator.
        covariates: Keys of ADJUSTMENT_COVARIATES to describe.

    One row per continuous covariate and per level of a categorical one, with
    n/mean(SD) or n(%) per group, the SMD, and -- in its own columns -- the
    percentage missing per group. Differential missingness sits next to the
    imbalance on purpose: a covariate one site never records cannot be balanced by
    any adjustment, so a reader can see which rungs of the ladder rest on a
    within-site-1 comparison only.
    """
    group = pd.to_numeric(data[site_col], errors="coerce")
    rows = []
    for key in [k for k in ADJUSTMENT_ORDER if k in covariates]:
        spec = ADJUSTMENT_COVARIATES[key]
        if spec["column"] not in data.columns:
            continue
        values = pd.to_numeric(data[spec["column"]], errors="coerce")
        miss = {s: 100.0 * values[group == s].isna().mean() for s in (0, 1)}

        if spec["kind"] == "continuous":
            summaries = {
                s: (f"{values[group == s].mean():.1f} ({values[group == s].std():.1f})"
                    if values[group == s].notna().any() else "not recorded")
                for s in (0, 1)
            }
            rows.append(dict(variable=spec["label"], level="mean (SD)",
                             site0=summaries[0], site1=summaries[1],
                             smd=standardized_mean_difference(values, group),
                             pct_missing_site0=miss[0], pct_missing_site1=miss[1]))
            continue

        for level, name in sorted(spec["levels"].items()):
            indicator = pd.Series(np.where(values.isna(), np.nan,
                                           (values == level).astype(float)),
                                  index=values.index)
            cells = {}
            for s in (0, 1):
                observed = indicator[group == s].dropna()
                cells[s] = (f"{int(observed.sum())} ({100.0 * observed.mean():.1f}%)"
                            if len(observed) else "not recorded")
            rows.append(dict(variable=spec["label"], level=name,
                             site0=cells[0], site1=cells[1],
                             smd=standardized_mean_difference(indicator, group),
                             pct_missing_site0=miss[0], pct_missing_site1=miss[1]))
    return pd.DataFrame(rows)


def reverse_km_followup(data, site_labels, site_col="site", duration_col="OS (days)",
                        status_col="status"):
    """Median potential follow-up per site group (Schemper-Smith reverse KM).

    Args:
        data: Table holding the site and survival columns.
        site_labels: {code: name} used to name the rows.
        site_col: Column holding the site indicator; every level present gets a
            row, so this works for more than two groups.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    The Kaplan-Meier estimator refitted with the event indicator flipped, so
    censoring becomes the event. The median of the *observed* times is dominated
    by patients who died early and understates how long a group was actually
    watched. This matters here because UCSF censors a large fraction of its
    subjects and the remaining cohorts almost none -- a gap that can by itself
    produce a log-rank difference having nothing to do with prognosis. A log-rank
    test on the censoring distribution is reported next to it.
    """
    rows = []
    censor_stats, censor_groups = [], []
    for i, site in enumerate(sorted(data[site_col].dropna().unique())):
        block = data[data[site_col] == site]
        block = block[block[duration_col].notna() & block[status_col].notna()]
        n, events = len(block), int((block[status_col] == 1).sum())
        censored = n - events

        time, surv, _ = km_curve(block, duration_col, status_col)
        median_os = km_median(time, surv)

        if censored:
            flipped = block.copy()
            flipped[status_col] = 1 - flipped[status_col]
            ctime, csurv, _ = km_curve(flipped, duration_col, status_col)
            median_fu = km_median(ctime, csurv)
            note = "" if censored / n >= 0.10 else "* under 10% censored"
        else:
            median_fu, note = np.nan, "no censoring -- follow-up not estimable"

        rows.append(dict(group=site_labels.get(site, site), n=n, events=events,
                         pct_censored=100.0 * censored / n if n else np.nan,
                         median_OS_days=median_os, median_followup_days=median_fu,
                         max_followup_days=block[duration_col].max(), note=note))
        censor_stats.extend(zip(block[status_col] == 0, block[duration_col].values))
        censor_groups.extend([i + 1] * n)

    table = pd.DataFrame(rows)
    if len(set(censor_groups)) >= 2:
        structured = as_structured([e for e, _ in censor_stats], [t for _, t in censor_stats])
        chi2, p_val = compare_survival(structured, censor_groups, return_stats=False)
        table.attrs["censoring_logrank"] = (float(chi2), float(p_val))
    return table


def truncation_sensitivity(data, horizons_months, covariates=(), site_col="site",
                           duration_col="OS (days)", status_col="status"):
    """Re-estimate the site log-HR under a common administrative horizon.

    Args:
        data: Table holding the site and survival columns.
        horizons_months: Horizons to truncate at, in months. A horizon beyond the
            last observed time of either group is skipped rather than reported as
            an untruncated estimate, and an untruncated row is always appended.
        covariates: Keys of ADJUSTMENT_COVARIATES for the adjusted column; empty
            reports the crude estimate only.
        site_col: Column holding the 0/1 site indicator.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Everybody is censored at the horizon (t' = min(t, H), status' = 0 when t > H),
    which makes the two site groups equally observed by construction. If the site
    difference were an artefact of one cohort being watched differently, the
    estimate would move as the horizon tightens; a log-HR stable across horizons
    is evidence that it is not.
    """
    rows = []
    for horizon in list(horizons_months) + [None]:
        block = data.copy()
        if horizon is not None:
            limit = horizon * daysXmonth
            if (block.groupby(site_col)[duration_col].max() < limit).any():
                continue  # a group is no longer under observation at this horizon
            block[status_col] = np.where(block[duration_col] > limit, 0, block[status_col])
            block[duration_col] = np.minimum(block[duration_col], limit)
        crude = estimate_site_logHR(block, (), duration_col, status_col, site_col)
        adjusted = (estimate_site_logHR(block, covariates, duration_col, status_col, site_col)
                    if covariates else None)
        rows.append(dict(
            horizon_months="none" if horizon is None else horizon,
            n=crude["n"], events=crude["events"],
            logHR_crude=crude["logHR"], p_crude=crude["p"],
            logHR_adjusted=adjusted["logHR"] if adjusted else np.nan,
            p_adjusted=adjusted["p"] if adjusted else np.nan,
            n_adjusted=adjusted["n"] if adjusted else np.nan,
        ))
    return pd.DataFrame(rows)


def plot_adjustment_ladder(ladder, RESULTS, stem, formats, show_plot=True,
                           title="Site log-HR under progressive adjustment"):
    """Forest plot of the ladder: log-HR with 95% CI, one row per rung.

    Args:
        ladder: Table from `adjustment_ladder`; rungs that could not be fitted
            are dropped, and nothing is drawn if none survive.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.
        title: Figure title.
    """
    rows = ladder[ladder["logHR"].notna()]
    if rows.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 0.6 * len(rows) + 2))
    y = np.arange(len(rows))[::-1]
    ax.errorbar(rows["logHR"], y,
                xerr=[rows["logHR"] - rows["logHR_ci_low"],
                      rows["logHR_ci_high"] - rows["logHR"]],
                fmt="o", color="tab:purple", ecolor="gray", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  (n={n})" for m, n in zip(rows["model"], rows["n"])], fontsize=9)
    ax.set_xlabel("log hazard ratio of site=1 vs site=0 (95% CI)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, RESULTS, stem, formats)
    plt.show() if show_plot else plt.close(fig)


def report_site_diagnostics(database, args, RESULTS, site_labels, formats, show_plot=True):
    """Everything a reader needs to judge whether 'site' is really case-mix.

    Args:
        database: Assembled table, before the correction is applied.
        args: Parsed command line. Read here: `ladder_covariates` (falling back to
            `adjust_covariates`, then to DEFAULT_LADDER) and `truncate_months`
            (falling back to 12 24 36 48).
        RESULTS: Results directory; tables and the figure land in its OS-stats/.
        site_labels: {code: name} used to name the site groups in every table.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.

    Returns the tables as a dict, which is also what gets written as CSVs.

    Writes as CSVs under OS-stats/, and into the report, the balance and
    missingness table between the two site groups, the same-sample adjustment
    ladder, the reverse-Kaplan-Meier follow-up comparison and the
    administrative-truncation sensitivity. These are supplement tables rather than
    lines in a log, so they are saved as well as reported.
    """
    ladder_covariates = args.ladder_covariates or list(args.adjust_covariates) or DEFAULT_LADDER
    horizons = args.truncate_months if args.truncate_months is not None else [12, 24, 36, 48]
    out = {}

    section("SITE DIAGNOSTICS: is the survival difference case-mix or entry point?")
    REPORT.heading("Site diagnostics: case-mix or entry point?")
    REPORT.paragraph(
        "Whether the survival difference between the site groups is a difference "
        "in who the patients are rather than in where they entered. Covariates "
        f"walked: {', '.join(ladder_covariates)}. Site groups: "
        + "; ".join(f"{k} = {v}" for k, v in site_labels.items()) + ".")
    print(f"Covariates walked: {', '.join(ladder_covariates)}")
    print(f"Site groups: " + "; ".join(f"{k} = {v}" for k, v in site_labels.items()))

    subsection("Balance and missingness between site groups")
    balance = site_balance_table(database, covariates=ladder_covariates)
    out["balance"] = balance
    if not balance.empty:
        REPORT.heading("Balance and missingness between site groups", level=3)
        REPORT.paragraph(
            "|SMD| > 0.10 marks an imbalance worth adjusting for. A covariate a "
            "site never records cannot be balanced by any adjustment, so the two "
            "pct_missing columns are read alongside the SMD.")
        REPORT.table(balance, float_format=lambda v: f"{v:.3f}")
        print(balance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n|SMD| > 0.10 marks an imbalance worth adjusting for. A covariate a site")
        print("never records cannot be balanced by any adjustment -- read the two")
        print("pct_missing columns alongside the SMD.")

    subsection("Adjustment ladder (every rung on the same complete-case sample)")
    ladder = adjustment_ladder(database, ladder_covariates)
    out["ladder"] = ladder
    REPORT.heading("Adjustment ladder", level=3)
    REPORT.paragraph(
        "Every rung is fitted on the subjects reporting every covariate in the "
        "ladder, so the rows differ only in what is adjusted for and never in who "
        "is in the model. pct_of_crude_removed is measured against the "
        "complete-case crude rung, not the all-subjects one.")
    REPORT.table(ladder, float_format=lambda v: f"{v:.4f}")
    print(ladder.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    failed = ladder[ladder["reason"].notna() & (ladder["reason"] != "None")]
    if len(failed):
        print(f"\nWARNING: {len(failed)} of {len(ladder)} rungs could not be fitted. The most")
        print(f"         common reason was: {failed['reason'].iloc[0]}")
        print("         A covariate a whole cohort never records shrinks the complete-case")
        print("         sample onto a single site, leaving nothing to compare. KPS does this")
        print("         (UCSF and LUMIERE record none), which is why it is not in the default")
        print("         ladder. Drop it, or read the balance table's missingness columns.")
    print("\nNote: these are lifelines (Efron) fits; the coefficient actually applied to")
    print("the survival times comes from scikit-survival (Breslow). With many tied")
    print("survival days the two differ in the 2nd-3rd decimal. That is expected.")
    plot_adjustment_ladder(ladder, RESULTS, "Site-diagnostics_adjustment-ladder",
                           formats, show_plot=show_plot)

    subsection("Follow-up by site group (reverse Kaplan-Meier)")
    followup = reverse_km_followup(database, site_labels)
    out["followup"] = followup
    REPORT.heading("Follow-up by site group (reverse Kaplan-Meier)", level=3)
    REPORT.paragraph(
        "Median potential follow-up, i.e. the Kaplan-Meier estimator refitted with "
        "the event indicator flipped. A difference here is a difference in how long "
        "the groups were watched, not in how long they survived.")
    REPORT.table(followup, float_format=lambda v: f"{v:.1f}")
    print(followup.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if "censoring_logrank" in followup.attrs:
        chi2, p_val = followup.attrs["censoring_logrank"]
        print(f"\nLog-rank on the censoring distribution: chi2 = {chi2:.4f}, p = {fmt_p(p_val).strip()}")
        print("A difference here is a difference in how long the groups were watched,")
        print("not in how long they survived.")

    subsection("Administrative truncation at a common horizon")
    truncation = truncation_sensitivity(database, horizons, ladder_covariates)
    out["truncation"] = truncation
    REPORT.heading("Administrative truncation at a common horizon", level=3)
    REPORT.paragraph(
        "Everybody is censored at the horizon, which makes the two groups equally "
        "observed by construction. An estimate that barely moves across horizons "
        "is evidence the difference is not an artefact of differing follow-up.")
    REPORT.table(truncation, float_format=lambda v: f"{v:.4f}")
    print(truncation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTruncating makes both groups equally observed by construction. An estimate")
    print("that barely moves across horizons is evidence the difference is not an")
    print("artefact of differing follow-up.")

    for name, table in out.items():
        if table is not None and not table.empty:
            table.to_csv(f"{RESULTS}/OS-stats/Site-diagnostics_{name}.csv", index=False)
    print(f"\nDiagnostic tables written to {RESULTS}/OS-stats/Site-diagnostics_*.csv")
    return out


# Thresholds the recommendation argues against. They are conventional rather than
# derived, and they are quoted in the report next to the number they judge, so a
# reader who disagrees with one can see exactly which sentence it produced.
ALPHA = 0.05            # a site term whose CI covers 0 is not evidence of a site effect
SMD_IMBALANCE = 0.10    # |SMD| above this is an imbalance worth adjusting for
HORIZON_SPREAD = 0.10   # log-HR range across truncation horizons, in log-HR units


def _final_rung(ladder):
    """The most adjusted rung of the ladder that was actually fitted.

    Args:
        ladder: Table from `adjustment_ladder`.

    Returns the last row whose model was fitted and whose name starts with "+",
    i.e. the fully adjusted one, or None when no adjusted rung could be fitted.
    """
    if ladder is None or ladder.empty:
        return None
    fitted = ladder[ladder["logHR"].notna() & ladder["model"].str.startswith("+")]
    return None if fitted.empty else fitted.iloc[-1]


def recommend_outcome_column(provenance, diagnostics, site_labels):
    """Recommend the raw or the corrected survival column, from this run's evidence.

    Args:
        provenance: The dict `apply_site_correction` built, which says what was
            actually applied to the table.
        diagnostics: The tables `report_site_diagnostics` returned.
        site_labels: {code: name}, for naming the groups in the argument.

    Returns (verdict, headline, evidence, caveats): `verdict` is one of "raw",
    "corrected" or "undetermined"; `evidence` and `caveats` are lists of strings.

    The argument is always the same one. A site coefficient is worth dividing out
    of the outcome only if it survives the case-mix that could have produced it.
    The adjustment ladder answers that on every run, whatever was applied, so the
    recommendation can disagree with what the correction column actually holds --
    and says so when it does.
    """
    ladder = diagnostics.get("ladder")
    balance = diagnostics.get("balance")
    followup = diagnostics.get("followup")
    truncation = diagnostics.get("truncation")

    evidence, caveats = [], []
    groups = f"{site_labels.get(0, 0)} (site 0) vs {site_labels.get(1, 1)} (site 1)"

    if provenance.get("mode") == "identity":
        return ("raw",
                "Use the raw survival times: no correction was estimated.",
                [f"Only one site group is present ({site_labels.get(0, 'the selection')}), "
                 f"so 'OS (days) - corrected' is a copy of 'OS (days)' and the two "
                 f"columns are interchangeable."],
                [])

    rung = _final_rung(ladder)
    crude = ladder[ladder["model"] == "crude (all subjects)"].iloc[0] if ladder is not None \
        and not ladder.empty and (ladder["model"] == "crude (all subjects)").any() else None

    if crude is not None and pd.notna(crude["logHR"]):
        evidence.append(
            f"Crude site effect, {groups}: log HR {crude['logHR']:.4f} "
            f"(HR {crude['HR']:.3f}, p = {fmt_p(crude['p']).strip()}) on all "
            f"{int(crude['n'])} subjects.")

    if rung is None:
        caveats.append(
            "No adjusted rung of the ladder could be fitted, so there is no evidence "
            "here that separates case-mix from entry point. Treat the correction as "
            "unverified.")
        return ("undetermined",
                "Undetermined: the adjusted models could not be fitted on this selection.",
                evidence, caveats)

    covers_zero = (pd.notna(rung["logHR_ci_low"]) and pd.notna(rung["logHR_ci_high"])
                   and rung["logHR_ci_low"] <= 0.0 <= rung["logHR_ci_high"])
    evidence.append(
        f"Adjusted site effect ({rung['model'].lstrip('+ ')}): log HR "
        f"{rung['logHR']:.4f} (95% CI {rung['logHR_ci_low']:.4f} to "
        f"{rung['logHR_ci_high']:.4f}, p = {fmt_p(rung['p']).strip()}), fitted on "
        f"{int(rung['n'])} subjects with {int(rung['events'])} events.")
    if pd.notna(rung["pct_of_crude_removed"]):
        evidence.append(
            f"Case-mix accounts for {rung['pct_of_crude_removed']:.1f}% of the site "
            f"effect measured on the same complete-case sample.")

    # The decision itself
    if covers_zero:
        verdict = "raw"
        headline = (
            "Use the RAW survival times, and adjust or stratify for cohort "
            "downstream: once case-mix is held fixed, the site effect is no longer "
            "distinguishable from zero.")
        evidence.append(
            f"The adjusted confidence interval covers 0 at alpha = {ALPHA}, so what "
            f"the crude comparison showed is explained by who the patients are, not "
            f"by where they entered. Rescaling the outcome by it would remove "
            f"prognostic signal that belongs to the covariates, and a downstream "
            f"model that also adjusts for them would remove it twice.")
    else:
        verdict = "corrected"
        headline = (
            "The CORRECTED survival times are defensible: a site effect survives "
            "adjustment for case-mix.")
        evidence.append(
            f"The adjusted confidence interval excludes 0 at alpha = {ALPHA}, so a "
            f"difference remains after case-mix is held fixed. Stratifying the "
            f"baseline hazard by cohort is the equivalent alternative and needs no "
            f"rescaling at all; pick one of the two, never both.")

    # Caveats, which qualify either verdict
    if pd.notna(rung["ph_p"]) and rung["ph_p"] < ALPHA:
        caveats.append(
            f"Proportional hazards is rejected for the adjusted site term "
            f"(p = {fmt_p(rung['ph_p']).strip()}). A single multiplicative factor "
            f"is then the wrong description of the difference at every follow-up "
            f"time, and stratification is the safer route whatever is decided here.")
    if int(rung["n"]) < int(crude["n"] if crude is not None else rung["n"]):
        lost = int(crude["n"]) - int(rung["n"])
        caveats.append(
            f"The adjusted estimate rests on {int(rung['n'])} of {int(crude['n'])} "
            f"subjects; {lost} were dropped for an incomplete covariate. Missingness "
            f"here is cohort-structured, not random, so the complete cases "
            f"over-represent the cohorts that record everything.")
    if balance is not None and not balance.empty and balance["smd"].notna().any():
        worst = balance.loc[balance["smd"].idxmax()]
        if worst["smd"] > SMD_IMBALANCE:
            caveats.append(
                f"The largest imbalance between the groups is "
                f"{worst['variable']} = {worst['level']} (|SMD| {worst['smd']:.2f}, "
                f"above {SMD_IMBALANCE}), which is the case-mix the adjustment is "
                f"working against.")
    if followup is not None and "censoring_logrank" in followup.attrs:
        chi2, p_censor = followup.attrs["censoring_logrank"]
        if p_censor < ALPHA:
            caveats.append(
                f"The censoring distributions differ between the groups "
                f"(log-rank p = {fmt_p(p_censor).strip()}): they were watched for "
                f"different lengths of time, which can produce a survival "
                f"difference on its own. The truncation table is the check on that.")
    if truncation is not None and not truncation.empty:
        finite = truncation[truncation["horizon_months"] != "none"]["logHR_crude"].dropna()
        if len(finite) >= 2:
            spread = float(finite.max() - finite.min())
            (evidence if spread <= HORIZON_SPREAD else caveats).append(
                f"Across the truncation horizons the crude log HR spans {spread:.4f} "
                f"in log-HR units, {'within' if spread <= HORIZON_SPREAD else 'beyond'}"
                f" the {HORIZON_SPREAD} considered stable.")

    # The applied column need not agree with the recommendation
    if verdict == "raw" and provenance.get("mode", "").startswith("rescale"):
        caveats.append(
            f"This run still wrote a correction: log HR {provenance['logHR']:.6f} "
            f"({provenance['mode']}). The corrected column is therefore present but "
            f"not recommended for this selection; nothing downstream has to use it.")
    if verdict == "corrected" and provenance.get("mode") == "rescale-crude":
        caveats.append(
            "The coefficient applied to the table is the crude one, fitted without "
            "covariates. Re-run with --adjust-covariates so the column carries the "
            "adjusted coefficient the recommendation is based on.")
    return verdict, headline, evidence, caveats


def report_recommendation(provenance, diagnostics, site_labels):
    """Print the recommendation and add it to the report as the closing section.

    Args:
        provenance: The dict `apply_site_correction` built.
        diagnostics: The tables `report_site_diagnostics` returned.
        site_labels: {code: name}, for naming the groups.

    Returns the verdict string, which also lands in the provenance JSON.
    """
    verdict, headline, evidence, caveats = recommend_outcome_column(
        provenance, diagnostics, site_labels)

    section("RECOMMENDATION: raw or corrected survival times?")
    print(headline + "\n")
    for line in evidence:
        print(f"  - {line}")
    if caveats:
        print("\n  Caveats:")
        for line in caveats:
            print(f"  - {line}")
    column = {"raw": "'OS (days)'", "corrected": "'OS (days) - corrected'"}.get(
        verdict, "neither column without further checks")
    print(f"\n  Verdict: {verdict.upper()} -- downstream analyses should read {column}.")

    REPORT.heading("Recommendation: raw or corrected survival times?")
    REPORT.callout(headline + f"  Downstream analyses should read {column}.")
    REPORT.paragraph(
        "A site coefficient is worth dividing out of the outcome only if it "
        "survives the case-mix that could have produced it. That is what the "
        "adjustment ladder above tests, on every run, whatever coefficient was "
        "applied -- so this section can disagree with the column the table "
        "actually carries, and says so when it does.")
    REPORT.heading("What this run found", level=3)
    for line in evidence:
        REPORT.paragraph(line)
    if caveats:
        REPORT.heading("What qualifies it", level=3)
        for line in caveats:
            REPORT.paragraph(line)
    REPORT.paragraph(
        "Whichever column is used, use one of the two remedies and not both: "
        "rescaling the outcome and stratifying the baseline hazard by cohort are "
        "alternatives, and applying them together corrects the same difference "
        "twice.")
    return verdict


def apply_site_correction(database, args, RESULTS, site_labels, formats):
    """Estimate the site effect, write the corrected survival column, and say so.

    Args:
        database: Assembled table; it gains the corrected column and is returned.
        args: Parsed command line. Read here: `adjust_covariates`, `n_perms` and
            `show`.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        site_labels: {code: name} used to name the two site groups.
        formats: Figure formats to write.


    Returns (database, provenance). The corrected column is always written,
    whatever happens -- a correction that could not be estimated falls back to the
    crude one, and a single-site selection to the identity -- so no downstream
    consumer ever meets a missing column.
    """
    sites = sorted(int(s) for s in database["site"].dropna().unique())
    # exp(logHR * site) silently becomes exp(2 * logHR) for a site code of 2
    assert set(sites) <= {0, 1}, f"site codes must be 0/1, got {sites}"

    provenance = dict(
        adjusted_for=list(args.adjust_covariates), design_columns=[],
        missing_strategy="complete-case", logHR=None, logHR_crude=None,
        n_model=None, events_model=None, n_applied=len(database),
        ph_test_p=None, notes=[],
    )

    section("SITE EFFECT")
    REPORT.heading("Site effect: the coefficient applied to the survival times")
    if len(sites) < 2:
        only = site_labels.get(sites[0], sites[0]) if sites else "none"
        logHR_site = 0.0
        provenance.update(mode="identity", logHR=0.0, notes=[
            f"Only one site group in the selected cohorts ({only}); no correction applied."
        ])
        print(f"Only one site group present ({only}). "
              f"'OS (days) - corrected' is a copy of 'OS (days)'.")
    else:
        analysable = database.dropna(subset=["OS (days)", "status"])
        n_site = analysable.groupby("site").size().reindex(sites, fill_value=0)

        # `adjusted` is estimated first so the figure can be drawn with the
        # coefficient that will actually be written to the file
        adjusted = None
        if args.adjust_covariates:
            adjusted = estimate_site_logHR(database, args.adjust_covariates)

        logHR_crude = inspect_survival_diffs_in_paired_cohorts(
            full_data=database,
            cohorts=list(sites),
            RESULTS=RESULTS,
            name_cohort=site_labels,
            colors=["tab:green", "salmon"],
            N_cohorts=[int(n_site.loc[s]) for s in sites],
            covariate_col="site",
            n_perms=args.n_perms,
            formats=formats,
            show_plot=args.show,
            logHR_override=(adjusted or {}).get("logHR"),
            # Only when that override exists: if the adjusted fit failed the crude
            # coefficient is the one applied below, and the panel must diagnose it
            adjust_covariates=(list(args.adjust_covariates)
                               if adjusted and adjusted["logHR"] is not None else ()),
        )
        provenance["logHR_crude"] = float(logHR_crude)

        if adjusted is None:
            logHR_site = logHR_crude
            provenance.update(mode="rescale-crude", logHR=float(logHR_site))
            print(f"\nEffect of 'site' (log HR) {logHR_site} in units of "
                  f"{site_labels.get(0)}/{site_labels.get(1)}")
        elif adjusted["logHR"] is None:
            logHR_site = logHR_crude
            provenance.update(mode="rescale-crude", logHR=float(logHR_site),
                              notes=[f"Adjusted model could not be fit ({adjusted['reason']}); "
                                     f"fell back to the crude estimate."])
            print(f"\nWARNING: the adjusted site model could not be fit "
                  f"({adjusted['reason']}).")
            print(f"         Falling back to the crude log HR {logHR_crude}.")
        else:
            logHR_site = adjusted["logHR"]
            provenance.update(
                mode="rescale-adjusted", logHR=float(logHR_site),
                design_columns=adjusted["columns"], n_model=adjusted["n"],
                events_model=adjusted["events"], ph_test_p=adjusted["ph_p"],
                HR=adjusted["HR"], se=adjusted["se"],
                ci95_logHR=list(adjusted["ci95"]),
                ci95_HR=[float(np.exp(adjusted["ci95"][0])), float(np.exp(adjusted["ci95"][1]))],
                p=adjusted["p"],
                notes=adjusted["notes"],
            )
            removed = (100.0 * (1.0 - logHR_site / logHR_crude)) if logHR_crude else np.nan
            lo, hi = adjusted["ci95"]
            print(f"\nEffect of 'site' adjusted for {', '.join(args.adjust_covariates)}:")
            print(f"  log HR = {logHR_site:.6f}  (95% CI {lo:.4f} to {hi:.4f})")
            print(f"  HR     = {adjusted['HR']:.4f}  (95% CI {np.exp(lo):.4f} to "
                  f"{np.exp(hi):.4f}, p = {fmt_p(adjusted['p']).strip()})")
            print(f"  crude log HR = {logHR_crude:.6f}  -->  "
                  f"{removed:.1f}% of it is explained by the covariates")
            print(f"  estimated on {adjusted['n']} of {len(database)} subjects "
                  f"({adjusted['events']} events); applied to all {len(database)}")
            print(f"  proportional-hazards test on the adjusted site term: "
                  f"p = {fmt_p(adjusted['ph_p']).strip()}")
            for note in adjusted["notes"]:
                print(f"  - {note}")

    database["OS (days) - corrected"] = database["OS (days)"] * np.exp(logHR_site * database["site"])
    # Written so the correction is exactly invertible per subject without the sidecar
    database["site correction factor"] = np.exp(logHR_site * database["site"])
    provenance["applied_as"] = "OS (days) - corrected = OS (days) * exp(logHR * site)"
    return database, provenance


def write_provenance(provenance, RESULTS, stem):
    """Record how the site correction was obtained, next to the table it produced.

    Args:
        provenance: The dict built by `apply_site_correction`.
        RESULTS: Directory to write into.
        stem: Base name of the assembled table, which the JSON is named after.


    A separate `<stem>_site-correction.json`, never keys-maps.json: the analysis
    pipeline's decoder iterates every top-level entry of that file and calls
    .items() on it, so anything that is not a {label: code} mapping breaks it.
    """
    path = f"{RESULTS}/{stem}_site-correction.json"
    with open(path, "w") as handle:
        json.dump(provenance, handle, indent=4, default=str)
    print(f"Site-correction provenance written to {path}")
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class Tee:
    """Write to two streams at once, so a run is both shown and stored.

    Args:
        stream: The original stream, kept for `isatty` as well as for writing.
        handle: Open file the same output is mirrored into.
    """

    def __init__(self, stream, handle):
        """Store the `stream` and the `handle` the class docstring describes."""
        self.stream = stream
        self.handle = handle

    def write(self, data):
        """Write to both streams.

        Args:
            data: Text to write. Its length is returned, as `write` must.
        """
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        """Flush both streams. Takes no arguments."""
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        """Whether the original stream is a terminal. Takes no arguments."""
        return self.stream.isatty()


# Set by main(). Read by `announce`, which has to know whether a line it puts on
# the terminal is already going there through stdout.
ECHO_TO_TERMINAL = False


def announce(text):
    """Put one line on the terminal even while stdout is going to the log file.

    Args:
        text: The line to show.

    A run writes its output to the log rather than the screen, so the few lines
    that say what was produced and where have to bypass that redirection. Under
    --verbose stdout reaches the terminal anyway and this would print each of them
    twice, so it then only writes to the log.
    """
    print(text)
    if not ECHO_TO_TERMINAL:
        print(text, file=sys.__stdout__, flush=True)


@contextlib.contextmanager
def tee_stdout(path, echo=False):
    """Send everything printed to `path`, and to the terminal only if `echo`.

    Args:
        path: File the run is written to. Always a path, never None: it is the
            only record of the run, since the terminal no longer gets one.
        echo: Also keep writing to the terminal, as --verbose asks.

    Progress bars are unaffected either way: tqdm writes to stderr, which is not
    redirected, so a long run still shows that it is alive.
    """
    original = sys.stdout
    with open(path, "w") as handle:
        sys.stdout = Tee(original, handle) if echo else handle
        try:
            yield
        finally:
            sys.stdout = original


def resolve_log_path(log_arg, RESULTS):
    """The file the run is written to.

    Args:
        log_arg: The --log value: None or "" for the default name, otherwise the
            path asked for.
        RESULTS: Directory a relative path is resolved under.

    Never None. The terminal shows only what `announce` puts there, so a run that
    wrote no log would leave no record of itself at all.
    """
    path = log_arg or "createDatabase_log.txt"
    return path if os.path.isabs(path) else f"{RESULTS}/{path}"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv: Argument list to parse, or None to read sys.argv. Passing a list
            makes the script callable from another module or a test.

    Each flag's own help text below is the documentation of what it does. The one
    check done here rather than by argparse is that --site-reference only names
    cohorts --cohorts actually selects.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("main_dir", type=str,
                        help="Root directory holding the per-cohort folders.")
    parser.add_argument("results_dir", type=str,
                        help="Output directory. Relative paths are resolved under MAIN_DIR.")
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS.keys()),
                        choices=list(COHORTS.keys()),
                        help="Cohorts to pool (default: all of them).")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Base name of the assembled table "
                             "(default: data-clinical_TD-tissues_<N>-cohorts).")
    parser.add_argument("--idh", type=str, default="WT",
                        help="IDH status the per-cohort pipeline was run with.")
    parser.add_argument("--grade", type=str, default="IV",
                        help="WHO grade the UCSF-PDGM pipeline was run with.")
    parser.add_argument("--stream-th", type=int, default=0,
                        help="Minimum streamline density used when extracting the indices.")
    parser.add_argument("--n-perms", type=int, default=1000,
                        help="Permutations for the C-index test of the site effect.")
    parser.add_argument("--site-reference", nargs="+", default=None, metavar="COHORT",
                        choices=list(COHORTS.keys()),
                        help="Cohorts forming the reference group (site 0), whose "
                             "survival times are left untouched; every other selected "
                             "cohort becomes site 1 and is rescaled by the fitted "
                             "coefficient. Default: the `site` field of COHORTS, which "
                             "puts UCSF alone in the reference group. Naming every "
                             "selected cohort leaves a single group and applies no "
                             "correction.")
    parser.add_argument("--pairwise", action="store_true",
                        help="Also inspect every pair of cohorts, not only the site effect. "
                             "Slow: it refits the permutation test for each pair.")
    parser.add_argument("--adjust-covariates", nargs="*", default=[], metavar="COV",
                        choices=list(ADJUSTMENT_COVARIATES),
                        help="Clinical covariates the site-effect model adjusts for "
                             "(any of: age sex eor mgmt kps). Default: none, i.e. the "
                             "crude site effect. The estimate is made on the subjects "
                             "reporting all of them and applied to every subject, so "
                             "the assembled table never shrinks.")
    parser.add_argument("--ladder-covariates", nargs="*", default=None, metavar="COV",
                        choices=list(ADJUSTMENT_COVARIATES),
                        help="Covariates the diagnostics ladder walks through "
                             "(default: --adjust-covariates if given, else age sex eor "
                             "mgmt; KPS is left out because two cohorts record none).")
    parser.add_argument("--truncate-months", nargs="*", type=float, default=None, metavar="M",
                        help="Horizons (months) for the diagnostics' administrative-truncation check "
                             "(default: 12 24 36 48, kept only where both site groups "
                             "are still under observation).")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "svg", "both"],
                        help="Figure format (default: pdf).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures as well as writing them to disk.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the permutation and bootstrap draws.")
    parser.add_argument("--log", nargs="?", const="", default=None, metavar="FILE",
                        help="Name of the text file the run is written to "
                             "(default: <RESULTS_DIR>/createDatabase_log.txt). Relative "
                             "paths are resolved under RESULTS_DIR. The run is always "
                             "logged; this only renames the file.")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print the run to the terminal. Off by default: the "
                             "log file and the HTML report are the record, and the "
                             "terminal gets only the paths they were written to.")
    args = parser.parse_args(argv)
    # A reference cohort that is not being pooled is a typo, not a no-op: silently
    # ignoring it would assemble the table under a partition nobody asked for
    if args.site_reference is not None:
        unselected = sorted(set(args.site_reference) - set(args.cohorts))
        if unselected:
            parser.error(f"--site-reference names {', '.join(unselected)}, which "
                         f"--cohorts does not select.")
    return args


def main(argv=None):
    """Parse the command line, prepare the output directories and run the assembly.

    Args:
        argv: Argument list to parse, or None to read sys.argv.

    Seeding, the covariate ordering, the output directories and the optional log
    file are settled here, so `assemble_database` can assume all four.
    """
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Fixed entry order, so the ladder is comparable however the flags were typed
    def in_order(keys):
        """Covariate keys in ADJUSTMENT_ORDER, however they were typed.

        Args:
            keys: Covariate keys from the command line, in any order.
        """
        return [k for k in ADJUSTMENT_ORDER if k in set(keys)]

    args.adjust_covariates = in_order(args.adjust_covariates)
    if args.ladder_covariates is not None:
        args.ladder_covariates = in_order(args.ladder_covariates)

    formats = ["pdf", "svg"] if args.format == "both" else [args.format]
    main_dir = os.path.abspath(args.main_dir)
    RESULTS = args.results_dir if os.path.isabs(args.results_dir) else f"{main_dir}/{args.results_dir}"
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(f"{RESULTS}/OS-stats", exist_ok=True)

    global ECHO_TO_TERMINAL
    ECHO_TO_TERMINAL = args.verbose
    log_path = resolve_log_path(args.log, RESULTS)
    args.log_path = log_path  # named in the report, so the reader can find the run

    if not args.verbose:
        print(f"createDatabase.py: pooling {', '.join(args.cohorts)} -> {RESULTS}",
              file=sys.__stdout__, flush=True)
        print(f"  running quietly; the run is written to {log_path} "
              f"(--verbose to watch it here)", file=sys.__stdout__, flush=True)
    with tee_stdout(log_path, echo=args.verbose):
        print(f"# createDatabase.py -- {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# {' '.join(sys.argv)}\n")
        assemble_database(args, main_dir, RESULTS, formats)


def assemble_database(args, main_dir, RESULTS, formats):
    """Harmonise the cohorts, correct the site effect and write the pooled table.

    Args:
        args: Parsed command line. Read here: `cohorts`, `site_reference`, `idh`,
            `grade`, `stream_th`, `pairwise`, `adjust_covariates`, `n_perms`,
            `output_name` and `show`; the flags shaping the diagnostics are read
            by `report_site_diagnostics`, which now runs on every call.
        main_dir: Root directory holding the per-cohort folders.
        RESULTS: Output directory. It and its OS-stats/ already exist by now.
        formats: Figure formats to write.

    The order is deliberate: the diagnostics are reported before the correction
    is applied, so the evidence for the coefficient comes before the coefficient.
    """
    # --- Per-cohort harmonisation ------------------------------------------
    names = sorted(set(args.cohorts), key=lambda n: COHORTS[n]["id"])
    site_of = resolve_site_codes(names, args.site_reference)
    tables = []
    for name in names:
        paths = cohort_paths(main_dir, name, args.idh, args.grade, args.stream_th)
        data = LOADERS[name](paths)
        data["cohort"] = COHORTS[name]["id"]
        data["site"] = site_of[name]  # The 0/1 group the correction acts on
        report_censoring(name, data)
        tables.append(data)

    database = pd.concat(tables, ignore_index=True)
    cohort_ids = [COHORTS[n]["id"] for n in names]
    name_cohort = {COHORTS[n]["id"]: n for n in names}
    colors = [COHORTS[n]["color"] for n in names]
    n_cohort = {COHORTS[n]["id"]: len(t) for n, t in zip(names, tables)}
    site_labels = build_site_labels(names, site_of)
    grouping = "default" if is_default_partition(site_of) else "custom (--site-reference)"
    print("\nSite partition (" + grouping + "): "
          + "; ".join(f"{code} = {label}" for code, label in sorted(site_labels.items())))
    if len(set(site_of.values())) < 2:
        print("Every selected cohort is in the reference group, so no correction "
              "will be applied.")

    REPORT.heading("Cohorts")
    REPORT.paragraph(
        f"Partition of the cohorts into the two site groups ({grouping}). Group 0 "
        "is the reference, whose survival times are left untouched; group 1's are "
        "rescaled by the fitted coefficient.")
    REPORT.table(pd.DataFrame([
        dict(cohort=n, id=COHORTS[n]["id"], n=n_cohort[COHORTS[n]["id"]],
             events=int((tables[k]["status"] == 1).sum()),
             pct_censored=100.0 * (tables[k]["status"] == 0).mean(),
             site=site_of[n], site_group=site_labels[site_of[n]])
        for k, n in enumerate(names)]), float_format=lambda v: f"{v:.1f}")

    # --- Survival before correction ----------------------------------------
    REPORT.heading("Survival before correction")
    print("\n" + "=" * 40 + "\nSurvival before correction\n" + "=" * 40)
    plot_cohort_survival(
        full_data=database,
        cohort_ids=cohort_ids,
        name_cohort=name_cohort,
        colors=colors,
        RESULTS=RESULTS,
        stem="OS-cohorts_before-correction",
        title="Survival before correction",
        duration_col="OS (days)",
        formats=formats,
        show_plot=args.show,
    )

    # --- Pairwise comparisons ----------------------------------------------
    if args.pairwise:
        REPORT.heading("Pairwise cohort comparisons")
        REPORT.paragraph(
            "One figure per pair of cohorts. The left panels diagnose the model "
            "whose coefficient the right panel applies, so an adjusted pair is "
            "diagnosed adjusted; the cloglog curves and their log-rank stay "
            "marginal, since a Kaplan-Meier curve has no covariates to hold fixed.")
        for i, j in itertools.combinations(cohort_ids, 2):
            # A pairwise difference is worth no more than the site difference is:
            # two cohorts differ in case-mix as readily as two sites do. When
            # covariates are being adjusted for at all, each pair gets its own
            # adjusted coefficient, estimated on that pair alone -- a coefficient
            # borrowed from the site model would describe a different contrast --
            # and the diagnostic panel is fitted on that same model.
            pair_adjusted = None
            if args.adjust_covariates:
                pair = database[database["cohort"].isin([i, j])].copy()
                pair["pair"] = pair["cohort"].map({i: 0, j: 1})
                pair_adjusted = estimate_site_logHR(pair, args.adjust_covariates,
                                                    site_col="pair")
                if pair_adjusted["logHR"] is None:
                    print(f"WARNING: the adjusted model for {name_cohort[i]} vs "
                          f"{name_cohort[j]} could not be fit "
                          f"({pair_adjusted['reason']}); that figure stays crude.")
                    pair_adjusted = None
            inspect_survival_diffs_in_paired_cohorts(
                full_data=database,
                cohorts=[i, j],
                RESULTS=RESULTS,
                name_cohort=name_cohort,
                colors=[COHORTS[name_cohort[i]]["color"], COHORTS[name_cohort[j]]["color"]],
                N_cohorts=[n_cohort[i], n_cohort[j]],
                covariate_col="cohort",
                n_perms=args.n_perms,
                formats=formats,
                show_plot=args.show,
                logHR_override=(pair_adjusted or {}).get("logHR"),
                adjust_covariates=(list(args.adjust_covariates) if pair_adjusted else ()),
            )

    # --- Diagnostics --------------------------------------------------------
    # Always, and before the correction is applied, so the evidence for the number
    # comes before the number itself. They were behind a flag once; a correction
    # whose justification is optional is a correction nobody checks.
    diagnostics = report_site_diagnostics(database, args, RESULTS, site_labels,
                                          formats, show_plot=args.show)

    # --- Site correction ----------------------------------------------------
    # Survival in UCSF-PDGM is recorded differently from the remaining cohorts, so the
    # difference between the two 'site' groups is estimated and divided out.
    database, provenance = apply_site_correction(database, args, RESULTS, site_labels, formats)

    # --- Save ---------------------------------------------------------------
    stem = args.output_name or default_output_stem(names)
    database.to_csv(f"{RESULTS}/{stem}.csv", sep=",", index=False)
    database.to_csv(f"{RESULTS}/{stem}.tsv", sep="\t", index=False)
    with open(f"{RESULTS}/keys-maps.json", "w") as f:
        json.dump(KEYS_MAPS, f, indent=4)

    provenance.update(
        script="createDatabase.py",
        timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        command=" ".join(sys.argv),
        cohorts=names,
        cohort_ids={n: COHORTS[n]["id"] for n in names},
        site_codes=site_of,
        site_labels={str(k): v for k, v in site_labels.items()},
        site_reference=sorted([n for n in names if site_of[n] == 0],
                              key=lambda n: COHORTS[n]["id"]),
        site_partition="default" if is_default_partition(site_of) else "custom",
        output=f"{stem}.csv",
    )
    # Written once, at the end of the run: the recommendation below is part of it
    print(f"\nAssembled {len(database)} subjects from {len(cohort_ids)} cohorts --> {RESULTS}/{stem}.csv")
    print(f"Applied site log HR {provenance['logHR']} "
          f"({provenance['mode']}"
          + (f", adjusted for {', '.join(provenance['adjusted_for'])}"
             if provenance["adjusted_for"] else "") + ")")

    # --- Survival after correction ------------------------------------------
    REPORT.heading("Survival after correction")
    print("\n" + "=" * 40 + "\nSurvival after correction\n" + "=" * 40)
    plot_cohort_survival(
        full_data=database,
        cohort_ids=cohort_ids,
        name_cohort=name_cohort,
        colors=colors,
        RESULTS=RESULTS,
        stem="OS-cohorts_after-correction",
        title="Survival after correction",
        duration_col="OS (days) - corrected",
        formats=formats,
        show_plot=args.show,
    )

    # --- Which column to use ------------------------------------------------
    # Last, so it is read against the figures and tables that argue it, and after
    # the correction exists, so it can name what was actually written.
    provenance["recommended_outcome"] = report_recommendation(
        provenance, diagnostics, site_labels)
    write_provenance(provenance, RESULTS, stem)

    # --- One report holding all of it ---------------------------------------
    REPORT.heading("Provenance")
    REPORT.paragraph(
        "How the correction was obtained, as written to "
        f"{stem}_site-correction.json next to the table.")
    REPORT.code(json.dumps(provenance, indent=4, default=str))
    REPORT.heading("Run log")
    REPORT.paragraph(
        "Everything this run printed -- sample sizes, censoring, every fit and "
        "every warning, in the order they happened -- is in the text file below, "
        "not in this page. It is the same content the sections above are drawn "
        "from, at the level of detail a log has rather than a report.")
    REPORT.code(getattr(args, "log_path", "createDatabase_log.txt"),
                caption="Run log")
    report_path = REPORT.write(
        f"{RESULTS}/{stem}_report.html",
        title=f"{', '.join(names)} -- pooled database and site correction",
        subtitle=" ".join(sys.argv),
    )

    announce(f"\nAssembled {len(database)} subjects from {len(cohort_ids)} cohorts.")
    for label, path in (("table", f"{RESULTS}/{stem}.csv"),
                        ("report", report_path),
                        ("log", getattr(args, "log_path", "createDatabase_log.txt"))):
        announce(f"  {label:7s} {path}")
    verdict = provenance["recommended_outcome"]
    column = {"raw": "'OS (days)'", "corrected": "'OS (days) - corrected'"}.get(
        verdict, "neither column without further checks")
    announce(f"  {'verdict':7s} {verdict.upper()} -- analyse {column}")


if __name__ == "__main__":
    main()
