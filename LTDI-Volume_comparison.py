#!/usr/bin/env python3
"""Is lesion tract density (L-TDI) or tumour volume the better prognostic marker?

Reads the pooled table written by createDatabase.py and fits Cox proportional-hazards
models of overall survival in three blocks:

  Head to head   pairs of markers in one model, no clinical covariates:
                 E.size + W.L-TDI, E.size + E.L-TDI, E.L-TDI + W.L-TDI.
  Pre-surgical   age + sex, alone (base) and with one marker added:
                 W.L-TDI, E.L-TDI or E.size.
  Post-surgical  the same, with MGMT and extent of resection added to the base.

Within a block, every marker model extends the base by one covariate, so each is
compared with the base by AIC and a likelihood-ratio test, and with the other markers
by AIC. The pre-surgical marker sets are then validated leaving one cohort out, with a
Cox, a Weibull AFT and a log-logistic AFT fitter.

W.L-TDI is the whole-lesion L-TDI, E.L-TDI the contrast-enhancing L-TDI and E.size the
contrast-enhancing volume in cm3.

Settings that change the results
--------------------------------
--duration-col   raw or site-corrected survival (see createDatabase.py's recommendation).
--stratify-for   stratify every in-sample Cox model by a column (default: cohort), or
                 `none`. The cross-validation never stratifies; when this is cohort or
                 site it analyses the site-corrected survival instead, and warns.
--standardize    z-score continuous covariates (HR per SD) and map categorical codes
                 onto [-1, 1]. Estimated on each model's own sample, and in the
                 cross-validation on the training cohorts only.

Every output file carries a tag of these settings,
`strata-<column|none>_standardized-<true|false>`, so runs with different settings sit
side by side in the output folder:

    <RESULTS>/<output-dir>/Presurgical_WLTDI_<tag>.svg, ...   one forest plot per model
    <RESULTS>/<output-dir>/LTDI-Volume_comparison_<tag>_report.html
    <RESULTS>/<output-dir>/LTDI-Volume_comparison_<tag>_{coefficients,model-fit,cross-validation}.csv
    <RESULTS>/<output-dir>/LTDI-Volume_comparison_<tag>_log.txt

Examples
--------
    # Defaults: stratified by cohort, native units
    python LTDI-Volume_comparison.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                                     RESULTS-GBM_4-cohorts_Tissues

    # Unstratified, standardized covariates, raw survival
    python LTDI-Volume_comparison.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                                     RESULTS-GBM_4-cohorts_Tissues \
                                     --stratify-for none --standardize \
                                     --duration-col "OS (days)"
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import runlog  # noqa: E402
from utils.cox_models import (  # noqa: E402
    CORRECTED_DURATION, coefficient_table, cox_frame, cv_duration_column, draw_forest,
    fit_cox, leave_one_cohort_out,
)
from utils.database.config import COHORT_NAME_BY_ID, KEYS_MAPS  # noqa: E402
from utils.formatting import fmt_p_inline, section  # noqa: E402
from utils.report import Report, save_figure  # noqa: E402
from utils.runlog import announce, resolve_log_path, tee_stdout  # noqa: E402
from utils.statistics import llr_pvalue  # noqa: E402

VOXEL_SIZE = (0.5 ** 3) / 1000  # 0.5^3 mm3 per voxel x 0.001 cm3 per mm3
MARKERS = ["Whole lesion TDMap", "Enhancing TDMap", "Enhancing size (cm3)"]
CONTINUOUS = ["age", *MARKERS]
CATEGORICAL = ["sex", "mgmt", "eor"]

# Short names used in tables, file names and plot titles
SHORT = {"Whole lesion TDMap": "W.L-TDI", "Enhancing TDMap": "E.L-TDI",
         "Enhancing size (cm3)": "E.size"}
LONG = {"Whole lesion TDMap": "Whole tumor L-TDI", "Enhancing TDMap": "Contrast-enhancing L-TDI",
        "Enhancing size (cm3)": "Contrast-enhancing size"}
FILE = {"Whole lesion TDMap": "WLTDI", "Enhancing TDMap": "ELTDI", "Enhancing size (cm3)": "Esize"}

HEAD_TO_HEAD = [
    ["Enhancing size (cm3)", "Whole lesion TDMap"],
    ["Enhancing size (cm3)", "Enhancing TDMap"],
    ["Enhancing TDMap", "Whole lesion TDMap"],
]
BLOCKS = {
    # name: (file prefix, base covariates, title of the base covariates)
    "Pre-surgical": ("Presurgical", ["age", "sex"], "Age + Sex"),
    "Post-surgical": ("Postsurgical", ["age", "sex", "mgmt", "eor"], "Age + Sex + MGMT + EOR"),
}


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv: Argument list to parse, or None to read sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("main_dir", type=str,
                        help="Root directory of the project.")
    parser.add_argument("results_dir", type=str,
                        help="Directory holding the pooled table (createDatabase.py's output). "
                             "Relative paths are resolved under MAIN_DIR.")
    parser.add_argument("--data", type=str, default="data-clinical_TD-tissues_4-cohorts.csv",
                        help="Pooled table inside RESULTS_DIR.")
    parser.add_argument("--output-dir", type=str, default="Forest-plots_Cox-models",
                        help="Folder under RESULTS_DIR the figures, tables and report go to.")
    parser.add_argument("--duration-col", type=str, default=CORRECTED_DURATION,
                        help="Survival column. Use the one createDatabase.py recommended "
                             f"(default: {CORRECTED_DURATION!r}).")
    parser.add_argument("--event-col", type=str, default="status",
                        help="Event indicator column (default: status).")
    parser.add_argument("--stratify-for", type=str, default="cohort",
                        help="Column every in-sample Cox model is stratified by, or `none` "
                             "(default: cohort).")
    parser.add_argument("--standardize", action="store_true",
                        help="Continuous covariates as z-scores (HR per SD), categorical codes "
                             "mapped onto [-1, 1]. Default: native units.")
    parser.add_argument("--n-bootstrap", type=int, default=5000,
                        help="Bootstrap resamples for the C-index intervals (default: 5000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed of the bootstrap (default: 42).")
    parser.add_argument("--format", type=str, default="svg", choices=["pdf", "svg", "both"],
                        help="Format of the per-model forest plots (default: svg).")
    parser.add_argument("--log", nargs="?", const="", default=None, metavar="FILE",
                        help="Name of the text file the run is written to (default: "
                             "LTDI-Volume_comparison_<tag>_log.txt in the output folder).")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print the run to the terminal.")
    args = parser.parse_args(argv)
    if args.stratify_for.lower() == "none":
        args.stratify_for = None
    return args


def settings_tag(args):
    """`strata-<column|none>_standardized-<true|false>`, carried by every output file.

    Args:
        args: Parsed command line.
    """
    strata = (args.stratify_for or "none").replace(" ", "-")
    return f"strata-{strata}_standardized-{str(args.standardize).lower()}"


def covariate_labels(standardize):
    """{parameter name: forest-plot label}, with the unit an HR refers to.

    Args:
        standardize: Whether the covariates were rescaled.

    Directions are read off the encodings createDatabase.py wrote, so a label cannot
    claim the opposite contrast from the one the codes define.
    """
    sex = {code: name for name, code in KEYS_MAPS["Sex"].items()}
    if standardize:
        return {"age": "Age (per SD)", "sex": f"Sex ({sex[1]} vs {sex[0]}, ±1)",
                "mgmt": "MGMT (±1)", "eor": "EOR (±1)",
                **{m: f"{SHORT[m]} (per SD)" for m in MARKERS}}
    return {"age": "Age (per year)", "sex": f"Sex ({sex[1]} vs {sex[0]})",
            "mgmt": "MGMT (per level)", "eor": "EOR (per level)",
            "Whole lesion TDMap": "W.L-TDI", "Enhancing TDMap": "E.L-TDI",
            "Enhancing size (cm3)": "E.size (per cm³)"}


def fmt_hr(row):
    """"HR (lower–upper)" at four decimals, enough for a per-cm3 HR close to 1.

    Args:
        row: Row of `coefficient_table`.
    """
    return f"{row['HR']:.4f} ({row['HR 95% CI lower']:.4f}–{row['HR 95% CI upper']:.4f})"


class Analysis:
    """One run: the data, the settings, the report and the tables it accumulates.

    Args:
        args: Parsed command line.
        data: Pooled table, markers complete.
        out_dir: Output folder.
        tag: Settings tag of the file names.
    """

    def __init__(self, args, data, out_dir, tag):
        """Store the arguments the class docstring describes."""
        self.args, self.data, self.out_dir, self.tag = args, data, out_dir, tag
        self.strata = [args.stratify_for] if args.stratify_for else None
        self.labels = covariate_labels(args.standardize)
        self.report = Report()
        self.formats = ["pdf", "svg"] if args.format == "both" else [args.format]
        self.coefficients, self.fits = [], []

    def fit(self, block, name, covariates):
        """Fit one model, print it and record its coefficients and fit statistics.

        Args:
            block: Block the model belongs to.
            name: Model name within the block.
            covariates: Model covariates.
        """
        a = self.args
        frame = cox_frame(self.data, covariates, a.duration_col, a.event_col, self.strata,
                          a.standardize, CONTINUOUS, CATEGORICAL)
        fit = fit_cox(frame, covariates, a.duration_col, a.event_col, self.strata,
                      a.n_bootstrap, a.seed)
        coefs = coefficient_table(fit["model"])
        coefs.insert(0, "model", name)
        coefs.insert(0, "block", block)
        self.coefficients.append(coefs)
        print(f"\n{block} -- {name} (N={fit['n']}, events={fit['events']})")
        print(coefs.drop(columns=["block", "model"]).to_string(index=False))
        print(f"  C-index {fit['cindex']:.4f}, bootstrap 95% CI "
              f"{fit['cindex_boot'][1]:.4f}-{fit['cindex_boot'][2]:.4f}; "
              f"log-likelihood {fit['loglik']:.4f}; AIC {fit['aic']:.4f}")
        return fit

    def fit_row(self, block, name, fit, base=None):
        """One row of the model-fit table; the comparison with `base` if given.

        Args:
            block: Block the model belongs to.
            name: Model name.
            fit: Record of `fit_cox`.
            base: Record of the base model this one extends by one covariate, or None.
        """
        row = dict(block=block, model=name, n=fit["n"], events=fit["events"],
                   cindex=fit["cindex"], cindex_lower=fit["cindex_boot"][1],
                   cindex_upper=fit["cindex_boot"][2], loglik=fit["loglik"], aic=fit["aic"],
                   delta_aic_vs_base=np.nan, llr_p_vs_base=np.nan)
        # A likelihood-ratio test compares models fitted to the same patients
        if base is not None and base["n"] == fit["n"]:
            row["delta_aic_vs_base"] = fit["aic"] - base["aic"]
            row["llr_p_vs_base"] = llr_pvalue(fit["loglik"], base["loglik"], df_diff=1)
        self.fits.append(row)
        return row

    # --- Blocks ----------------------------------------------------------
    def head_to_head(self):
        """Pairs of markers in one model, no clinical covariates."""
        section("Head to head")
        r = self.report
        r.heading("L-TDI or volume, head to head")
        r.paragraph(
            "Two markers enter one Cox model together, without clinical covariates, so each "
            "hazard ratio is that marker's association with survival holding the other fixed. "
            "A marker whose interval still excludes 1 carries information the other lacks.")
        rows = []
        for covariates in HEAD_TO_HEAD:
            name = " + ".join(SHORT[c] for c in covariates)
            fit = self.fit("Head to head", name, covariates)
            self.fit_row("Head to head", name, fit)
            for _, c in coefficient_table(fit["model"]).iterrows():
                rows.append({"Model": name, "Covariate": self.labels.get(c["covariate"], c["covariate"]),
                             "HR (95% CI)": fmt_hr(c), "p": fmt_p_inline(c["p"], 4),
                             "N": fit["n"], "C-index": f"{fit['cindex']:.3f}"})
        r.table(pd.DataFrame(rows), caption="Hazard ratios of the head-to-head models.")

    def nested_block(self, block):
        """Base model and the base plus one marker at a time.

        Args:
            block: Key of BLOCKS.
        """
        section(block)
        prefix, base_covs, base_title = BLOCKS[block]
        models = [(SHORT[m], FILE[m], [*base_covs, m], f"OS ~ {base_title} + {LONG[m]}")
                  for m in MARKERS]
        models.append(("Base", "base", base_covs, f"OS ~ {base_title}"))

        fits = {name: self.fit(block, name, covs) for name, _, covs, _ in models}
        rows = [self.fit_row(block, name, fits[name], None if name == "Base" else fits["Base"])
                for name, _, _, _ in models]

        # One file per model for the manuscript, one grid for the report
        grid, axes = plt.subplots(2, 2, figsize=(12, 6.5))
        for ax, (name, stem, covs, title) in zip(axes.flat, models):
            draw_forest(ax, fits[name]["model"], self.labels, title)
            fig, single = plt.subplots(figsize=(6, 3))
            draw_forest(single, fits[name]["model"], self.labels, title)
            fig.tight_layout()
            save_figure(fig, os.path.dirname(self.out_dir), f"{prefix}_{stem}_{self.tag}",
                        self.formats, subdir=os.path.basename(self.out_dir), report=None)
            plt.close(fig)
        grid.tight_layout()

        r = self.report
        r.heading(f"{block} models")
        r.paragraph(
            f"The base model is {base_title}; each marker model adds one marker to it. "
            "ΔAIC and the likelihood-ratio p-value compare each marker model with the base "
            "(same patients), and the AIC column ranks the three markers against one another.")
        r.figure(grid, caption=(
            "Hazard ratios with 95% CI. A salmon square marks a covariate whose interval "
            "crosses HR = 1; a white one, an interval that excludes it."))
        plt.close(grid)
        table = pd.DataFrame([{
            "Model": row["model"], "N": row["n"], "Events": row["events"],
            "C-index (95% CI)": f"{row['cindex']:.3f} ({row['cindex_lower']:.3f}–{row['cindex_upper']:.3f})",
            "Log-likelihood": f"{row['loglik']:.4f}", "AIC": f"{row['aic']:.4f}",
            "ΔAIC vs base": "" if np.isnan(row["delta_aic_vs_base"]) else f"{row['delta_aic_vs_base']:+.4f}",
            "LLR p vs base": "" if np.isnan(row["llr_p_vs_base"]) else fmt_p_inline(row["llr_p_vs_base"], 4),
        } for row in rows])
        r.table(table, caption=f"Fit of the {block.lower()} models.")

    def cross_validation(self, cv_duration):
        """Leave-one-cohort-out validation of the pre-surgical marker sets.

        Args:
            cv_duration: Survival column the validation analyses.
        """
        section("Leave-one-cohort-out validation")
        a = self.args
        sets = [[*BLOCKS["Pre-surgical"][1], m] for m in MARKERS] + [BLOCKS["Pre-surgical"][1]]
        frames = []
        for covs in sets:
            name = "Age + Sex" + "".join(f" + {SHORT[c]}" for c in covs if c in MARKERS)
            cv = leave_one_cohort_out(self.data, covs, cv_duration, a.event_col, a.standardize,
                                      CONTINUOUS, CATEGORICAL)
            cv.insert(0, "covariates", name)
            frames.append(cv)
        cv = pd.concat(frames, ignore_index=True)
        cv["held_out"] = cv["held_out"].map(lambda c: COHORT_NAME_BY_ID.get(int(c), c))
        print(cv.to_string(index=False))

        wide = cv.pivot_table(index=["covariates", "fitter"], columns="held_out",
                              values="test_cindex", sort=False)
        per_cohort = wide.copy()
        wide["Mean"] = per_cohort.mean(axis=1)
        # Variability across held-out cohorts: sample SD (n - 1) and the worst cohort
        wide["SD"] = per_cohort.std(axis=1, ddof=1)
        wide["Min"] = per_cohort.min(axis=1)
        wide["RMSE (days)"] = cv.groupby(["covariates", "fitter"], sort=False)["rmse_days"].mean()
        wide = wide.reset_index().rename(columns={"covariates": "Covariates", "fitter": "Fitter"})
        wide.columns.name = None  # else to_html prints "held_out" as an extra header row

        r = self.report
        r.heading("Leave-one-cohort-out validation")
        r.paragraph(
            "Each model is fitted on all cohorts but one and scored on the one left out, "
            f"analysing {cv_duration!r} without strata. "
            + ("Standardization is estimated on the training cohorts and applied unchanged "
               "to the held-out cohort. " if a.standardize else "")
            + "Cells are held-out C-indices. Mean, SD and Min summarise them across the "
            "held-out cohorts: SD is the sample standard deviation (n − 1), how much "
            "discrimination depends on which cohort is left out, and Min is the worst "
            "held-out cohort. RMSE is the root mean squared error of the "
            "predicted expected survival on held-out patients whose death was observed, "
            "averaged over the held-out cohorts.")
        r.table(wide, caption="Held-out C-index by cohort left out.",
                formatters={c: "{:.3f}".format for c in wide.columns if c not in
                            ("Covariates", "Fitter", "RMSE (days)")}
                | {"RMSE (days)": "{:.1f}".format})
        return cv


def run(args, data, out_dir, tag, log_path):
    """Fit every block, then write the figures, tables and the report.

    Args:
        args: Parsed command line.
        data: Pooled table, markers complete.
        out_dir: Output folder.
        tag: Settings tag of the file names.
        log_path: Log file, named in the report.
    """
    analysis = Analysis(args, data, out_dir, tag)
    r = analysis.report

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        cv_duration = cv_duration_column(data, args.duration_col, args.stratify_for)
    cv_warnings = [str(w.message) for w in caught]
    for message in cv_warnings:
        print(f"WARNING: {message}")
        print(f"WARNING: {message}", file=sys.stderr)

    # --- Settings ------------------------------------------------------------
    r.heading("Run settings")
    hr_units = ("continuous covariates per SD; categorical codes on [-1, 1] (sex ±1, MGMT and "
                "EOR -1/0/1), so a binary covariate's full contrast is HR²"
                if args.standardize else "native units: per year, per cm³, per L-TDI unit, per "
                "category level")
    r.table(pd.DataFrame([
        ("Data", f"{args.data} ({len(data)} patients with every marker)"),
        ("Survival column", args.duration_col),
        ("Event column", args.event_col),
        ("Stratification (in-sample Cox)", args.stratify_for or "none"),
        ("Standardization", f"{'on' if args.standardize else 'off'} — HRs in {hr_units}"),
        ("Survival column (cross-validation)", f"{cv_duration} (never stratified)"),
        ("Bootstrap resamples / seed", f"{args.n_bootstrap} / {args.seed}"),
        ("File tag", tag),
    ], columns=["Setting", "Value"]))
    if args.stratify_for in ("cohort", "site") and args.duration_col == CORRECTED_DURATION:
        r.callout(
            f"Survival is site-corrected and the models are also stratified by "
            f"{args.stratify_for}. The correction rescales every time in a stratum by the same "
            "factor, which leaves the within-stratum ordering, and hence the Cox coefficients, "
            "unchanged; it only moves the C-index, which is pooled across strata.")
    for message in cv_warnings:
        r.callout(message)
    r.paragraph(
        "Terms. HR: hazard ratio, exp(β), the multiplicative change in the hazard per unit of "
        "the covariate (above 1 means shorter survival). 95% CI: Wald confidence interval. "
        "C-index: Harrell's concordance, the fraction of comparable patient pairs whose predicted "
        "risks order their survival correctly (0.5 is chance), pooled across strata; its interval "
        "is a percentile bootstrap over patients with the fitted model held fixed. AIC: Akaike "
        "information criterion of the partial likelihood, 2k − 2 log L, lower is better. "
        "LLR: likelihood-ratio test of a model against the base it extends by one covariate "
        "(χ², 1 degree of freedom).")

    # --- Models ----------------------------------------------------------------
    analysis.head_to_head()
    for block in BLOCKS:
        analysis.nested_block(block)
    cv = analysis.cross_validation(cv_duration)

    # --- Tables and report ---------------------------------------------------
    stem = f"{out_dir}/LTDI-Volume_comparison_{tag}"
    paths = {
        "coefficients": f"{stem}_coefficients.csv",
        "model fit": f"{stem}_model-fit.csv",
        "cross-validation": f"{stem}_cross-validation.csv",
    }
    pd.concat(analysis.coefficients, ignore_index=True).to_csv(paths["coefficients"], index=False)
    pd.DataFrame(analysis.fits).to_csv(paths["model fit"], index=False)
    cv.to_csv(paths["cross-validation"], index=False)

    r.heading("Files")
    r.paragraph(
        "Full-precision tables: " + ", ".join(os.path.basename(p) for p in paths.values())
        + f". Run log: {os.path.basename(log_path)}. Per-model forest plots: "
        f"<Pre|Post>surgical_<marker>_{tag}.{'/'.join(analysis.formats)}. All in {out_dir}.")
    paths["report"] = r.write(f"{stem}_report.html",
                              title="L-TDI or tumour volume: Cox models of survival",
                              subtitle=" ".join(sys.argv))
    return paths


def main(argv=None):
    """Parse the command line, load the table and run the comparison quietly.

    Args:
        argv: Argument list to parse, or None to read sys.argv.
    """
    args = parse_args(argv)
    main_dir = os.path.abspath(args.main_dir)
    RESULTS = args.results_dir if os.path.isabs(args.results_dir) else f"{main_dir}/{args.results_dir}"
    out_dir = f"{RESULTS}/{args.output_dir}"
    os.makedirs(out_dir, exist_ok=True)
    tag = settings_tag(args)

    data = pd.read_csv(f"{RESULTS}/{args.data}")
    data["Enhancing size (cm3)"] = VOXEL_SIZE * data["Enhancing size (voxels)"]
    # Every model sees the same patients' markers, so AIC and log-likelihood compare
    data = data.dropna(subset=MARKERS)
    for col in (args.duration_col, args.event_col, args.stratify_for):
        if col is not None and col not in data.columns:
            sys.exit(f"Column {col!r} is not in {args.data}.")

    runlog.ECHO_TO_TERMINAL = args.verbose
    log_path = resolve_log_path(args.log, out_dir, default=f"LTDI-Volume_comparison_{tag}_log.txt")
    if not args.verbose:
        print(f"LTDI-Volume_comparison.py: {tag} -> {out_dir}", file=sys.__stdout__, flush=True)
        print(f"  running quietly; the run is written to {log_path} (--verbose to watch it here)",
              file=sys.__stdout__, flush=True)
    with tee_stdout(log_path, echo=args.verbose):
        print(f"# LTDI-Volume_comparison.py -- {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# {' '.join(sys.argv)}\n")
        paths = run(args, data, out_dir, tag, log_path)
        announce("")
        for label, path in (*paths.items(), ("log", log_path)):
            announce(f"  {label:16s} {path}")


if __name__ == "__main__":
    main()
