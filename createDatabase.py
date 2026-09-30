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
method -- it defaults to the `site` field of COHORTS (UCSF alone) and
`--site-reference` names any other group.
The resolved partition is recorded in the provenance JSON, so a table always says
which grouping produced it.

By default the effect is estimated from the `site` indicator alone and the survival
times of group 1 are multiplied by exp(log HR). A crude effect mixes three things:
a difference in when the survival clock starts (entry point), which should be
removed; a difference in which patients each cohort enrolled (case-mix), which
is real prognostic information and should be kept; and a difference in how
follow-up was lost (censoring), which a constant rescaling can neither remove
nor model. `--adjust-covariates` lets the site model condition on any of age,
sex, EOR, MGMT and KPS, which takes out case-mix; the censoring diagnostics say
how much of what remains censoring could explain, and only the rest is a
candidate for entry point.

The adjusted coefficient is estimated on the subjects reporting every chosen
covariate and then applied to every subject, so the assembled table never
shrinks -- missingness is severe and cohort-structured (EOR is unrecorded for all
of TCGA, MGMT for all of RHUH, KPS for all of UCSF and LUMIERE), and a
complete-case *table* would cost most of the sample. How the correction was
obtained is recorded in `<stem>_site-correction.json` next to the table.

The evidence behind that choice is reported on every run -- covariate balance and
missingness between the site groups, a same-sample adjustment ladder,
proportional-hazards tests for every term of the adjusted model (with the failing
terms described over follow-up time), a reverse-Kaplan-Meier follow-up
comparison, person-time completeness of follow-up, Cox models of what predicts
censoring, and a tipping-point analysis of how much informative censoring the
adjusted site effect can absorb. `--ladder-covariates` shapes it.

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
import itertools
import json
import os
import sys
from datetime import datetime

import matplotlib

# Figures are written to disk; only open a window when explicitly asked for. This
# has to run before anything imports pyplot -- the utils modules below do -- or the
# backend is already chosen by the time it is set.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import runlog  # noqa: E402
from utils.database.cohorts import (  # noqa: E402
    LOADERS, build_site_labels, cohort_paths, default_output_stem, is_default_partition,
    report_censoring, resolve_site_codes,
)
from utils.database.config import (  # noqa: E402
    ADJUSTMENT_COVARIATES, ADJUSTMENT_ORDER, COHORTS, DEFAULT_TIPPING_PLAUSIBLE, KEYS_MAPS,
)
from utils.database.correction import apply_site_correction, write_provenance  # noqa: E402
from utils.database.diagnostics import (  # noqa: E402
    assess_proportional_hazards, report_site_diagnostics,
)
from utils.database.method_text import report_method_and_references  # noqa: E402
from utils.database.plots import (  # noqa: E402
    inspect_survival_diffs_in_paired_cohorts, plot_cohort_survival,
)
from utils.database.recommendation import report_recommendation  # noqa: E402
from utils.database.site_model import build_site_design, estimate_site_logHR  # noqa: E402
from utils.formatting import subsection  # noqa: E402
from utils.report import REPORT  # noqa: E402
from utils.runlog import announce, resolve_log_path, tee_stdout  # noqa: E402


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def plausible_band(text):
    """argparse type of --tipping-plausible: a float above 1.

    Args:
        text: The value as typed.
    """
    value = float(text)
    if not value > 1.0:
        raise argparse.ArgumentTypeError(f"must be above 1 (got {text}); the band is "
                                         f"[1/B, B], so B = 1 would leave no band at all")
    return value


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
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "svg", "both"],
                        help="Figure format (default: pdf).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures as well as writing them to disk.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the permutation and bootstrap draws.")
    parser.add_argument("--tipping-plausible", type=plausible_band,
                        default=DEFAULT_TIPPING_PLAUSIBLE, metavar="B",
                        help="Band of delta, [1/B, B], the censoring tipping point "
                             "treats as a plausible departure from independent "
                             "censoring: lost patients dying up to B times as fast "
                             "(or 1/B as fast) as comparable patients who stayed. A "
                             "site effect that reaches HR = 1 inside it cannot be told "
                             f"apart from informative censoring (default: "
                             f"{DEFAULT_TIPPING_PLAUSIBLE:g}; must be above 1).")
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

    # Read by `announce`, which must know whether stdout already reaches the terminal
    runlog.ECHO_TO_TERMINAL = args.verbose
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
            # The same assessment the pooled comparison gets: the coefficient
            # drawn on that figure is only as good as the model it came from,
            # and a pair can fail proportional hazards where the pool does not
            names_pair = f"{name_cohort[i]} vs {name_cohort[j]}"
            pair = database[database["cohort"].isin([i, j])].copy()
            pair["pair"] = pair["cohort"].map({i: 0, j: 1})
            design, _, _ = build_site_design(pair, args.adjust_covariates)
            ph_frame = pd.concat(
                [design, pair[["pair"]],
                 pd.to_numeric(pair["OS (days)"], errors="coerce").rename("OS (days)"),
                 pd.to_numeric(pair["status"], errors="coerce").rename("status")],
                axis=1).dropna()
            ph_frame = ph_frame[ph_frame["OS (days)"] > 0]
            subsection(f"Proportional hazards: {names_pair}")
            assess_proportional_hazards(
                ph_frame, RESULTS,
                f"Site-effects_non-proportional-terms_{name_cohort[i]}-{name_cohort[j]}",
                formats, label=names_pair, show_plot=args.show)

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

    # --- Method and references ----------------------------------------------
    # The README says how to run the script; the reasoning behind what it does
    # belongs with the numbers it produced, which is here.
    report_method_and_references()

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
