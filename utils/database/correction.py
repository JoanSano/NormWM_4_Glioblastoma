"""The site correction written to the table, and the record of how it was obtained."""

import json

import numpy as np

from utils.database.plots import inspect_survival_diffs_in_paired_cohorts
from utils.database.site_model import estimate_site_logHR
from utils.formatting import fmt_p_phrase, section
from utils.report import REPORT


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
                  f"{np.exp(hi):.4f}, {fmt_p_phrase(adjusted['p'])})")
            print(f"  crude log HR = {logHR_crude:.6f}  -->  "
                  f"{removed:.1f}% of it is explained by the covariates")
            print(f"  estimated on {adjusted['n']} of {len(database)} subjects "
                  f"({adjusted['events']} events); applied to all {len(database)}")
            print(f"  proportional-hazards test on the adjusted site term: "
                  f"{fmt_p_phrase(adjusted['ph_p'])}")
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
