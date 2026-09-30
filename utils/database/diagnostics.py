"""The site diagnostics section of the report: case-mix, entry point or censoring?"""

import pandas as pd

from utils.database.censoring import (censoring_hazard_table, censoring_tipping_point,
                                      followup_completeness, reverse_km_followup)
from utils.database.config import (DEFAULT_LADDER, DEFAULT_TIPPING_PLAUSIBLE, TIPPING_DELTAS,
                                   TIPPING_IMPUTATIONS)
from utils.database.plots import (plot_adjustment_ladder, plot_reverse_km, plot_time_varying_terms,
                                  plot_tipping_point)
from utils.database.site_model import (adjustment_ladder, build_site_design, fit_cox,
                                       proportional_hazards_table, site_balance_table,
                                       time_varying_terms)
from utils.formatting import fmt_p, fmt_p_phrase, section, subsection
from utils.report import REPORT


def assess_proportional_hazards(frame, RESULTS, stem, formats, label,
                                duration_col="OS (days)", status_col="status",
                                show_plot=True, heading_level=3):
    """Test PH for every term, and describe beta(t) for the terms that fail.

    Args:
        frame: Design matrix plus duration and status, as the model is fitted.
        RESULTS: Results directory; any figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        formats: Figure formats to write.
        label: What is being assessed, for the headings and the figure title.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        show_plot: Display the figure as well as writing it.
        heading_level: HTML heading level for the report sections.

    Returns (gt, varying): the per-term Grambsch-Therneau table, and the
    time-varying fits for the terms that failed it -- an empty DataFrame when
    every term is proportional, which is the outcome worth hoping for.

    Run for the pooled site comparison and for each pair of cohorts, because the
    coefficient applied in either case is only as good as the model it came from.
    """
    empty = pd.DataFrame()
    model, reason = fit_cox(frame, duration_col, status_col)
    if model is None:
        print(f"Proportional-hazards assessment could not be fitted ({reason}).")
        REPORT.heading(f"Proportional hazards: {label}", level=heading_level)
        REPORT.paragraph(f"Not assessed: {reason}")
        return empty, empty

    gt = proportional_hazards_table(model, frame)
    violators = gt.loc[gt["violates"], "term"].tolist()

    REPORT.heading(f"Proportional hazards: {label}", level=heading_level)
    REPORT.paragraph(
        "Grambsch-Therneau [4] against "
        f"{gt.attrs['transform']}-transformed time, for every term of the model "
        "the coefficient comes from -- not the site term alone. A term with a "
        "small p has an effect that drifts over follow-up, so the single "
        "coefficient reported for it is an average over the whole curve. Both the "
        "raw p and the Bonferroni-adjusted one are given: the raw p is one term's "
        f"evidence, and p_bonferroni reads it against the {gt.attrs['terms']} "
        "terms tested at once. A term is flagged, and described over time below, "
        "on the adjusted p -- the whole model is screened in one pass, so the raw "
        "p of the worst term is not the evidence it appears to be. lifelines "
        "offers no equivalent of R's cox.zph global row, so the smallest adjusted "
        "p stands in for it.")
    gt_formats = dict(test_statistic=lambda v: f"{v:.4f}",
                      p=lambda v: fmt_p(v, 4),
                      p_bonferroni=lambda v: fmt_p(v, 4))
    REPORT.table(gt, formatters=gt_formats)
    print(gt.to_string(index=False, formatters=gt_formats))
    print(f"  p_bonferroni is the raw p multiplied by the {gt.attrs['terms']} terms "
          f"tested, capped at 1; 'violates' is set on it, not on the raw p.")

    if not violators:
        REPORT.paragraph("Every term is proportional; no term needs describing "
                         "over time.")
        print("\nEvery term satisfies proportional hazards.")
        return gt, empty

    print(f"\n{len(violators)} term(s) fail: {', '.join(violators)}. "
          f"Describing each over time.")
    varying, curves = time_varying_terms(frame, violators, duration_col, status_col)
    REPORT.paragraph(
        f"{len(violators)} term(s) fail the test. Each is refitted with a "
        "log-time interaction -- \\(\\beta(t) = \\beta + \\theta\\log(t/t_{\\mathrm{ref}})\\), "
        "the data split at the event times -- and compared against the "
        "constant-coefficient model by likelihood ratio. \\(\\theta\\) is the "
        "change in log hazard ratio per unit of log time, \\(\\beta\\) the log "
        "hazard ratio at the median event time \\(t_{\\mathrm{ref}}\\), and "
        "\\(\\theta = 0\\) is proportional hazards. The equations are in the "
        "Method section at the end.")
    varying_formats = dict(p=lambda v: fmt_p(v, 4), lr_chi2=lambda v: f"{v:.4f}",
                           rows=lambda v: f"{int(v):d}")
    REPORT.table(varying, float_format=lambda v: f"{v:.4f}",
                 formatters=varying_formats)
    print(varying.to_string(index=False, float_format=lambda v: f"{v:.4f}",
                            formatters=varying_formats))
    plot_time_varying_terms(curves, varying, RESULTS, stem, formats,
                            show_plot=show_plot,
                            title=f"Terms that are not proportional: {label}")
    return gt, varying


def report_site_diagnostics(database, args, RESULTS, site_labels, formats, show_plot=True):
    """Everything a reader needs to judge whether 'site' is case-mix, entry point or censoring.

    Args:
        database: Assembled table, before the correction is applied.
        args: Parsed command line. Read here: `ladder_covariates` (falling back to
            `adjust_covariates`, then to DEFAULT_LADDER).
        RESULTS: Results directory; tables and the figure land in its OS-stats/.
        site_labels: {code: name} used to name the site groups in every table.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.

    Returns the tables as a dict, which is also what gets written as CSVs.

    Writes as CSVs under OS-stats/, and into the report, the balance and
    missingness table between the two site groups, the same-sample adjustment
    ladder, the reverse-Kaplan-Meier follow-up comparison, the person-time
    completeness of follow-up, the censoring-hazard models and the
    informative-censoring tipping point, plus the figure of the site effect
    against follow-up time. These are supplement tables rather
    than lines in a log, so they are saved as well as reported.
    """
    ladder_covariates = args.ladder_covariates or list(args.adjust_covariates) or DEFAULT_LADDER
    out = {}

    section("SITE DIAGNOSTICS: is the survival difference case-mix, entry point or censoring?")
    REPORT.heading("Site diagnostics: case-mix, entry point or censoring?")
    REPORT.paragraph(
        "A survival difference between the site groups can come from three "
        "places. Case-mix: the groups enrolled different patients (older, fewer "
        "resections, less methylated MGMT). Entry point: the survival clock starts "
        "at a different event in one group, e.g. at the preoperative MRI in one and "
        "at diagnosis in another. Censoring: one group lost more of its patients "
        "to follow-up, and the patients it lost were not like those it kept. The "
        "three are not exclusive, and they call for different responses: "
        "case-mix is real and must be kept, entry point is an artefact the "
        "corrected column is meant to remove, and censoring is an artefact that "
        "no rescaling removes.")
    REPORT.paragraph(
        "The sections below take them in turn. Balance and the adjustment ladder "
        "(the first two) measure case-mix: what is left of the site effect once "
        "the covariates are held fixed. The four follow-up and censoring sections "
        "ask whether censoring could produce that remainder. Entry point has no "
        "direct measurement in these tables -- nothing records when each cohort "
        "started its clock -- so it can only be a candidate for what is left once "
        "the other two are accounted for, alongside case-mix nobody recorded, "
        "differences in treatment after the clock starts, and chance. Both entry point and censoring tend to concentrate the "
        "difference in the first months, which the proportional-hazards section "
        "would show as a site effect that drifts over follow-up; they cannot be "
        "separated from each other with these data.")
    REPORT.paragraph(
        f"Covariates walked: {', '.join(ladder_covariates)}. Site groups: "
        + "; ".join(f"{k} = {v}" for k, v in site_labels.items()) + ".")
    print(f"Covariates walked: {', '.join(ladder_covariates)}")
    print(f"Site groups: " + "; ".join(f"{k} = {v}" for k, v in site_labels.items()))

    subsection("Balance and missingness between site groups")
    balance = site_balance_table(database, covariates=ladder_covariates)
    out["balance"] = balance
    if not balance.empty:
        REPORT.heading("Balance and missingness between site groups", level=3)
        REPORT.paragraph(
            "SMD is the standardised mean difference: the difference between the "
            "two site groups' means divided by their pooled standard deviation "
            "(defined under Methods). For a categorical covariate it is computed "
            "per level, on the 0/1 indicator of that level, so the means are "
            "proportions. |SMD| > 0.10 marks an imbalance worth adjusting for. A "
            "covariate a site never records cannot be balanced by any adjustment, "
            "so the two pct_missing columns are read alongside the SMD.")
        REPORT.table(balance, float_format=lambda v: f"{v:.3f}")
        print(balance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nSMD = standardised mean difference: |mean1 - mean0| / pooled SD, per level")
        print("for a categorical covariate (so the means are proportions).")
        print("|SMD| > 0.10 marks an imbalance worth adjusting for. A covariate a site")
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
    ladder_formats = dict(p=lambda v: fmt_p(v, 4), ph_p=lambda v: fmt_p(v, 4),
                          n=lambda v: f"{int(v):d}", events=lambda v: f"{int(v):d}")
    REPORT.table(ladder, float_format=lambda v: f"{v:.4f}", formatters=ladder_formats)
    print(ladder.to_string(index=False, float_format=lambda v: f"{v:.4f}",
                           formatters=ladder_formats))
    failed = ladder[ladder["reason"].notna() & (ladder["reason"] != "None")]
    if len(failed):
        print(f"\nWARNING: {len(failed)} of {len(ladder)} rungs could not be fitted. The most")
        print(f"         common reason was: {failed['reason'].iloc[0]}")
        print("         A covariate a whole cohort never records shrinks the complete-case")
        print("         sample onto a single site, leaving nothing to compare. KPS does this")
        print("         (UCSF and LUMIERE record none), which is why it is not in the default")
        print("         ladder. Drop it, or read the balance table's missingness columns.")
    print("\nNote: the baseline hazard is Breslow's throughout, but lifelines breaks")
    print("ties in the partial likelihood by Efron and offers no alternative, while")
    print("the coefficient actually applied to the survival times comes from")
    print("scikit-survival, which uses Breslow for both. With many tied survival days")
    print("the two differ in the 2nd-3rd decimal. That is expected.")
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
        line = (f"Log-rank test on the censoring distributions: chi2 = {chi2:.4f}, "
                f"{fmt_p_phrase(p_val)}.")
        print("\n" + line)
        print("A difference here is a difference in how long the groups were watched,")
        print("not in how long they survived.")
        REPORT.paragraph(line)
    plot_reverse_km(database, site_labels, RESULTS, "Site-diagnostics_reverse-KM",
                    formats, show_plot=show_plot)

    subsection("Completeness of follow-up (person-time)")
    completeness = followup_completeness(database, site_labels)
    out["followup-completeness"] = completeness
    REPORT.heading("Completeness of follow-up", level=3)
    REPORT.paragraph(
        "The median follow-up above says how LONG each group was watched. This "
        "table says how COMPLETELY: of the follow-up each group owed up to a "
        "horizon (12 and 24 months), how much was actually observed. A patient "
        "who dies before the horizon has been followed completely -- their outcome "
        "is known -- so only patients censored before the horizon count as "
        "incomplete. Every column runs from 0 to 1, 1 meaning nobody was lost. "
        "Indented rows are the cohorts inside a site group that pools several.")
    for name, meaning in [
        ("lost_before_horizon",
         "patients censored before the horizon, i.e. lost to follow-up."),
        ("percentage",
         "the fraction of patients not lost. It treats a patient lost at month 11 "
         "as if lost at month 0, so it is the harshest reading."),
        ("reverse_km",
         "the reverse Kaplan-Meier curve of the figure above, read at the "
         "horizon. It counts every death as a loss of follow-up too, so a group "
         "with many early deaths looks poorly followed; shown for comparison only."),
        ("CCI",
         "Clark's completeness index [12]: observed follow-up time divided by the "
         "time that would have been observed had nobody been lost, assuming each "
         "lost patient would have lived to the horizon. That assumption owes them "
         "too much time, so CCI is a lower bound."),
        ("SPT",
         "Xue's simplified person-time rate [11]: lost patients credited with the "
         "time they were actually seen, every other patient with the whole "
         "horizon. It credits deaths with time they did not live, so it is a "
         "slight upper bound."),
        ("FPT",
         "Xue's formal person-time rate [11]: observed follow-up time divided by "
         "the time owed had nobody been lost, with the time owed estimated from "
         "the group's own survival curve. The best single estimate; it lies "
         "between CCI and SPT."),
    ]:
        REPORT.paragraph(f"{name}: {meaning}")
    completeness_formats = dict(n=lambda v: f"{int(v):d}",
                                horizon_months=lambda v: f"{int(v):d}",
                                lost_before_horizon=lambda v: f"{int(v):d}")
    REPORT.table(completeness, float_format=lambda v: f"{v:.3f}",
                 formatters=completeness_formats)
    print(completeness.to_string(index=False, float_format=lambda v: f"{v:.3f}",
                                 formatters=completeness_formats))
    REPORT.paragraph(
        "What to look at: FPT, and the gap in it between the site groups. Groups "
        "with similar FPT lost similar shares of their follow-up, and censoring "
        "is unlikely to separate them. A group whose FPT is well below the other's "
        "(say 0.80 against 0.97) lost follow-up that the other did not, and its "
        "survival curve rests on the assumption that the patients it lost were "
        "no sicker than those it kept -- which the next section tests. The "
        "percentage and reverse_km columns will look worse than FPT; that is how "
        "they are built, not a second problem.")

    subsection("What predicts censoring? (Cox on the censoring hazard)")
    censoring = censoring_hazard_table(database, ladder_covariates, site_labels)
    out["censoring-hazard"] = censoring
    REPORT.heading("What predicts censoring?", level=3)
    REPORT.paragraph(
        "Were the patients lost to follow-up different from those who stayed? "
        "Within each site group, two Cox models on the same patients and "
        "covariates: one where the 'event' is being censored (HR_censoring), and "
        "the ordinary one where the event is death (HR_death).")
    REPORT.paragraph(
        "Reading a row: a hazard ratio above 1 means the covariate makes that "
        "outcome come sooner, below 1 later. same_side = yes when both hazard "
        "ratios are above 1 or both below 1: the patients the covariate makes "
        "more likely to be lost are then also the ones it makes more likely to "
        "die. An HR_censoring of 1.03 per year of age next to an HR_death of 1.04 "
        "means older patients both drop out sooner and die sooner. When the "
        "covariates that predict censoring all do this, the patients lost to "
        "follow-up were, on average, the sicker ones.")
    censoring_formats = dict(p_censoring=lambda v: fmt_p(v, 4), p_death=lambda v: fmt_p(v, 4),
                             n=lambda v: f"{int(v):d}", censored=lambda v: f"{int(v):d}")
    REPORT.table(censoring, float_format=lambda v: f"{v:.4f}", formatters=censoring_formats)
    print(censoring.to_string(index=False, float_format=lambda v: f"{v:.4f}",
                              formatters=censoring_formats))
    for label, (chi2, df, p_val) in censoring.attrs["global"].items():
        line = (f"{label}: do the covariates predict censoring at all? Likelihood "
                f"ratio test against a model without them, chi2 = {chi2:.4f} on "
                f"{df} df, {fmt_p_phrase(p_val)}.")
        print(line)
        REPORT.paragraph(line)
    skipped = censoring[censoring["term"] == "(not modelled)"]
    for row in skipped.itertuples():
        line = (f"{row.group} is not modelled: {row.reason}. The model needs every "
                f"covariate, so only complete cases enter it, and in this group "
                f"they are {censoring.attrs['composition'].get(row.group, 'n/a')}.")
        print(line)
        REPORT.paragraph(line)
    REPORT.paragraph(
        "Why it matters. The adjusted site model conditions on these same "
        "covariates, so censoring that runs through them does not bias the "
        "adjusted site hazard ratio. It does bias every Kaplan-Meier curve, crude "
        "or corrected, since a curve conditions on nothing. And a group that loses "
        "its sicker patients on the recorded covariates may also be losing them on "
        "an unrecorded one, such as performance status, which no adjustment can "
        "reach; the next section asks how much that could matter.")
    # to_csv drops attrs, so the global tests travel as a table of their own
    out["censoring-hazard-global"] = pd.DataFrame(
        [dict(group=k, chi2=v[0], df=v[1], p=v[2])
         for k, v in censoring.attrs["global"].items()])

    subsection("Tipping point: informative censoring of one site group")
    plausible = getattr(args, "tipping_plausible", DEFAULT_TIPPING_PLAUSIBLE)
    # The band's own edges are always tested, so a band wider than the default
    # grid is never judged from a line that stops short of it
    deltas = sorted(set(TIPPING_DELTAS) | {plausible, 1.0 / plausible})
    tipping = censoring_tipping_point(database, ladder_covariates, site_labels,
                                      deltas=deltas)
    tipping.attrs["plausible"] = plausible
    out["censoring-tipping-point"] = tipping
    REPORT.heading("Tipping point for informative censoring", level=3)
    if tipping.empty:
        print(f"Not computed: {tipping.attrs['reason']}")
        REPORT.paragraph(f"Not computed: {tipping.attrs['reason']}")
    else:
        REPORT.paragraph(
            "The question. A censored patient's death was never observed. Every "
            "estimate above assumes that, after being censored, they went on to "
            "die at the same rate as comparable patients who stayed in follow-up "
            "-- same covariates, same site. That is 'independent censoring', and "
            "the data cannot confirm it, because what happened after censoring is "
            "exactly what is missing. What can be done is to ask: if the censored "
            "patients had in fact died faster (or slower) than that, how much "
            "would the adjusted site effect change?")
        REPORT.paragraph(
            "How it is done [13]. delta is how much faster than a comparable "
            "patient who stayed. delta = 1 is the independent-censoring "
            "assumption itself; delta = 2 says they died at twice the rate; "
            "delta = 0.5 at half. For one value of delta: (1) every censored "
            "patient of one site group is given a plausible death time, drawn "
            "from the adjusted Cox model with their own covariates and site, "
            "later than the day they were censored, and with the hazard after "
            "censoring multiplied by delta; (2) with those deaths now in the data, "
            "the adjusted site model is fitted again; (3) steps 1-2 are repeated "
            f"{TIPPING_IMPUTATIONS} times, since the death times are drawn at "
            "random, and the site hazard ratios are combined by Rubin's rules [14], "
            "whose interval includes the spread between the repetitions. This is "
            "done for each value of delta, and separately for the censored "
            "patients of each site group, since which group lost its patients "
            "informatively is not known.")
        tipping_formats = dict(p=lambda v: fmt_p(v, 4), imputed=lambda v: f"{int(v):d}",
                               imputations=lambda v: f"{int(v):d}",
                               delta=lambda v: f"{v:.2f}")
        REPORT.table(tipping, float_format=lambda v: f"{v:.4f}", formatters=tipping_formats)
        print(tipping.to_string(index=False, float_format=lambda v: f"{v:.4f}",
                                formatters=tipping_formats))
        plot_tipping_point(tipping, RESULTS, "Site-diagnostics_censoring-tipping-point",
                           formats, show_plot=show_plot, plausible=plausible)
        REPORT.paragraph(
            f"How to read it. Each row is one delta for one group: 'imputed' is "
            f"how many censored patients received a death time, and HR is the "
            f"adjusted site hazard ratio after they did. At delta = 1 the HR "
            f"should be close to the one actually estimated "
            f"({tipping.attrs['observed']:.4f}, dotted line in the figure). The "
            f"number to look for is the delta at which the HR reaches 1 (dashed "
            f"line): how different the lost patients would have to be for the "
            f"whole site effect to be a product of censoring. The shaded band, "
            f"delta between {1 / plausible:.2g} and {plausible:g}, is the range "
            f"this run was told to treat as plausible (--tipping-plausible, a "
            f"judgement rather than an established threshold). If the HR reaches 1 "
            f"inside the band, lost patients need only have died up to "
            f"{plausible:g} times as fast as comparable patients who stayed for "
            f"censoring to account for the whole site effect, so the site effect "
            f"cannot be told apart from informative censoring. If it reaches 1 "
            f"only outside the band, or not at all, censoring would have to be "
            f"implausibly informative to explain it. A group with few censored "
            f"patients barely moves the HR whatever delta is, because there is "
            f"little to impute.")
        for label, delta in tipping.attrs["null_delta"].items():
            if delta is None:
                line = (f"Censored patients of {label}: the site HR does not reach 1 "
                        f"anywhere in delta = {min(deltas):.2g}-{max(deltas):g}, the "
                        f"whole range tested.")
            else:
                inside = 1 / plausible <= delta <= plausible
                line = (f"Censored patients of {label}: the site HR reaches 1 at "
                        f"delta = {delta:.2f}, "
                        + (f"inside the band. If they died about {delta:.1f} times "
                           f"as fast as comparable patients who stayed, there "
                           f"would be no site effect at all."
                           if inside else "outside the band."))
            print(line)
            REPORT.paragraph(line)
        REPORT.paragraph(
            "What this can and cannot conclude. Inside the band, censoring is a "
            "sufficient explanation, so the site effect cannot be attributed to "
            "entry point. Outside the band, censoring is an unlikely sole "
            "explanation, but that does not make the remainder entry point: it "
            "rules out one source, not all others. The remainder could equally be "
            "case-mix nobody recorded (KPS is missing for all of UCSF), a genuine "
            "difference in treatment or care after the clock starts, or chance. "
            "And the analysis only has something to explain when the adjusted site "
            "effect is itself distinguishable from zero; when its confidence "
            "interval already covers HR = 1, the question is how easily censoring "
            "could have produced or hidden an effect, not where an established one "
            "came from. Finally, delta shifts every censored patient of a group "
            "equally. Censoring that is informative for some of them and not for "
            "others, as when some leave for hospice and others simply move, is "
            "not what the table describes.")

    subsection("Proportional hazards, every term of the adjusted model")
    design, _, _ = build_site_design(database, ladder_covariates)
    ph_frame = pd.concat(
        [design, database[["site"]],
         pd.to_numeric(database["OS (days)"], errors="coerce").rename("OS (days)"),
         pd.to_numeric(database["status"], errors="coerce").rename("status")],
        axis=1).dropna()
    ph_frame = ph_frame[ph_frame["OS (days)"] > 0]
    gt, varying = assess_proportional_hazards(
        ph_frame, RESULTS, "Site-diagnostics_non-proportional-terms", formats,
        label="pooled site comparison", show_plot=show_plot)
    out["proportional-hazards"] = gt
    if not varying.empty:
        out["non-proportional-terms"] = varying

    for name, table in out.items():
        if table is not None and not table.empty:
            table.to_csv(f"{RESULTS}/OS-stats/Site-diagnostics_{name}.csv", index=False)
    print(f"\nDiagnostic tables written to {RESULTS}/OS-stats/Site-diagnostics_*.csv")
    return out
