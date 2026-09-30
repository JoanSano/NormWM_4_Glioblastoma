"""Raw or corrected survival times: the verdict, argued from the run's own diagnostics."""

import pandas as pd

from utils.database.config import ALPHA, DEFAULT_TIPPING_PLAUSIBLE, SMD_IMBALANCE
from utils.formatting import fmt_p_phrase, section
from utils.report import REPORT


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


def censoring_caveats(diagnostics):
    """The recommendation's sentences about censoring, from this run's diagnostics.

    Args:
        diagnostics: The tables `report_site_diagnostics` returned.

    Returns a list of strings, empty when the censoring diagnostics raised nothing.
    """
    caveats = []

    completeness = diagnostics.get("followup-completeness")
    if completeness is not None and not completeness.empty:
        last = completeness[(completeness["horizon_months"] == completeness["horizon_months"].max())
                            & ~completeness["group"].str.startswith(" ")]
        if len(last) >= 2 and last["FPT"].max() - last["FPT"].min() > 0.05:
            worst = last.loc[last["FPT"].idxmin()]
            best = last.loc[last["FPT"].idxmax()]
            caveats.append(
                f"By {int(worst['horizon_months'])} months {worst['group']} observed "
                f"{100 * worst['FPT']:.1f}% of the person-time it owed and "
                f"{best['group']} {100 * best['FPT']:.1f}% (formal person-time "
                f"rate): the loss to follow-up is concentrated in one group.")

    censoring = diagnostics.get("censoring-hazard")
    if censoring is not None and "global" in censoring.attrs:
        for group, (_, _, p_val) in censoring.attrs["global"].items():
            if p_val >= ALPHA:
                continue
            terms = censoring[(censoring["group"] == group) & (censoring["p_censoring"] < ALPHA)]
            same = terms[terms["same_side"] == "yes"]
            described = ", ".join(
                f"{t} (censoring HR {h_c:.3f}, death HR {h_d:.3f})"
                for t, h_c, h_d in zip(terms["term"], terms["HR_censoring"], terms["HR_death"]))
            sentence = (f"Censoring in {group} depends on the covariates "
                        f"({fmt_p_phrase(p_val)}): {described}.")
            if len(same) and len(same) == len(terms):
                sentence += (
                    " For each, both hazard ratios lie on the same side of 1: the "
                    "patients more likely to be lost were also more likely to die, so "
                    "those lost to follow-up were the sicker ones. The adjusted site model "
                    "absorbs the part of this that runs through its covariates; the "
                    "Kaplan-Meier curves, crude and corrected, do not.")
            caveats.append(sentence)

    tipping = diagnostics.get("censoring-tipping-point")
    if tipping is not None and not tipping.empty:
        high = tipping.attrs.get("plausible", DEFAULT_TIPPING_PLAUSIBLE)
        low = 1.0 / high
        for group, delta in tipping.attrs["null_delta"].items():
            if delta is None:
                continue
            rows = tipping[tipping["censored_group"] == group]
            flips = rows[(rows["p"] < ALPHA)]
            flip_text = "; ".join(
                f"delta = {r.delta:.3g} (HR {r.HR:.3f}, {fmt_p_phrase(r.p)})"
                for r in flips.itertuples())
            sentence = (
                f"If the censored patients of {group} died at delta times the rate "
                f"the adjusted model predicts for comparable patients who stayed, "
                f"the site HR would reach 1 at delta = {delta:.2f}.")
            if flip_text:
                sentence += (f" The site effect would be significant at alpha = "
                             f"{ALPHA} for {flip_text}.")
            if low <= delta <= high:
                sentence += (
                    f" That is inside the band taken as plausible ({low:.2g}-{high:g}, "
                    f"--tipping-plausible), so censoring alone could account for the "
                    f"adjusted site effect, and it cannot be attributed to entry "
                    f"point.")
            else:
                sentence += (
                    f" That is outside the band taken as plausible ({low:.2g}-{high:g}, "
                    f"--tipping-plausible), so censoring alone is an unlikely "
                    f"explanation of the adjusted site effect -- which does not by "
                    f"itself make it entry point (see the tipping-point section).")
            caveats.append(sentence)
    return caveats


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
            f"(HR {crude['HR']:.3f}, {fmt_p_phrase(crude['p'])}) on all "
            f"{int(crude['n'])} subjects.")

    if rung is None:
        caveats.append(
            "No adjusted rung of the ladder could be fitted, so there is no evidence "
            "here that separates case-mix from entry point or censoring. Treat the "
            "correction as unverified.")
        return ("undetermined",
                "Undetermined: the adjusted models could not be fitted on this selection.",
                evidence, caveats)

    covers_zero = (pd.notna(rung["logHR_ci_low"]) and pd.notna(rung["logHR_ci_high"])
                   and rung["logHR_ci_low"] <= 0.0 <= rung["logHR_ci_high"])
    evidence.append(
        f"Adjusted site effect ({rung['model'].lstrip('+ ')}): log HR "
        f"{rung['logHR']:.4f} (95% CI {rung['logHR_ci_low']:.4f} to "
        f"{rung['logHR_ci_high']:.4f}, {fmt_p_phrase(rung['p'])}), fitted on "
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
            f"the crude comparison showed is accounted for by who the patients are, "
            f"and what remains is too small to attribute to entry point or to "
            f"censoring. Rescaling the outcome by it would remove prognostic signal "
            f"that belongs to the covariates, and a downstream model that also "
            f"adjusts for them would remove it twice.")
    else:
        verdict = "corrected"
        headline = (
            "The CORRECTED survival times are defensible: a site effect survives "
            "adjustment for case-mix.")
        evidence.append(
            f"The adjusted confidence interval excludes 0 at alpha = {ALPHA}, so a "
            f"difference remains after case-mix is held fixed. The corrected column "
            f"treats that remainder as an entry-point artefact; if the censoring "
            f"caveats below say informative censoring could produce it, that "
            f"attribution is not established. Stratifying the baseline hazard by "
            f"cohort is the equivalent alternative and needs no rescaling at all; "
            f"pick one of the two, never both.")

    # Caveats, which qualify either verdict
    if pd.notna(rung["ph_p"]) and rung["ph_p"] < ALPHA:
        caveats.append(
            f"Proportional hazards is rejected for the adjusted site term "
            f"({fmt_p_phrase(rung['ph_p'])}). A single multiplicative factor "
            f"is then the wrong description of the difference at every follow-up "
            f"time, and stratification is the safer route whatever is decided here.")
    ph_table = diagnostics.get("proportional-hazards")
    if ph_table is not None and not ph_table.empty:
        failed = ph_table.loc[ph_table["violates"] & (ph_table["term"] != "site"),
                              "term"].tolist()
        if failed:
            caveats.append(
                f"The site term is proportional, but {len(failed)} covariate(s) in "
                f"the adjustment are not ({', '.join(failed)}). The site "
                f"coefficient is therefore estimated while holding fixed a "
                f"covariate whose own effect drifts with follow-up, so it is an "
                f"average over a model that does not hold at every time. The "
                f"log-time fits report the drift for each of them.")
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
                f"(log-rank {fmt_p_phrase(p_censor)}): they were watched for "
                f"different lengths of time. That costs precision, not "
                f"unbiasedness, as long as censoring is independent -- which the "
                f"next points probe but cannot prove.")
    caveats.extend(censoring_caveats(diagnostics))
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
