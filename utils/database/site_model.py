"""Site effect: Cox estimation, adjustment, balance and proportional hazards."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test

from utils.database.config import ADJUSTMENT_COVARIATES, ADJUSTMENT_ORDER, ALPHA
from utils.statistics import llr_pvalue
from utils.survival import daysXmonth, split_at_event_times


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

    The baseline hazard is Breslow's estimator, asked for explicitly rather than
    left to the default. Ties in the partial likelihood are a different matter:
    lifelines implements Efron only, and there is no option to change it, so every
    fit here breaks ties by Efron whatever the baseline method says.
    """
    try:
        model = CoxPHFitter(baseline_estimation_method="breslow")
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


def complete_case_frame(data, covariates, duration_col="OS (days)", status_col="status",
                        site_col=None):
    """Design of `covariates`, the site column and the survival columns, complete cases only.

    Args:
        data: Table to read the covariate, site and survival columns from.
        covariates: Keys of ADJUSTMENT_COVARIATES, built by `build_site_design`.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        site_col: Group indicator placed between the design and the survival
            columns, or None to leave it out.

    Rows with any missing value, or a duration that is not strictly positive, are
    dropped: this is the frame a Cox model is fitted on as it stands. The columns
    come in that order -- design, site, duration, status -- because the fitted
    model reports its terms in column order.
    """
    design, _, _ = build_site_design(data, covariates)
    site = [data[[site_col]]] if site_col is not None else []
    frame = pd.concat([design, *site,
                       pd.to_numeric(data[duration_col], errors="coerce").rename(duration_col),
                       pd.to_numeric(data[status_col], errors="coerce").rename(status_col)],
                      axis=1).dropna()
    return frame[frame[duration_col] > 0]


def pair_frame(data, first, second, cohort_col="cohort", pair_col="pair"):
    """The subjects of two cohorts, with a 0/1 `pair_col` telling them apart.

    Args:
        data: Pooled table holding `cohort_col`.
        first: Cohort id coded 0, the reference of the pair.
        second: Cohort id coded 1, the cohort a pairwise coefficient rescales.
        cohort_col: Column holding the cohort id.
        pair_col: Name of the 0/1 indicator column added.
    """
    pair = data[data[cohort_col].isin([first, second])].copy()
    pair[pair_col] = pair[cohort_col].map({first: 0, second: 1})
    return pair


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

    |m1 - m0| / sqrt((s0^2 + s1^2) / 2), where m_g is the group mean and s_g^2
    the unbiased (n - 1) sample variance of group g. An indicator goes through
    the same formula: its mean is the proportion p and its unbiased variance is
    p(1 - p) n / (n - 1), rather than the plug-in p(1 - p) of Austin (2009) --
    one definition for both kinds of covariate, and the same SD the table's
    mean (SD) column reports. The two variances are averaged unweighted, so the
    larger group does not dominate the scale and equal variances are not
    assumed. Unlike a p-value it does not shrink as n grows, so it measures
    imbalance rather than the power to detect it. |SMD| above 0.10 is the
    conventional threshold for an imbalance worth adjusting for.
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


def proportional_hazards_table(model, frame, transform="rank"):
    """Grambsch-Therneau test for EVERY term of a fitted model, not only site.

    Args:
        model: A fitted lifelines CoxPHFitter.
        frame: The design it was fitted on.
        transform: Time transform the residuals are tested against.

    Returns a DataFrame with one row per term -- `term`, `test_statistic`, `p`,
    `p_bonferroni`, `violates` -- sorted with the worst offender first, and `n`,
    `events`, `transform`, `terms` and `bonferroni` in `.attrs`.

    `violates`, which decides which terms get described over time, is set on the
    BONFERRONI-adjusted p. Every term of the model is tested at once, so the raw p
    of the worst of them is not the evidence it appears to be, and a term followed
    up in error is not free: it produces a panel in the report and a caveat on the
    recommendation, both of which a reader takes at face value. The raw p is
    reported beside it so the more sensitive reading stays visible.

    The correction applies a single constant, so the site term must be
    proportional; but the coefficient it applies comes out of a model that also
    holds the case-mix covariates fixed, and a covariate whose own effect drifts
    with time makes that adjustment a misspecified one. Reporting the site row
    alone, as this script used to, hides exactly that.

    lifelines has no equivalent of R's `cox.zph` GLOBAL row, so `bonferroni` --
    the smallest p multiplied by the number of terms -- stands in for it. It is
    conservative rather than exact, and is labelled as such wherever it is shown.
    """
    test = proportional_hazard_test(model, frame, time_transform=transform)
    rows = test.summary.reset_index()
    rows = rows.rename(columns={rows.columns[0]: "term"})
    rows = rows[["term", "test_statistic", "p"]].copy()
    # Both are reported: the raw p is one term's evidence, the adjusted one is
    # that evidence read against the fact that every term of the model was tested
    rows["p_bonferroni"] = np.minimum(1.0, rows["p"] * len(rows))
    rows["violates"] = rows["p_bonferroni"] < ALPHA
    rows = rows.sort_values("p").reset_index(drop=True)
    rows.attrs.update(
        n=len(frame), events=int(model.event_observed.sum()), transform=transform,
        terms=len(rows),
        bonferroni=float(rows["p_bonferroni"].min()) if len(rows) else np.nan,
    )
    return rows


def time_varying_terms(frame, terms, duration_col="OS (days)", status_col="status",
                       max_cuts=1000):
    """Fit beta(t) = beta + theta*log(t) for the terms that failed the PH test.

    Args:
        frame: Design matrix plus duration and status, exactly as fitted.
        terms: Column names to describe over time, one model each.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        max_cuts: Passed to `split_at_event_times`.

    Returns (table, curves): a DataFrame of `term`, `beta`, `theta`, its CI, the
    likelihood-ratio chi2 and p, `ref_months`, `rows`, `reason`; and a dict of
    term -> {months, beta_t, lower, upper, beta, theta, ref_months} for plotting.

    One model per offending term rather than one model with every interaction at
    once: the terms are then each described against the same constant-coefficient
    baseline, and a term that cannot be fitted does not take the others with it.

    Two details decide whether this is an estimate or an artefact. The interaction
    is evaluated at each interval's START, the value every member of a risk set
    shares -- evaluating it at the interval end gives the subject who fails a
    systematically smaller time than the controls it is compared against, which
    manufactures a large negative theta out of data with no time trend at all.
    And there is no main effect of time: it is a function of time alone, so the
    baseline hazard absorbs it and the fit will not converge.
    """
    rows, curves = [], {}
    covariates = [c for c in frame.columns if c not in (duration_col, status_col)]
    long = split_at_event_times(frame.reset_index(drop=True), duration_col,
                                status_col, covariates, max_cuts)
    reference = float(frame.loc[frame[status_col] == 1, duration_col].median())
    log_time = np.log(long["start"].clip(lower=1.0)) - np.log(reference)
    fixed = ["id", "start", "stop", status_col] + covariates

    try:
        constant = CoxTimeVaryingFitter().fit(
            long[fixed], id_col="id", event_col=status_col,
            start_col="start", stop_col="stop", show_progress=False)
    except Exception as exc:
        return pd.DataFrame([dict(term=t, reason=f"{type(exc).__name__}") for t in terms]), {}

    observed = frame.loc[frame[status_col] == 1, duration_col]
    grid = np.linspace(max(observed.quantile(0.01), 1.0), observed.quantile(0.99), 200)
    u = np.log(grid) - np.log(reference)

    for term in terms:
        interaction = f"{term} x log(t)"
        block = long[fixed].copy()
        block[interaction] = long[term].values * log_time.values
        try:
            varying = CoxTimeVaryingFitter().fit(
                block, id_col="id", event_col=status_col,
                start_col="start", stop_col="stop", show_progress=False)
        except Exception as exc:
            rows.append(dict(term=term, reason=f"{type(exc).__name__}: "
                                                f"{str(exc).splitlines()[0][:80]}"))
            continue

        beta = float(varying.params_[term])
        theta = float(varying.params_[interaction])
        lr = 2.0 * (varying.log_likelihood_ - constant.log_likelihood_)
        covariance = varying.variance_matrix_.values
        names = list(varying.params_.index)
        b, t = names.index(term), names.index(interaction)
        variance = (covariance[b, b] + (u ** 2) * covariance[t, t]
                    + 2 * u * covariance[b, t])
        se = np.sqrt(np.clip(variance, 0.0, None))

        rows.append(dict(
            term=term, beta=beta, theta=theta,
            theta_ci_low=float(varying.summary.loc[interaction, "coef lower 95%"]),
            theta_ci_high=float(varying.summary.loc[interaction, "coef upper 95%"]),
            lr_chi2=float(lr),
            p=float(llr_pvalue(varying.log_likelihood_, constant.log_likelihood_, 1)),
            ref_months=reference / daysXmonth, rows=len(long), reason=""))
        curves[term] = dict(months=grid / daysXmonth, beta_t=beta + theta * u,
                            lower=beta + theta * u - 1.96 * se,
                            upper=beta + theta * u + 1.96 * se,
                            beta=beta, theta=theta,
                            ref_months=reference / daysXmonth)
    return pd.DataFrame(rows), curves
