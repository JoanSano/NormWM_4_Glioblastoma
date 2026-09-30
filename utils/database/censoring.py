"""Follow-up and censoring diagnostics of the site groups."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sksurv.compare import compare_survival

from utils.database.config import (COHORTS, TIPPING_DELTAS, TIPPING_IMPUTATIONS,
                                   FOLLOWUP_HORIZONS_MONTHS, ADJUSTMENT_COVARIATES)
from utils.database.site_model import complete_case_frame, fit_cox
from utils.survival import (as_structured, breslow_cumulative_hazard, daysXmonth, km_curve,
                            km_median, restricted_mean)


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


def followup_completeness(data, site_labels, horizons=FOLLOWUP_HORIZONS_MONTHS,
                          site_col="site", cohort_col="cohort",
                          duration_col="OS (days)", status_col="status"):
    """How much of the follow-up each group owed by a horizon was actually observed.

    Args:
        data: Table holding the site, cohort and survival columns.
        site_labels: {code: name} used to name the site-group rows.
        horizons: Horizons, in months, at which completeness is measured.
        site_col: Column holding the site indicator.
        cohort_col: Column holding the cohort id; each cohort gets its own rows
            after the site groups, since one site group can mix a cohort that
            censors administratively with one that loses patients early. Skipped
            when absent.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    The reverse Kaplan-Meier answers how LONG a group was watched, not how
    COMPLETELY: it treats a death as censoring of follow-up, so a group with many
    early deaths looks poorly followed however diligently it was. With GBM's
    early mortality that bias is large. The person-time rates of Clark et al.
    and Xue et al. ask instead what fraction of the person-time owed by the
    horizon tau was observed, a death counting as complete follow-up:

      percentage  1 - fraction censored before tau (every dropout counted as lost
                  at time 0 -- the CONSORT convention, a floor)
      CCI         observed / potential person-time, dropouts owed the full tau
                  (Clark's completeness index; a lower bound on the true rate)
      SPT         dropouts credited with their observed time, everyone else with
                  tau (Xue's simplified person-time; a slight upper bound)
      FPT         observed person-time / N x restricted mean survival to tau,
                  i.e. the person-time owed had nobody dropped out, estimated
                  from the Kaplan-Meier curve (Xue's formal person-time)

    CCI and SPT bracket the true rate; FPT estimates it, under the same
    independent-censoring assumption every other estimate here makes. (Treating
    death as a competing risk for loss to follow-up, the other correction of the
    reverse Kaplan-Meier that Xue et al. suggest, adds nothing here: with loss the
    only cause of censoring it reduces exactly to the percentage method.)
    """
    blocks = [(site_labels.get(s, s), data[data[site_col] == s])
              for s in sorted(data[site_col].dropna().unique())]
    if cohort_col in data.columns:
        cohort_names = {spec["id"]: name for name, spec in COHORTS.items()}
        # Only inside a site group that pools several cohorts; a group of one
        # would repeat its own row
        pooled = data.groupby(site_col)[cohort_col].transform("nunique") > 1
        blocks += [(f"  {cohort_names.get(c, c)}", data[data[cohort_col] == c])
                   for c in sorted(data.loc[pooled, cohort_col].dropna().unique())]

    rows = []
    for label, block in blocks:
        block = block[block[duration_col].notna() & block[status_col].notna()]
        time = block[duration_col].to_numpy(float)
        event = (block[status_col] == 1).to_numpy()
        n = len(time)
        if not n:
            continue
        km_t, km_s, _ = km_curve(block, duration_col, status_col)
        flipped = block.assign(**{status_col: 1 - block[status_col]})
        rkm_t, rkm_s, _ = km_curve(flipped, duration_col, status_col)

        for months in horizons:
            tau = months * daysXmonth
            dropout = ~event & (time < tau)
            observed = np.minimum(time, tau).sum()
            owed = n * restricted_mean(km_t, km_s, tau)

            rows.append(dict(
                group=label, horizon_months=months, n=n,
                lost_before_horizon=int(dropout.sum()),
                percentage=1.0 - dropout.mean(),
                reverse_km=float(rkm_s[np.searchsorted(rkm_t, tau, side="right") - 1]),
                CCI=observed / np.where(dropout, tau, np.minimum(time, tau)).sum(),
                SPT=np.where(dropout, time, tau).sum() / (n * tau),
                FPT=observed / owed if owed > 0 else np.nan,
            ))
    return pd.DataFrame(rows)


def censoring_hazard_table(data, covariates, site_labels, site_col="site",
                           duration_col="OS (days)", status_col="status",
                           cohort_col="cohort", min_censored=10):
    """What predicts being censored, next to what predicts dying, per site group.

    Args:
        data: Table holding the site, survival and covariate columns.
        covariates: Keys of ADJUSTMENT_COVARIATES to model both hazards on.
        site_labels: {code: name} used to name the groups.
        site_col: Column holding the site indicator.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        cohort_col: Column holding the cohort id, used only to say which cohorts
            the complete cases of each group come from. Skipped when absent.
        min_censored: Fewest censorings a group needs before its censoring
            hazard is modelled at all.

    Two Cox models per group on the same complete-case sample: one with the event
    indicator flipped, so censoring is the event, and the ordinary one for death.
    Censoring that depends on the covariates is not a failure of the adjusted
    site model -- conditional on those covariates it is still independent, so the
    adjusted hazard ratio stays consistent -- but it does bias every marginal
    Kaplan-Meier curve, crude or corrected. And when the covariates that predict
    censoring are the ones that predict death, with the same signs, the patients
    lost to follow-up are the sicker ones, which is the pattern that makes it
    plausible that censoring also depends on a prognostic factor nobody recorded.

    Returns one row per group and term, `same_side` saying whether the two
    hazard ratios fall on the same side of 1; `attrs["global"]` holds, per group,
    the likelihood-ratio test of the censoring model against the empty one as
    (chi2, df, p), and `attrs["composition"]`, per group, which cohorts its
    complete cases come from and which covariate removes the rest.
    """
    rows, global_tests, composition = [], {}, {}
    cohort_names = {spec["id"]: name for name, spec in COHORTS.items()}
    for site in sorted(data[site_col].dropna().unique()):
        label = site_labels.get(site, site)
        block = data[data[site_col] == site]
        frame = complete_case_frame(block, covariates, duration_col, status_col)

        if cohort_col in block.columns:
            parts = []
            for c in sorted(block[cohort_col].dropna().unique()):
                members = block[block[cohort_col] == c]
                kept = int(members.index.isin(frame.index).sum())
                part = f"{cohort_names.get(c, c)} {kept} of {len(members)}"
                absent = [ADJUSTMENT_COVARIATES[k]["label"] for k in covariates
                          if pd.to_numeric(members[ADJUSTMENT_COVARIATES[k]["column"]],
                                           errors="coerce").isna().all()]
                if absent:
                    part += f" ({', '.join(absent)} never recorded)"
                parts.append(part)
            composition[label] = "; ".join(parts)
        # A level the complete cases of this group never show is a column of zeros
        frame = frame.loc[:, frame.nunique() > 1]
        censored = int((frame[status_col] == 0).sum())

        base = dict(group=label, n=len(frame), censored=censored)
        if censored < min_censored:
            rows.append(dict(base, term="(not modelled)",
                             reason=f"only {censored} censored among the complete "
                                    f"cases; at least {min_censored} needed"))
            continue
        censoring, reason = fit_cox(frame.assign(**{status_col: 1 - frame[status_col]}),
                                    duration_col, status_col)
        death, reason_death = fit_cox(frame, duration_col, status_col)
        if censoring is None or death is None:
            rows.append(dict(base, term="(not modelled)", reason=reason or reason_death))
            continue

        lr = censoring.log_likelihood_ratio_test()
        global_tests[label] = (float(lr.test_statistic), int(lr.degrees_freedom),
                               float(lr.p_value))
        for term in censoring.params_.index:
            rows.append(dict(
                base, term=term,
                HR_censoring=float(censoring.summary.loc[term, "exp(coef)"]),
                p_censoring=float(censoring.summary.loc[term, "p"]),
                HR_death=float(death.summary.loc[term, "exp(coef)"]),
                p_death=float(death.summary.loc[term, "p"]),
                same_side=("yes" if (censoring.params_[term] > 0)
                           == (death.params_[term] > 0) else "no"),
                reason=None,
            ))
    table = pd.DataFrame(rows)
    table.attrs["global"] = global_tests
    table.attrs["composition"] = composition
    return table


def censoring_tipping_point(data, covariates, site_labels, deltas=TIPPING_DELTAS,
                            n_imputations=TIPPING_IMPUTATIONS, site_col="site",
                            duration_col="OS (days)", status_col="status"):
    """How much informative censoring the adjusted site effect can absorb.

    Args:
        data: Table holding the site, survival and covariate columns.
        covariates: Keys of ADJUSTMENT_COVARIATES of the adjusted site model.
        site_labels: {code: name} used to name the groups.
        deltas: Multipliers of the post-censoring hazard to walk.
        n_imputations: Imputed datasets pooled at every delta.
        site_col: Column holding the 0/1 site indicator.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Independent censoring cannot be tested, only stressed. Following Jackson et
    al. (2014), every censored subject of ONE site group is given a death time
    drawn from the adjusted Cox model itself, conditional on having survived to
    the censoring time, with the hazard after censoring multiplied by delta:

        H0(T*) = H0(c) + E / (delta * exp(x'beta)),   E ~ Exp(1)

    delta = 1 is independent censoring given the covariates and site; delta > 1
    says the patients lost to follow-up were dying faster than comparable
    patients who stayed. Each imputed dataset is refitted with the same model and
    the site coefficients pooled by Rubin's rules. The coefficients are redrawn
    from their sampling distribution, and the Breslow baseline recomputed under
    them, for every imputation, so the pooled interval carries the imputation
    model's own uncertainty. A draw past the last event time cannot be placed and
    stays censored there. The shift is applied to each group in turn -- which
    group lost its patients informatively is exactly what is not known.

    Returns one row per shifted group and delta; `attrs["null_delta"]` holds, per
    group, the delta at which the pooled site log HR crosses 0 (interpolated on
    log delta), or None when the grid does not bracket it, and
    `attrs["observed"]` the site HR of the unimputed fit.
    """
    frame = complete_case_frame(data, covariates, duration_col, status_col, site_col)
    model, reason = fit_cox(frame, duration_col, status_col)
    if model is None:
        table = pd.DataFrame()
        table.attrs.update(null_delta={}, observed=None, reason=reason)
        return table

    terms = list(model.params_.index)
    beta, cov = model.params_.to_numpy(), model.variance_matrix_.to_numpy()
    X = frame[terms].to_numpy(float)
    time = frame[duration_col].to_numpy(float)
    event = frame[status_col].to_numpy(int)
    grid = np.unique(time[event == 1])

    rows, null_delta = [], {}
    for shifted in sorted(frame[site_col].unique()):
        label = site_labels.get(shifted, shifted)
        # Censored before the last event time: anyone later has nothing to impute
        targets = np.flatnonzero((frame[site_col].to_numpy() == shifted)
                                 & (event == 0) & (time < grid[-1]))
        curve = []
        for delta in deltas:
            estimates, variances = [], []
            for _ in range(n_imputations):
                b = np.random.multivariate_normal(beta, cov)
                risk = np.exp(X @ b)
                H0 = breslow_cumulative_hazard(time, event, risk, grid)
                # H0 at each target's censoring time: the last step at or before it
                at_c = np.searchsorted(grid, time[targets], side="right") - 1
                start = np.where(at_c >= 0, H0[np.maximum(at_c, 0)], 0.0)
                draw = start + np.random.standard_exponential(len(targets)) / (
                    delta * risk[targets])
                step = np.searchsorted(H0, draw, side="left")
                placed = step < len(grid)

                imputed = frame.copy()
                rows_placed = imputed.index[targets[placed]]
                imputed.loc[rows_placed, duration_col] = grid[step[placed]]
                imputed.loc[rows_placed, status_col] = 1
                imputed.loc[imputed.index[targets[~placed]], duration_col] = grid[-1]

                fitted, _ = fit_cox(imputed, duration_col, status_col)
                if fitted is None:
                    continue
                estimates.append(float(fitted.params_[site_col]))
                variances.append(float(fitted.standard_errors_[site_col]) ** 2)

            m = len(estimates)
            if m < 2:
                continue
            pooled = float(np.mean(estimates))
            total = np.mean(variances) + (1 + 1 / m) * np.var(estimates, ddof=1)
            se = float(np.sqrt(total))
            p = float(2 * scipy_stats.norm.sf(abs(pooled / se)))
            curve.append((delta, pooled))
            rows.append(dict(
                censored_group=label, delta=delta, imputed=len(targets),
                imputations=m, logHR=pooled, HR=np.exp(pooled),
                HR_ci_low=np.exp(pooled - 1.96 * se),
                HR_ci_high=np.exp(pooled + 1.96 * se), p=p,
            ))

        null_delta[label] = None
        for (d0, b0), (d1, b1) in zip(curve, curve[1:]):
            if b0 == 0.0:
                null_delta[label] = d0
                break
            if b0 * b1 < 0:
                w = b0 / (b0 - b1)
                null_delta[label] = float(np.exp(np.log(d0) + w * (np.log(d1) - np.log(d0))))
                break

    table = pd.DataFrame(rows)
    table.attrs.update(null_delta=null_delta,
                       observed=float(np.exp(model.params_[site_col])), reason=None)
    return table
