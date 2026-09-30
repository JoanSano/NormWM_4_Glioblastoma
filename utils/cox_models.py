"""Cox models of the markers: design frames, fits, forest plots and leave-one-cohort-out validation."""

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, LogLogisticAFTFitter, WeibullAFTFitter

from utils.statistics import bootstrap_cindex

# The site-corrected survival column createDatabase.py writes
CORRECTED_DURATION = "OS (days) - corrected"


# ---------------------------------------------------------------------------
# Design frames
# ---------------------------------------------------------------------------
def standardization_params(train, covariates, continuous, categorical):
    """(center, scale) per covariate, estimated on `train` only so it can be reused out of sample.

    Args:
        train: Frame the parameters are estimated on.
        covariates: Covariates to rescale; any in neither list below is left as is.
        continuous: Covariates z-scored with their mean and sample SD (n - 1).
        categorical: Coded covariates mapped onto [-1, 1] by the midpoint and half-range
            of their codes (sex 0/1 -> -1/1; MGMT, EOR 0/1/2 -> -1/0/1). A ±1 coding of a
            balanced binary covariate has an SD close to 1, so its HR is on roughly the
            same footing as a per-SD HR; the HR of the full contrast is then its square.
    """
    params = {}
    for c in covariates:
        if c in continuous:
            params[c] = (train[c].mean(), train[c].std(ddof=1))
        elif c in categorical:
            lo, hi = train[c].min(), train[c].max()
            params[c] = ((hi + lo) / 2, (hi - lo) / 2)
    return params


def apply_standardization(df, params):
    """A copy of `df` with every covariate in `params` rescaled as (x - center) / scale.

    Args:
        df: Frame to rescale.
        params: {column: (center, scale)}, as `standardization_params` returns.
    """
    out = df.copy()
    for c, (center, scale) in params.items():
        out[c] = (out[c] - center) / scale
    return out


def cox_frame(df, covariates, duration_col, event_col, strata=None, standardize=False,
              continuous=(), categorical=()):
    """Complete-case frame for one Cox model: outcome, covariates and the strata column.

    Args:
        df: Source table.
        covariates: Model covariates.
        duration_col: Survival time column.
        event_col: Event indicator column.
        strata: List with the stratification column, or None.
        standardize: Rescale the covariates within this subset (`standardization_params`).
        continuous: Covariates z-scored when standardizing.
        categorical: Covariates mapped onto [-1, 1] when standardizing.
    """
    out = df[[duration_col, event_col, *covariates, *(strata or [])]].dropna().copy()
    if standardize:
        out = apply_standardization(
            out, standardization_params(out, covariates, continuous, categorical))
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_cox(frame, covariates, duration_col, event_col, strata=None, n_bootstrap=1000, seed=42):
    """Fit a Breslow Cox model and collect what the comparison needs from it.

    Args:
        frame: Design frame, as `cox_frame` returns.
        covariates: Model covariates.
        duration_col: Survival time column.
        event_col: Event indicator column.
        strata: List with the stratification column, or None.
        n_bootstrap: Resamples for the C-index confidence interval.
        seed: Seed of the resampling.

    The C-index is pooled over all patients, strata included, which is also what
    lifelines' `score` computes; the bootstrap resamples the fitted risks, so its
    interval reflects sampling of the patients, not refitting (it is not
    optimism-corrected).
    """
    model = CoxPHFitter(baseline_estimation_method="breslow")
    with warnings.catch_warnings():
        # Stratified fits and scores regroup the rows internally and warn about it; harmless
        warnings.filterwarnings("ignore", message="DataFrame Index is not unique")
        model.fit(frame, duration_col=duration_col, event_col=event_col, strata=strata)
        cindex = model.score(frame, scoring_method="concordance_index")
    (boot_mean, boot_lo, boot_hi), _ = bootstrap_cindex(
        model, frame, [duration_col, event_col, *covariates], status=event_col,
        survival=duration_col, n_bootstrap=n_bootstrap, alpha_CI=0.05, seed=seed)
    return dict(
        model=model,
        n=len(frame),
        events=int(frame[event_col].sum()),
        cindex=cindex,
        cindex_boot=(boot_mean, boot_lo, boot_hi),
        loglik=model.log_likelihood_,
        aic=model.AIC_partial_,
    )


def coefficient_table(model):
    """HR, 95% CI and p-value per covariate, in the model's parameter order.

    Args:
        model: Fitted lifelines Cox model.
    """
    s = model.summary
    return pd.DataFrame({
        "covariate": s.index,
        "log-HR": s["coef"].values,
        "HR": s["exp(coef)"].values,
        "HR 95% CI lower": s["exp(coef) lower 95%"].values,
        "HR 95% CI upper": s["exp(coef) upper 95%"].values,
        "p": s["p"].values,
        "CI excludes 1": ((s["exp(coef) lower 95%"] > 1) | (s["exp(coef) upper 95%"] < 1)).values,
    })


# ---------------------------------------------------------------------------
# Forest plots
# ---------------------------------------------------------------------------
def fill_nonsignificant(model, ax, color="salmon"):
    """Fill the square marker of covariates whose HR CI crosses 1 (lifelines forest plot).

    Args:
        model: Fitted lifelines Cox model already drawn on `ax` by `model.plot`.
        ax: Axes of that plot.
        color: Fill of the non-significant markers.
    """
    log_hr = model.params_.values
    ci = model.confidence_intervals_.values  # log-HR scale: lower, upper
    for y, i in enumerate(np.argsort(log_hr)):  # same row order as model.plot()
        if ci[i, 0] <= 0 <= ci[i, 1]:
            ax.plot(np.exp(log_hr[i]), y, marker="s", markerfacecolor=color,
                    markeredgecolor="k", markeredgewidth=1.25, linestyle="none", zorder=3)
    return ax


def draw_forest(ax, model, labels, title):
    """Hazard-ratio forest plot of one model, non-significant covariates filled salmon.

    Args:
        ax: Axes to draw on.
        model: Fitted lifelines Cox model.
        labels: {parameter name: tick label}; a name missing from it is shown as is.
        title: Axes title.

    Tick labels are looked up by parameter name in lifelines' own row order, so they
    cannot drift out of step with the rows the way a hand-ordered list can.
    """
    model.plot(hazard_ratios=True, ax=ax)
    fill_nonsignificant(model, ax)
    names = model.params_.index[np.argsort(model.params_.values)]
    ax.set_yticklabels([labels.get(n, n) for n in names])
    ax.set_ylim([-1, len(names)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontweight="bold", fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Leave-one-cohort-out validation
# ---------------------------------------------------------------------------
def cv_duration_column(data, duration_col, stratify_for):
    """Survival column for the cross-validation, which never stratifies.

    Args:
        data: Source table.
        duration_col: Survival column of the in-sample models.
        stratify_for: Stratification column of the in-sample models, or None.

    A held-out cohort would be a stratum without a baseline hazard, and the AFT
    fitters take no strata. When the in-sample models stratify by cohort or site,
    the site difference is instead handled by the site-corrected survival, if the
    table has it. Either way a warning says what was done.
    """
    if stratify_for not in ("cohort", "site"):
        return duration_col
    if CORRECTED_DURATION in data.columns:
        warnings.warn(f"stratify_for={stratify_for!r} is not applied in cross-validation; "
                      f"using the site-corrected survival {CORRECTED_DURATION!r} instead.")
        return CORRECTED_DURATION
    warnings.warn(f"stratify_for={stratify_for!r} is not applied in cross-validation and no "
                  f"site-corrected survival column is present; using {duration_col!r} unadjusted.")
    return duration_col


CV_FITTERS = {
    "Cox": lambda: CoxPHFitter(baseline_estimation_method="breslow"),
    "Weibull AFT": WeibullAFTFitter,
    "LogLogistic AFT": LogLogisticAFTFitter,
}


def leave_one_cohort_out(data, covariates, duration_col, event_col, standardize=False,
                         continuous=(), categorical=(), cohort_col="cohort"):
    """Fit on every cohort but one, evaluate on the one left out, for each fitter.

    Args:
        data: Source table.
        covariates: Model covariates.
        duration_col: Survival time column.
        event_col: Event indicator column.
        standardize: Rescale the covariates. The parameters are estimated on the
            training cohorts and applied unchanged to the held-out one.
        continuous: Covariates z-scored when standardizing.
        categorical: Covariates mapped onto [-1, 1] when standardizing.
        cohort_col: Column identifying the cohorts.

    One row per fitter and held-out cohort: training and test C-index, and the root
    mean squared error of the predicted expected survival on the test patients whose
    death was observed (censored times are lower bounds, not targets).
    """
    cols = [duration_col, event_col, *covariates]
    rows = []
    for name, make_fitter in CV_FITTERS.items():
        for cohort in sorted(data[cohort_col].dropna().unique()):
            test = data.loc[data[cohort_col] == cohort, cols].dropna()
            train = data.loc[data[cohort_col] != cohort, cols].dropna()
            if standardize:
                params = standardization_params(train, covariates, continuous, categorical)
                train = apply_standardization(train, params)
                test = apply_standardization(test, params)
            fitter = make_fitter().fit(train, duration_col=duration_col, event_col=event_col)
            events = test[test[event_col] == 1]
            predicted = fitter.predict_expectation(events)
            rows.append(dict(
                fitter=name,
                held_out=cohort,
                n_test=len(test),
                train_cindex=fitter.score(train, scoring_method="concordance_index"),
                test_cindex=fitter.score(test, scoring_method="concordance_index"),
                rmse_days=float(np.sqrt(np.mean((predicted.values - events[duration_col].values) ** 2))),
            ))
    return pd.DataFrame(rows)
