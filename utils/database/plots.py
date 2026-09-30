"""Figures of the pooled database: survival curves, the site effect and its diagnostics."""

import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test
from lifelines.utils.lowess import lowess
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm

from utils.database.censoring import reverse_km_followup
from utils.database.config import ALPHA, DEFAULT_TIPPING_PLAUSIBLE
from utils.database.site_model import build_site_design, fit_cox
from utils.formatting import fmt_p_inline, fmt_p_phrase
from utils.report import save_figure
from utils.survival import as_structured, at_risk_and_censored, daysXmonth, draw_at_risk_table, km_curve


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
        ax.text(0.70, 0.90, r"$\chi^2 =$" + f"{chisquared:.4f}, "
                + f"\n{fmt_p_phrase(p_val)}",
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
                   f"chi-squared of {float(chisquared):.4f} with p-value of "
                   f"{fmt_p_inline(float(p_val))} (two-sided log-rank test)")
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
             r"$\chi^2 =$" + f"{chisquared:.4f}, \n{fmt_p_phrase(p_val)}"
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
    # A permutation p-value cannot be smaller than one draw out of n_perms, and
    # reporting the 0 that np.mean returns would claim a precision the test lacks
    p_value = np.mean(np.array(pop) >= c_index)
    p_shown = (f"p < {1.0 / n_perms:.2g}" if p_value == 0 else f"p = {p_value:.4g}")
    print(f"C-index = {c_index:.6f} (permutation {p_shown}, {n_perms} draws)")
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
        ax3.set_xlabel(f"rank-transformed time\n"
                       f"(GT={rank_stat:.4f}; {fmt_p_phrase(p_val_GT)})\n"
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
    ax2.text(0.70, 0.90, r"$\chi^2 =$" + f"{chisquared:.4f}, "
             + f"\n{fmt_p_phrase(p_val)}",
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


def plot_time_varying_terms(curves, table, RESULTS, stem, formats, show_plot=True,
                            title="Terms whose effect is not proportional over time"):
    """One panel per term that failed the PH test: its fitted beta(t) with a band.

    Args:
        curves: The dict `time_varying_terms` returned.
        table: The DataFrame it returned, for each term's theta and p.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.
        title: Figure title.

    A band that stays flat and covers the constant means the term is proportional
    after all; a sloped band shows the shape the single coefficient is averaging
    over.
    """
    if not curves:
        return
    terms = list(curves)[:4]
    fig, axes = plt.subplots(1, len(terms), figsize=(5.5 * len(terms), 4.6),
                             squeeze=False)
    for ax, term in zip(axes[0], terms):
        c = curves[term]
        row = table[table["term"] == term].iloc[0]
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
        ax.fill_between(c["months"], c["lower"], c["upper"], color="tab:blue",
                        alpha=0.18, zorder=2)
        ax.plot(c["months"], c["beta_t"], color="tab:blue", linewidth=2.5, zorder=3)
        ax.axhline(c["beta"], color="tab:purple", linewidth=1.5, zorder=4)
        ax.axvline(c["ref_months"], color="tab:purple", linewidth=0.8,
                   linestyle=":", zorder=1)
        ax.set_title(f"{term}\n"+r'$\theta$'+f" = {row['theta']:+.4f} "
                     f"({row['theta_ci_low']:+.3f} to {row['theta_ci_high']:+.3f}), "
                     f"{fmt_p_phrase(row['p'])}", fontsize=9)
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("log hazard ratio", fontsize=10)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, RESULTS, stem, formats)
    plt.show() if show_plot else plt.close(fig)


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
                fmt="none", ecolor="black", capsize=3)
    # Filled square when the 95% CI excludes 0, hollow otherwise.
    significant = (rows["logHR_ci_low"] > 0) | (rows["logHR_ci_high"] < 0)
    ax.scatter(rows["logHR"], y, marker="s", s=40, edgecolors="black", zorder=3,
               facecolors=np.where(significant, "black", "white"))
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  (n={n})" for m, n in zip(rows["model"], rows["n"])], fontsize=9)
    ax.set_xlabel("log hazard ratio of site=1 vs site=0 (95% CI)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, RESULTS, stem, formats)
    plt.show() if show_plot else plt.close(fig)


def plot_reverse_km(data, site_labels, RESULTS, stem, formats, show_plot=True,
                    site_col="site", duration_col="OS (days)", status_col="status",
                    colors=("tab:green", "salmon"),
                    title="Follow-up by site group (reverse Kaplan-Meier)"):
    """Reverse Kaplan-Meier curves, one per site group, with the log-rank test.

    Args:
        data: Table holding the site and survival columns.
        site_labels: {code: name} used in the legend.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.
        site_col: Column holding the site indicator.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        colors: One colour per site group, the same as the site-effect figure.
        title: Figure title.

    The curve is the probability of still being under follow-up: it drops at a
    censoring and a death leaves it untouched. Where it crosses 0.5 is the median
    potential follow-up of the table above; a group that censors almost nobody
    stays near 1 however many of its patients die.

    The rows beneath are cumulative counts before each month: patients
    right-censored, which are the steps of the curve, and in brackets the deaths,
    which leave it without being losses -- the reverse of the at-risk table under
    an ordinary survival curve.
    """
    months = list(range(0, 121, 10))
    sites = sorted(data[site_col].dropna().unique())
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    counts = []
    for i, site in enumerate(sites):
        block = data[(data[site_col] == site) & data[duration_col].notna()
                     & data[status_col].notna()]
        before = [block[duration_col] < m * daysXmonth for m in months]
        counts.append(([int((b & (block[status_col] == 0)).sum()) for b in before],
                       [int((b & (block[status_col] == 1)).sum()) for b in before]))
        flipped = block.assign(**{status_col: 1 - block[status_col]})
        time, prob, conf_int = km_curve(flipped, duration_col, status_col)
        color = colors[i % len(colors)]
        ax.step(time / daysXmonth, prob, where="post", color=color, linewidth=2,
                label=f"{site_labels.get(site, site)} (n={len(block)}, "
                      f"{int((block[status_col] == 0).sum())} censored)")
        ax.fill_between(time / daysXmonth, conf_int[0], conf_int[1], alpha=0.10,
                        step="post", color=color)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")

    table = reverse_km_followup(data, site_labels, site_col, duration_col, status_col)
    if "censoring_logrank" in table.attrs:
        chi2, p_val = table.attrs["censoring_logrank"]
        ax.text(0.97, 0.97, r"$\chi^2 =$" + f"{chi2:.4f},\n{fmt_p_phrase(p_val)}",
                transform=ax.transAxes, fontsize=10, ha="right", va="top",
                bbox=dict(boxstyle="round", alpha=0.1),
                color="red" if p_val <= ALPHA else "black")
    # Room below 0 for the table, as under the survival curves
    ax.set_ylim([-(0.085 + 0.06 * len(sites)), 1.05])
    # A cohort with administrative censoring runs out to many years; past ten
    # the curves say nothing the table does not
    ax.set_xlim([-5, 125])
    ax.set_xticks(months)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, months[-1])
    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Probability of still being under follow-up", fontsize=11)
    ax.legend(loc="center right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    draw_at_risk_table(ax, months, counts, [colors[i % len(colors)] for i in range(len(sites))],
                       header="No. right-censored (deaths)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, RESULTS, stem, formats)
    plt.show() if show_plot else plt.close(fig)


def plot_tipping_point(tipping, RESULTS, stem, formats, show_plot=True,
                       plausible=DEFAULT_TIPPING_PLAUSIBLE,
                       colors=("tab:green", "salmon"),
                       title="How much informative censoring the site effect absorbs"):
    """Pooled site HR against delta, one line per group whose censoring is shifted.

    Args:
        tipping: Table from `censoring_tipping_point`.
        RESULTS: Results directory; the figure lands in its OS-stats/.
        stem: File name of the figure, without extension.
        formats: Figure formats to write.
        show_plot: Display the figure as well as writing it.
        plausible: Upper edge b of the shaded band [1 / b, b] (--tipping-plausible).
        colors: One colour per site group, the same as the site-effect figure.
        title: Figure title.

    The shaded band is the range of delta the reader has declared plausible; the
    dotted line is the site HR with no imputation at all.
    """
    if tipping.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.axvspan(1 / plausible, plausible, color="gray", alpha=0.12,
               label=f"plausible delta, {1 / plausible:.2g}-{plausible:g} "
                     f"(--tipping-plausible)")
    for i, (group, rows) in enumerate(tipping.groupby("censored_group", sort=False)):
        color = colors[i % len(colors)]
        ax.plot(rows["delta"], rows["HR"], marker="o", color=color, linewidth=2,
                label=f"censored patients of {group} shifted "
                      f"({int(rows['imputed'].iloc[0])} imputed)")
        ax.fill_between(rows["delta"], rows["HR_ci_low"], rows["HR_ci_high"],
                        color=color, alpha=0.15)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(tipping.attrs["observed"], color="black", linewidth=0.8, linestyle=":",
               label=f"observed site HR ({tipping.attrs['observed']:.3f})")
    ax.axvline(1.0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    deltas = sorted(tipping["delta"].unique())
    ax.set_xticks(deltas)
    ax.set_xticklabels([f"{d:.3g}" for d in deltas], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("delta: hazard of the censored patients after censoring, "
                  "relative to comparable patients", fontsize=10)
    ax.set_ylabel("Adjusted site HR (95% CI)", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    # Room underneath for the legend, which would otherwise sit on the band
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    save_figure(fig, RESULTS, stem, formats)
    plt.show() if show_plot else plt.close(fig)
