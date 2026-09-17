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

Example
-------
    python createDatabase.py /home/joan/Desktop/PROJECTS/Glioblastomas \
                             RESULTS-GBM_5-cohorts_Tissues \
                             --idh WT --grade IV --stream-th 0 --format pdf
"""

import argparse
import contextlib
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
# Cohorts. `site` encodes how survival was recorded, which is what the
# site-effect correction below acts on (UCSF differs from every other cohort).
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
# Helpers
# ---------------------------------------------------------------------------
def cohort_paths(main_dir, name, idh, grade, stream_th):
    """Locations of the per-cohort pipeline outputs."""
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
    """Merge the tract-density and morphology tables on their shared clinical columns."""
    shared = [c for c in td.columns if c in set(morph.columns)]
    return pd.merge(td, morph, on=shared)


def report_censoring(name, data):
    n = data["status"].value_counts().sum()
    censored = (data["status"] == 0).sum()
    print(f"{name}: {n} subjects -- percentage of censoring: {round(100 * censored / n, 2)}%")


def as_structured(event, time):
    """Right-censored survival data in the structured-array form scikit-survival expects."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", "bool"), ("time", "float")],
    )


def km_curve(data, duration_col, status_col):
    """Kaplan-Meier estimate with log-log confidence bands, anchored at (0, 1)."""
    time, survival_prob, conf_int = kaplan_meier_estimator(
        data[status_col] == 1, data[duration_col], conf_type="log-log"
    )
    time = np.insert(time, 0, 0)
    survival_prob = np.insert(survival_prob, 0, 1)
    conf_int = np.insert(conf_int, 0, 1, axis=1)
    return time, survival_prob, conf_int


def save_figure(fig, results, stem, formats):
    for fmt in formats:
        fig.savefig(f"{results}/OS-stats/{stem}.{fmt}", dpi=200, format=fmt)


def eor_to_category(eor_series, threshold=90.0):
    """Convert continuous EOR (%) to GTR/STR/NaN, preserving missing values."""
    return pd.cut(
        eor_series,
        bins=[-np.inf, threshold, np.inf],
        labels=["STR", "GTR"],
        right=False,  # [threshold, inf) -> GTR, i.e. >= threshold
    ).astype(object).where(eor_series.notna(), other=np.nan)


# ---------------------------------------------------------------------------
# Per-cohort harmonisation
# ---------------------------------------------------------------------------
def load_lumiere(paths):
    """LUMIERE: IDH-wildtype with known survival; every subject experienced an event."""
    def include(df):
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
    """RHUH: IDH-wildtype, treatment-naive, with a known censoring status."""
    def include(df):
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
    """TCGA-GBM: IDH-wildtype subset of the pre-operative collection."""
    def include(df):
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
    """UCSF-PDGM: IDH-wildtype glioblastomas (WHO 2021) with known survival."""
    def include(df):
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
    """UPENN-GBM: de novo glioblastoma; the pipeline tables are already IDH-wildtype."""
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
    """Kaplan-Meier curves of every cohort, with pairwise log-rank tests."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    OS_STATS = []
    GROUP_STATS = []
    nums = np.empty(shape=(len(cohort_ids),), dtype=object)
    for i, cohort in enumerate(cohort_ids):
        diag = full_data[full_data[covariate_col] == cohort]
        diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])]

        nums[i] = [(diag[duration_col] >= (t * daysXmonth)).sum() for t in months]

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
    chisquared, p_val, stats, covariance = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
    ax.text(0.70, 0.90, r"$\chi^2 =$" + f"{round(chisquared, 4)}, \np = {round(p_val, 4)}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val <= 0.05 else "black")

    # Numbers at risk, one row per cohort
    for i, t in enumerate(months):
        for k in range(len(cohort_ids)):
            ax.text(t - 2, -0.07 - 0.06 * k, f"{nums[k][i]}", transform=ax.transData,
                    fontsize=11, verticalalignment="top", color=colors[k])
    ax.text(-2, -0.01, "No. at risk", transform=ax.transData, fontsize=11,
            verticalalignment="top", color="black", fontweight="bold")
    ax.hlines(0, -5, months[-1] + 5, color="black", linewidth=0.5)
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
    ):
    """Quantify the survival difference between two groups, and correct for it.

    Left panels show the complementary log-log survival curves and the scaled
    Schoenfeld residuals of the Grambsch-Therneau test (i.e. whether the difference
    is a proportional-hazards one); the right panel shows the effect of rescaling
    the survival times of the second group by the fitted hazard ratio.

    Returns the log hazard ratio of the second group relative to the first. It is
    contingent on the {0, 1} recoding of the two groups, not on the original IDs.
    """
    cohorts = sorted(cohorts)

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
    ax1.text(0.05, 0.90, r"$\chi^2 =$" + f"{round(chisquared, 4)}, \np = {round(p_val, 4)}",
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

    # Plot Schoenfeld residuals
    lifelines_model = CoxPHFitter()
    data = full_data[[covariate_col, duration_col, status_col]].dropna()
    data = data[data[covariate_col].isin(cohorts)]
    lifelines_model.fit(data, duration_col=duration_col, event_col=status_col)
    test = proportional_hazard_test(lifelines_model, data, time_transform="all")  # Grambsch-Therneau
    print("\nResults from the Grambsch-Therneau test:")
    print(test.summary)
    rank_stat, p_val_GT = test.test_statistic[0], test.p_value[0]
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
    ax3.set_xlabel(f"rank-transformed time\n(GT={rank_stat:.4f}; p={p_val_GT:.4f})", fontsize=8)
    ax3.set_ylabel("scaled-Schoenfeld Residuals", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)

    # Post-correction
    OS_STATS_deSITE = []
    GROUP_STATS = []
    nums = np.empty(shape=(2,), dtype=object)
    for i, cohort in enumerate(cohorts):
        diag = full_data[full_data[covariate_col] == cohort]
        diag = diag[~np.isnan(diag[status_col]) & ~np.isnan(diag[duration_col])].copy()
        diag[duration_col] = diag[duration_col] * np.exp(Cmodel.coef_[0] * i)
        nums[i] = [(diag[duration_col] >= (t * daysXmonth)).sum() for t in months]

        time, survival_prob, conf_int = km_curve(diag, duration_col, status_col)
        ax2.step(time / daysXmonth, survival_prob, where="post", color=colors[i])
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

    for i, t in enumerate(months):
        ax2.text(t - 2, -0.07, f"{nums[0][i]}", transform=ax2.transData, fontsize=11,
                 verticalalignment="top", color=colors[0])
        ax2.text(t - 2, -0.13, f"{nums[1][i]}", transform=ax2.transData, fontsize=11,
                 verticalalignment="top", color=colors[1])
    ax2.text(-2, -0.01, "No. at risk", transform=ax2.transData, fontsize=11,
             verticalalignment="top", color="black", fontweight="bold")
    ax2.hlines(0, -5, months[-1] + 5, color="black", linewidth=0.5)
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

    fig.suptitle(f"{name_cohort[cohorts[0]]} vs. {name_cohort[cohorts[1]]}", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, RESULTS,
                f"Site-effects_Survival-times_{name_cohort[cohorts[0]]}-{name_cohort[cohorts[1]]}",
                formats)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print("+" * 40)
    return Cmodel.coef_[0]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class Tee:
    """Write to two streams at once, so a run is both shown and stored."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data):
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return self.stream.isatty()


@contextlib.contextmanager
def tee_stdout(path):
    """Mirror everything printed to stdout into `path` as well.

    Progress bars are left out: tqdm writes to stderr, which is not captured.
    """
    if path is None:
        yield
        return
    with open(path, "w") as handle:
        original = sys.stdout
        sys.stdout = Tee(original, handle)
        try:
            yield
        finally:
            sys.stdout = original


def resolve_log_path(log_arg, RESULTS):
    """None when --log was not given; otherwise the file the run is stored in."""
    if log_arg is None:
        return None
    path = log_arg or "createDatabase_log.txt"
    return path if os.path.isabs(path) else f"{RESULTS}/{path}"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args(argv=None):
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
    parser.add_argument("--pairwise", action="store_true",
                        help="Also inspect every pair of cohorts, not only the site effect. "
                             "Slow: it refits the permutation test for each pair.")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "svg", "both"],
                        help="Figure format (default: pdf).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures as well as writing them to disk.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the permutation and bootstrap draws.")
    parser.add_argument("--log", nargs="?", const="", default=None, metavar="FILE",
                        help="Also store everything printed by the run in a text file "
                             "(default: <RESULTS_DIR>/createDatabase_log.txt). Relative "
                             "paths are resolved under RESULTS_DIR.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    formats = ["pdf", "svg"] if args.format == "both" else [args.format]
    main_dir = os.path.abspath(args.main_dir)
    RESULTS = args.results_dir if os.path.isabs(args.results_dir) else f"{main_dir}/{args.results_dir}"
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(f"{RESULTS}/OS-stats", exist_ok=True)

    log_path = resolve_log_path(args.log, RESULTS)
    with tee_stdout(log_path):
        if log_path is not None:
            print(f"# createDatabase.py -- {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"# {' '.join(sys.argv)}\n")
        assemble_database(args, main_dir, RESULTS, formats)
    if log_path is not None:
        print(f"Log written to {log_path}")


def assemble_database(args, main_dir, RESULTS, formats):
    """Harmonise the cohorts, correct the site effect and write the pooled table."""
    # --- Per-cohort harmonisation ------------------------------------------
    names = sorted(set(args.cohorts), key=lambda n: COHORTS[n]["id"])
    tables = []
    for name in names:
        paths = cohort_paths(main_dir, name, args.idh, args.grade, args.stream_th)
        data = LOADERS[name](paths)
        data["cohort"] = COHORTS[name]["id"]
        data["site"] = COHORTS[name]["site"]  # Related to how survival was recorded
        report_censoring(name, data)
        tables.append(data)

    database = pd.concat(tables, ignore_index=True)
    cohort_ids = [COHORTS[n]["id"] for n in names]
    name_cohort = {COHORTS[n]["id"]: n for n in names}
    colors = [COHORTS[n]["color"] for n in names]
    n_cohort = {COHORTS[n]["id"]: len(t) for n, t in zip(names, tables)}

    # --- Survival before correction ----------------------------------------
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
        for i, j in itertools.combinations(cohort_ids, 2):
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
            )

    # --- Site correction ----------------------------------------------------
    # Survival in UCSF-PDGM is recorded differently from the remaining cohorts, so the
    # difference between the two 'site' groups is estimated and divided out.
    sites = sorted(database["site"].unique())
    if len(sites) == 2:
        n_site = database.groupby("site").size()
        logHR_site = inspect_survival_diffs_in_paired_cohorts(
            full_data=database,
            cohorts=[0, 1],
            RESULTS=RESULTS,
            name_cohort={0: "UCSF", 1: "OTHERS"},
            colors=["tab:green", "salmon"],
            N_cohorts=[n_site[0], n_site[1]],
            covariate_col="site",
            n_perms=args.n_perms,
            formats=formats,
            show_plot=args.show,
        )
        print("Effect of 'site' (log HR)", logHR_site, "in units of UCSF/OTHERS")
    else:
        logHR_site = 0.0
        print("\nOnly one 'site' group in the selected cohorts: no site correction applied.")

    database["OS (days) - corrected"] = database["OS (days)"] * np.exp(logHR_site * database["site"])

    # --- Save ---------------------------------------------------------------
    stem = args.output_name or f"data-clinical_TD-tissues_{len(cohort_ids)}-cohorts"
    database.to_csv(f"{RESULTS}/{stem}.csv", sep=",", index=False)
    database.to_csv(f"{RESULTS}/{stem}.tsv", sep="\t", index=False)
    with open(f"{RESULTS}/keys-maps.json", "w") as f:
        json.dump(KEYS_MAPS, f, indent=4)
    print(f"\nAssembled {len(database)} subjects from {len(cohort_ids)} cohorts --> {RESULTS}/{stem}.csv")

    # --- Survival after correction ------------------------------------------
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


if __name__ == "__main__":
    main()
