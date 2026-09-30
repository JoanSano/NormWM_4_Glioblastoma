"""Per-cohort harmonisation, and the partition of the cohorts into site groups."""

import numpy as np
import pandas as pd

from utils.database.config import COHORTS, COVARIATES_OF_INTEREST, MAP_EOR
from utils.survival import daysXmonth, daysXweek


# ---------------------------------------------------------------------------
# Paths, site partition and output names
# ---------------------------------------------------------------------------
def cohort_paths(main_dir, name, idh, grade, stream_th):
    """Locations of the per-cohort pipeline outputs.

    Args:
        main_dir: Root directory holding one folder per cohort.
        name: Cohort key into COHORTS ("UCSF", "UPENN", "TCGA", "RHUH", "LUMIERE").
        idh: IDH status the per-cohort pipeline was run with; fills the {idh} slot
            of that cohort's `subdir` template.
        grade: WHO grade the pipeline was run with; fills the {grade} slot, used by
            UCSF-PDGM only.
        stream_th: Minimum streamline density the indices were extracted at; part
            of the tract-density filename, not a filter applied here.

    Returns a dict with "td" and "morph", plus "eor" for LUMIERE, whose extent of
    resection is recomputed from the longitudinal scans rather than distributed.
    """
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
    """Merge the tract-density and morphology tables on their shared clinical columns.

    Args:
        td: Tract-density table for one cohort.
        morph: Morphology table for the same cohort, already filtered the same way.

    The join keys are whatever columns the two tables have in common, so both must
    have been through the same inclusion filter or the merge silently drops rows.
    """
    shared = [c for c in td.columns if c in set(morph.columns)]
    return pd.merge(td, morph, on=shared)


def report_censoring(name, data):
    """Print how many subjects a cohort contributes and what fraction is censored.

    Args:
        name: Cohort name to print.
        data: Harmonised table with a 0/1 `status` column (1 = event observed).
    """
    n = data["status"].value_counts().sum()
    censored = (data["status"] == 0).sum()
    print(f"{name}: {n} subjects -- percentage of censoring: {round(100 * censored / n, 2)}%")


def eor_to_category(eor_series, threshold=90.0):
    """Convert continuous EOR (%) to GTR/STR/NaN, preserving missing values.

    Args:
        eor_series: Extent of resection as a percentage.
        threshold: Percentage at or above which a resection counts as gross total.

    A subject with no recorded percentage stays NaN rather than falling into the
    lower category, which `pd.cut` would otherwise do silently.
    """
    return pd.cut(
        eor_series,
        bins=[-np.inf, threshold, np.inf],
        labels=["STR", "GTR"],
        right=False,  # [threshold, inf) -> GTR, i.e. >= threshold
    ).astype(object).where(eor_series.notna(), other=np.nan)


def resolve_site_codes(names, reference=None):
    """Assign each selected cohort to the 0/1 group the correction acts on.

    Args:
        names: Cohort names pooled in this run.
        reference: Cohort names forming the reference group (site 0), or None to
            take the `site` field of COHORTS.

    Returns {cohort name: 0 or 1}. Which group is the reference is not cosmetic:
    site 0 keeps its survival times untouched and site 1's are rescaled by
    exp(logHR), so naming the reference chooses which group is being corrected
    *onto*. A reference covering every selected cohort leaves one group, which is
    the supported way of assembling the table with no correction at all.
    """
    if reference is None:
        return {n: COHORTS[n]["site"] for n in names}
    reference = set(reference)
    return {n: (0 if n in reference else 1) for n in names}


def is_default_partition(site_of):
    """Whether a resolved partition agrees with the `site` field of COHORTS.

    Args:
        site_of: {cohort name: 0 or 1}, as `resolve_site_codes` returns.

    Only the default partition may reuse the legacy "OTHERS" label, so a custom
    run cannot overwrite the figures of the canonical one.
    """
    return all(site == COHORTS[name]["site"] for name, site in site_of.items())


def build_site_labels(names, site_of=None):
    """Name each site group from the cohorts actually pooled into it.

    Args:
        names: Cohort names being pooled in this run, in any order.
        site_of: {cohort name: 0 or 1} from `resolve_site_codes`, or None for the
            default partition.

    The legacy name "OTHERS" is kept when a group holds every cohort of that site
    -- which is what the full run pools, and what every figure already on disk is
    named after. Any narrower selection is named after the cohorts in it, because
    a figure labelled "OTHERS" while only two of the four were pooled is both
    wrong and liable to overwrite the full run's figure. A partition that is not
    the default one never earns the legacy name, for the same reason: it would
    put a differently grouped run under the canonical run's file names.
    """
    site_of = resolve_site_codes(names) if site_of is None else site_of
    legacy = is_default_partition(site_of)
    labels = {}
    for site in sorted({site_of[n] for n in names}):
        selected = [n for n in names if site_of[n] == site]
        everything = [n for n in COHORTS if COHORTS[n]["site"] == site]
        if len(selected) == 1:
            labels[site] = selected[0]
        elif legacy and set(selected) == set(everything):
            labels[site] = "OTHERS"
        else:
            labels[site] = "+".join(sorted(selected, key=lambda n: COHORTS[n]["id"]))
    return labels


def default_output_stem(names, prefix="data-clinical_TD-tissues"):
    """Base name of the assembled table for a given cohort selection.

    Args:
        names: Cohort names being pooled in this run.
        prefix: Leading part of the file name, before the cohort description.

    `<prefix>_<N>-cohorts` is ambiguous the moment --cohorts is used: {UCSF, TCGA}
    and {UPENN, RHUH} are both "2-cohorts" and would overwrite each other -- and
    each other's keys-maps.json and figures -- in the same results directory. The
    legacy name is therefore kept only when the selection is the first N cohorts
    by ID, which is what every existing *_4-cohorts.csv and *_5-cohorts.csv on
    disk is, and any other selection is named after the cohorts in it.
    """
    ordered = sorted(names, key=lambda n: COHORTS[n]["id"])
    ids = [COHORTS[n]["id"] for n in ordered]
    if ids == list(range(len(ids))):
        return f"{prefix}_{len(ids)}-cohorts"
    return f"{prefix}_{'-'.join(ordered)}"


# ---------------------------------------------------------------------------
# Per-cohort harmonisation
# ---------------------------------------------------------------------------
def load_lumiere(paths):
    """LUMIERE: IDH-wildtype with known survival; every subject experienced an event.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs plus the recomputed extent-of-resection table.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
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
    """RHUH: IDH-wildtype, treatment-naive, with a known censoring status.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
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
    """TCGA-GBM: IDH-wildtype subset of the pre-operative collection.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
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
    """UCSF-PDGM: IDH-wildtype glioblastomas (WHO 2021) with known survival.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
    def include(df):
        """Rows this cohort contributes.

        Args:
            df: One of the cohort's two raw tables. Both are filtered by the same
                predicate, so the merge that follows joins matching rows only.
        """
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
    """UPENN-GBM: de novo glioblastoma; the pipeline tables are already IDH-wildtype.

    Args:
        paths: Mapping returned by cohort_paths, holding the tract-density and
            morphology CSVs.

    Returns the COVARIATES_OF_INTEREST columns only, on the cohort's own scale
    already converted to days and its codings already mapped to the shared ones.
    """
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
