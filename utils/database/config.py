"""Configuration of the pooled database: columns, encodings, cohorts, covariates, thresholds."""

import numpy as np


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
# Cohorts. `site` is the DEFAULT partition the site-effect correction acts on:
# 0 is the reference group, whose survival times are left untouched, and 1 the
# group whose times the fitted coefficient rescales. It encodes how survival was
# recorded, which for these five cohorts separates UCSF from the rest -- but that
# is a property of this particular selection, not of the method, so it is a
# default rather than a law. --site-reference names a different reference group,
# and the resolved partition travels with the output in the provenance JSON.
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
# Clinical covariates the site-effect model may adjust for. `scale` divides a
# continuous covariate before it enters the design; it is 1.0 throughout, so every
# hazard ratio reads per native unit (one year, one KPS point). `levels` names the
# dummy contrasts of a categorical one against its lowest-coded reference level.
# ---------------------------------------------------------------------------
ADJUSTMENT_COVARIATES = {
    "age":  dict(column="age",         kind="continuous",  scale=1.0,  label="Age (per year)"),
    "sex":  dict(column="sex",         kind="categorical", scale=1.0,  label="Sex",
                 levels={0: "Male", 1: "Female"}),
    "eor":  dict(column="eor",         kind="categorical", scale=1.0,  label="EOR",
                 levels={0: "Biopsy", 1: "Subtotal (<90%)", 2: "Gross total (>=90%)"}),
    "mgmt": dict(column="mgmt",        kind="categorical", scale=1.0,  label="MGMT",
                 levels={0: "Unmethylated", 1: "Intermediate", 2: "Methylated"}),
    "kps":  dict(column="kps (preop)", kind="continuous",  scale=1.0,  label="KPS (per point)"),
}


# Fixed entry order, so a ladder is comparable between runs whatever order the
# covariates were typed on the command line.
ADJUSTMENT_ORDER = ["age", "sex", "eor", "mgmt", "kps"]


# Ladder walked by the diagnostics when no adjustment was requested. KPS is left
# out: UCSF and LUMIERE record none at all, so including it costs 100% of two
# cohorts and turns a complete-case model into a UPENN-only one.
DEFAULT_LADDER = ["age", "sex", "eor", "mgmt"]


# Horizons, in months, at which the completeness of follow-up is measured. GBM
# median survival is 12-16 months, so these bracket the part of the curve every
# downstream model leans on.
FOLLOWUP_HORIZONS_MONTHS = (12, 24)


# Multipliers of the post-censoring hazard the tipping-point analysis walks, and
# the imputations pooled at each. delta = 1 is independent censoring.
TIPPING_DELTAS = (0.2, 1 / 3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)


TIPPING_IMPUTATIONS = 20


# Default of --tipping-plausible: the band [1 / b, b] of delta the report treats
# as a plausible departure from independent censoring -- here, patients lost to
# follow-up dying at up to twice (or half) the rate of comparable patients who
# stayed. A convention, not an established threshold, hence a command-line input.
DEFAULT_TIPPING_PLAUSIBLE = 2.0


# Thresholds the recommendation argues against. They are conventional rather than
# derived, and they are quoted in the report next to the number they judge, so a
# reader who disagrees with one can see exactly which sentence it produced.
ALPHA = 0.05            # a site term whose CI covers 0 is not evidence of a site effect


SMD_IMBALANCE = 0.10    # |SMD| above this is an imbalance worth adjusting for
