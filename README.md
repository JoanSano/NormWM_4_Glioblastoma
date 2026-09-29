# NormWM_4_Glioblastoma

**Normative white matter approaches to human glioblastoma.**

Code to quantify how glioblastomas interact with the white matter architecture of the
human brain, and to test whether that interaction explains patient outcomes better than
the tumour's own size and location.

The central quantity is the **Lesion-Tract Density Index (L-TDI)**: rather than measuring
a tumour by its volume, each tumour compartment is intersected with a large *normative*
tractogram, and the streamlines passing through it are mapped back into the brain. The
resulting tract-density maps summarise, in a single number, how much of the brain's
structural wiring a lesion disturbs — including wiring far away from the lesion itself.
This reframes glioblastoma as a network disease rather than a focal one.

> This code is research code under active development. It is shared for transparency and
> reuse, and is **not yet intended as a standalone, turnkey solution**. Paths are largely
> hard-coded to the authors' storage layout and will need adapting.

---

## Table of contents

- [Overview](#overview)
- [Before you start](#before-you-start)
  - [Installation](#installation)
  - [Data](#data)
  - [Template and tractogram](#template-and-tractogram)
- [Step 1 — Spatial normalisation](#step-1--spatial-normalisation)
- [Step 2 — Tract-density mapping](#step-2--tract-density-mapping)
- [Step 3 — Database assembly](#step-3--database-assembly)
  - [What it does](#what-it-does)
  - [Running it](#running-it)
  - [What it writes](#what-it-writes)
  - [Reading the report](#reading-the-report)
  - [Choosing raw or corrected survival](#choosing-raw-or-corrected-survival)
- [Step 4 — Statistics and modelling](#step-4--statistics-and-modelling)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

The pipeline runs in four steps, in this order. Each step reads what the previous one
wrote, so they cannot be skipped or reordered.

| Step | Where it runs | Entry point | Produces |
|---|---|---|---|
| [1. Spatial normalisation](#step-1--spatial-normalisation) | once per cohort | `<COHORT>/normalize_MNI.sh` | Scans and tumour segmentations in MNI space |
| [2. Tract-density mapping](#step-2--tract-density-mapping) | once per cohort | `<COHORT>/TDMaps.sh` | Per-subject TDI, L-TDI and compartment volumes |
| [3. Database assembly](#step-3--database-assembly) | once, pooled | `createDatabase.py` | One harmonised clinical + imaging table, and a report |
| [4. Statistics and modelling](#step-4--statistics-and-modelling) | on the pooled table | `*_stats.py`, notebooks | Survival analyses and figures |

Steps 1–2 are shell pipelines that live in each cohort's folder, because every dataset
ships a different directory structure, file naming and metadata schema. Steps 3–4 are
cohort-agnostic and run from the repository root.

---

## Before you start

### Installation

Python 3.12 is recommended. Everything here was developed and tested inside a conda
environment, which is also the easiest way to satisfy the version floors in
`requirements.txt`:

```bash
git clone https://github.com/JoanSano/NormWM_4_Glioblastoma.git
cd NormWM_4_Glioblastoma
conda create -n normwm python=3.12
conda activate normwm
pip install -r requirements.txt
```

Optional extras (tumour segmentation, streamline conversion, consensus community
detection) are listed, commented out, at the bottom of `requirements.txt`.

Steps 1–2 also call the following neuroimaging tools directly. They are not
pip-installable and must be on your `PATH`:

| Software | Used for | Commands |
|---|---|---|
| [ANTs](https://github.com/ANTsX/ANTs) | Registration to MNI, bias correction | `antsRegistrationSyNQuick.sh`, `antsApplyTransforms`, `N4BiasFieldCorrection`, `ImageMath` |
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) | Mask arithmetic, compartment extraction | `fslmaths` |
| [MRtrix3](https://www.mrtrix.org/) | Streamline selection and tract-density mapping | `tckedit`, `tckmap` |

### Data

No patient data is distributed with this repository. All five cohorts are publicly
available and must be obtained from their original providers under their own data use
agreements.

| Cohort | Subjects | Source |
|---|---|---|
| UCSF-PDGM | 495 diffuse gliomas, 403 grade IV | [TCIA](https://doi.org/10.7937/tcia.bdgf-8v37) |
| UPENN-GBM | 630 de novo GBM | [TCIA](https://doi.org/10.7937/TCIA.709X-DN49) |
| TCGA-GBM | 135 with pre-operative scans | [TCIA](https://www.cancerimagingarchive.net/analysis-result/brats-tcga-gbm/) |
| RHUH-GBM | 40 GBM | [TCIA](https://doi.org/10.7937/4545-c905) |
| LUMIERE | 91 GBM, longitudinal | [figshare](https://doi.org/10.6084/m9.figshare.c.5904905) |

Counts are the full public releases; the analyses apply further inclusion criteria (IDH
status, grade, data completeness), so the modelled samples are smaller.

For scans without expert segmentations, `segmentation_TCGA_example.ipynb` shows inference
with the MONAI BraTS `SegResNet` bundle.

### Template and tractogram

Steps 1–2 need, in MNI ICBM 2009b NLIN ASYM space, the template itself and the normative
tractogram (`dTOR_full_tractogram.tck` and its TD map). Their locations are set by the
`MNI_TEMPLATE` variable at the top of each shell script. See
[Acknowledgements](#acknowledgements) for where to obtain them.

---

## Step 1 — Spatial normalisation

**What it does.** Registers each patient's structural MRI to the MNI ICBM 2009b NLIN ASYM
template with ANTs (`antsRegistrationSyNQuick.sh`), using a cost-function mask that
excludes the lesion so the tumour does not bias the warp. The tumour segmentations are
then carried into template space with the resulting transforms.

**Running it.** From a cohort folder, after editing `BASE_DIR` and `MNI_TEMPLATE` at the
top of the script:

```bash
cd UCSF-PDGM
./normalize_MNI.sh
python quality-control_registration-MNI.py     # flags failed registrations for review
```

Repeat for every cohort you intend to pool. RHUH-GBM registers its baseline timepoint with
`normalizeT0_MNI.sh` instead.

**Before moving on.** Look at the registrations the QC script flags. A failed warp puts
the tumour on the wrong tracts, and nothing downstream will notice.

---

## Step 2 — Tract-density mapping

**What it does.** Splits each MNI-space segmentation into tumour compartments (whole,
core, non-enhancing, enhancing, core+enhancing). For every compartment, MRtrix3
(`tckedit`) selects the normative streamlines traversing it and `tckmap` renders them as
a **lesion tract-density map**. Two indices are extracted per compartment:

| Index | Definition | Interpretation |
|---|---|---|
| **TDI** | Mean normative streamline density *within* the compartment mask | Local white matter density of the tissue the tumour occupies |
| **L-TDI** | Mean density of the map formed by the streamlines the compartment *intercepts* | Whole-brain disruption footprint — the non-local marker |

Compartment volumes are recorded alongside, so volume and L-TDI can later be compared
head to head.

**Running it.** From the same cohort folder, after editing `MAIN_DIR` and `MNI_TEMPLATE`:

```bash
# -g grade (UCSF-PDGM) / -i IDH status (other cohorts)
# -n subjects in parallel   -s min. streamline density   -k keep lesion .tck files
./TDMaps.sh -g IV -n 4 -s 0 -k 0          # UCSF-PDGM
./TDMaps.sh -i WT -n 4 -s 0 -k 0          # UPENN-GBM, TCGA-GBM, RHUH-GBM, LUMIERE-GBM
```

**What it writes**, per cohort, in its tract-density folder — the two files Step 3 reads:

| File | Contents | Written by |
|---|---|---|
| `demographics-TDMaps_streamTH-<s>.csv` | TDI and L-TDI per subject and compartment, with the cohort's clinical fields | `TDMaps-extraction.py` |
| `morphology-tissues.csv` | Compartment volumes per subject | `morphology-extraction.py` |

`<s>` is the `-s` threshold (default 0 everywhere), and **Step 3 must be given the same
value** (`--stream-th`, also default 0), or it will look for a file that does not exist.

---

## Step 3 — Database assembly

`createDatabase.py` is the least self-explanatory part of the repository, and the one most
likely to change a published number. Read this section before running it.

### What it does

1. **Pools the cohorts.** Reads each cohort's two Step 2 tables and applies the per-cohort
   inclusion criteria (IDH-wildtype, grade IV, treatment-naive, known survival).
2. **Harmonises the clinical variables.** Each cohort reports age, sex, KPS, extent of
   resection (EOR), MGMT, IDH, overall survival and censoring under its own schema and
   coding. The script converts survival to days and recodes sex, MGMT and EOR onto common
   integer keys.
3. **Estimates a site effect on survival.** The cohorts are split into two `site` groups:
   a reference group (UCSF-PDGM by default) and the rest. Cohorts can differ in *when*
   the survival clock starts, which is an artefact of record-keeping, and in *which
   patients* they enrolled, which is real prognostic information. The script fits the
   difference between the groups as a Cox log hazard ratio, optionally adjusted for
   case-mix, and writes a second survival column rescaled by it.
4. **Checks whether that correction is justified.** Covariate balance, an adjustment
   ladder, proportional-hazards tests, follow-up comparison and Kaplan–Meier curves — run
   on every invocation, with no flag to skip them.
5. **Recommends a survival column.** The run ends by saying whether to analyse the raw or
   the corrected survival times, and why.

Everything the run finds goes into an HTML report; how to read it is
[below](#reading-the-report). The methodological reasoning — what the correction assumes,
how missing covariates are handled, the proportional-hazards tests and their equations,
and the references — is written in the report itself, next to the numbers it produced,
not here.

### Running it

From the repository root. The first argument is the directory holding the cohort
folders; the second is the output directory (resolved under the first if relative).

```bash
# Recommended: condition the site effect on case-mix
python createDatabase.py /path/to/Glioblastomas RESULTS-GBM_5-cohorts_Tissues \
                         --idh WT --grade IV --stream-th 0 \
                         --adjust-covariates age sex eor mgmt

# Crude site effect (no covariates)
python createDatabase.py /path/to/Glioblastomas RESULTS-GBM_5-cohorts_Tissues \
                         --idh WT --grade IV --stream-th 0

# A subset of cohorts, with every pair of cohorts also inspected
python createDatabase.py /path/to/Glioblastomas RESULTS-GBM_2-cohorts \
                         --cohorts UCSF UPENN --adjust-covariates age sex eor mgmt --pairwise
```

The recommendation at the end of the run is based on the adjusted model, so
`--adjust-covariates` is the setting to use unless you specifically want the crude
correction in the table.

| Flag | What it does |
|---|---|
| `--idh`, `--grade`, `--stream-th` | Which Step 2 outputs to read: IDH status, WHO grade (UCSF-PDGM only) and the streamline-density threshold. `--stream-th` must match the `-s` given to `TDMaps.sh`. |
| `--cohorts` | Restricts the pool to a subset (default: all five). A selection that is not the first N cohorts by ID is named after its cohorts, so two subsets cannot overwrite each other. |
| `--adjust-covariates` | Covariates the site model conditions on: any of `age sex eor mgmt kps`. Default: none, i.e. the crude site effect. |
| `--site-reference` | Cohorts forming the reference group (site 0), whose survival times are never rescaled. Default: UCSF alone. Naming *every* selected cohort leaves one group and applies no correction. |
| `--ladder-covariates` | Covariates the adjustment ladder walks (default: `--adjust-covariates` if given, else `age sex eor mgmt`). |
| `--pairwise` | Also inspect every *pair* of cohorts, each with its own adjusted coefficient. Slow: it refits the permutation test per pair. |
| `--output-name` | Base name (`<stem>`) of the table and everything named after it (default: `data-clinical_TD-tissues_<N>-cohorts`). |
| `--n-perms`, `--seed` | Permutations for the concordance test, and their seed. |
| `--format`, `--show` | Figure format (`pdf`, `svg`, `both`), and whether to display figures as they are made. |
| `--log` | Rename the log file (default `createDatabase_log.txt`). The run is always logged; this only changes where. |
| `--verbose` | Mirror the log to the terminal. |

See `python createDatabase.py --help` for the full description.

### What it writes

**The terminal stays quiet.** A run announces where it is writing, and when it finishes
prints the files it produced and the recommended survival column:

```
createDatabase.py: pooling <cohorts> -> <results_dir>
  running quietly; the run is written to .../createDatabase_log.txt (--verbose to watch it here)

Assembled <N> subjects from <K> cohorts.
  table   .../<stem>.csv
  report  .../<stem>_report.html
  log     .../createDatabase_log.txt
  verdict <RAW | CORRECTED | UNDETERMINED> -- analyse '<column>'
```

Everything in between — sample sizes, fits, tests and warnings — goes to the log. Progress
bars still appear, on stderr, so a long run shows that it is alive.

In the output directory:

| File | Contents |
|---|---|
| `<stem>_report.html` | **Open this first.** Every figure, every diagnostic table, the recommendation and the method, in one self-contained file. Print it to PDF from the browser if needed. |
| `<stem>.csv` / `.tsv` | The pooled table, with both `OS (days)` and `OS (days) - corrected`, and the `site correction factor` each subject's time was multiplied by (divide by it to recover the raw time). |
| `createDatabase_log.txt` | The full run, line by line. The report points to it rather than embedding it. |
| `<stem>_site-correction.json` | Provenance: the coefficient and its CI, the model design, the site partition, and the recommended column (`recommended_outcome`). Quote this in a methods section. |
| `keys-maps.json` | The integer codes used for the categorical variables. |
| `OS-stats/` | The figures as `.pdf`/`.svg`, and every diagnostic table as `Site-diagnostics_*.csv`. |

### Reading the report

The report follows the order the analysis ran. What each part answers:

| Section | The question it answers |
|---|---|
| **Balance and missingness** | How different are the two site groups to begin with? An absolute SMD above 0.10 marks an imbalance worth adjusting for; the missingness columns show which covariates a group never records. |
| **Adjustment ladder** (+ forest plot) | How much of the site effect is case-mix? Each rung adds covariates, all on one fixed sample. A site coefficient that shrinks as covariates enter is being explained by who the patients were. |
| **Proportional hazards** | Does the model the coefficient comes from hold? One test per term, with the raw and the Bonferroni-adjusted p. |
| **Non-proportional terms** (figure) | Only when a term fails: how its effect drifts over follow-up. |
| **Follow-up (reverse KM)** | Were the groups followed for equally long? |
| **Site-effect figure** | Survival curves before and after rescaling, and the residual diagnostic for the applied model. |
| **Kaplan–Meier curves** | Survival per cohort, before and after correction. The rows beneath read `at risk (right-censored)`. |
| **Recommendation** | Which survival column to analyse, and what qualifies that verdict. |
| **Method, and the choices behind it** | The assumptions, equations and references. |

With `--pairwise`, the site-effect figure and the proportional-hazards section are
repeated for every pair of cohorts.

Hazard ratios are per native unit (one year of age, one KPS point). A p-value below the
last digit shown is printed as `p < 0.0001`, never `p = 0.0000`.

### Choosing raw or corrected survival

The rule the script applies:

> Fit the site effect adjusted for case-mix. **If its 95% confidence interval covers zero,
> use the raw survival times** and adjust or stratify for cohort downstream. If it
> excludes zero, the corrected column is defensible.

Practical consequences:

- **The verdict belongs to the pool, not to the method.** A different `--cohorts` or
  `--site-reference` can land the other way, so re-run rather than reuse a verdict.
- **The verdict can disagree with the column you applied.** The adjusted model is fitted
  on every run, so a crude correction can still end in a *raw* recommendation — and the
  report says so. Both columns are always written; nothing downstream is obliged to use
  the corrected one.
- **Use one remedy, not both.** Rescaling survival and stratifying the Cox baseline by
  cohort correct the same difference; applying both removes it twice.
- **Read the qualifications.** The recommendation lists what weakens it in that run:
  terms failing proportional hazards, a small complete-case sample, remaining imbalance,
  or differing follow-up.

---

## Step 4 — Statistics and modelling

**What it does.** Relates the markers to survival on the pooled table from Step 3.

| Entry point | Question it answers |
|---|---|
| `TDIndices_stats.py`, `LTDIndices_stats.py`, `Volumes_stats.py` | Per-compartment univariate survival statistics for TDI, L-TDI and volume |
| `LTDI-Volume_comparison.ipynb` | Is L-TDI or tumour volume the better prognostic marker, pre- and post-surgery? |
| `multivariate_survival.ipynb` | Cox models with clinical covariates; equivalence of L-TDI to TDI + volume |
| `anatomical_TractDensityMarkers.ipynb` | Anatomical localisation of the markers; site effects across cohorts |

Statistical inference is deliberately conservative and lives in `utils/statistics.py`:
selective inference across correlated compartment families via the **Benjamini–Bogomolov
(BB) procedure**, bootstrap confidence intervals and permutation tests for concordance
indices, bootstrapped median-survival differences, DeLong tests for AUC comparison, and
IPCW-corrected concordance for right-censored data.

**Running it.** The scripts take the Step 3 output directory, an output folder name and
the assembled table:

```bash
python LTDIndices_stats.py  /path/to/RESULTS-GBM_4-cohorts_Tissues/ \
                            LesionTract-density_Tissue-types \
                            data-clinical_TD-tissues_4-cohorts.csv \
                            --format pdf --cohort -1
```

`--cohort` selects the cohort to analyse (`-1` for all; see `--help` for the mapping), and
`--format` chooses `pdf` or `svg` figures. `TDIndices_stats.py` and `Volumes_stats.py`
take the same arguments. The notebooks run in the order of the table above.

Before running Step 4, check which survival column
[Step 3 recommended](#choosing-raw-or-corrected-survival) and point the analyses at it.

---

## Repository layout

```
.
├── <COHORT>/                       # UCSF-PDGM, UPENN-GBM, TCGA-GBM, RHUH-GBM, LUMIERE-GBM
│   ├── normalize_MNI.sh            #   Step 1: registration to MNI space
│   ├── quality-control_registration-MNI.py
│   ├── get-modality_space-MNI_*.py #           cohort-specific modality/subtype sorting
│   ├── TDMaps.sh                   #   Step 2: tract-density mapping
│   ├── TDMaps-extraction.py        #           TDI / L-TDI extraction
│   └── morphology-extraction.py    #           compartment volumes
│
├── createDatabase.py               #   Step 3: pooled clinical + imaging table, report
│
├── TDIndices_stats.py              #   Step 4: statistics and modelling
├── LTDIndices_stats.py
├── Volumes_stats.py
├── LTDI-Volume_comparison.ipynb
├── multivariate_survival.ipynb
├── anatomical_TractDensityMarkers.ipynb
│
├── segmentation_TCGA_example.ipynb # optional: BraTS segmentation of unlabelled scans
├── utils/
│   ├── metrics.py                  # survival metrics, quantile OS, concordance
│   └── statistics.py               # BB procedure, bootstrap/permutation tests, DeLong
└── requirements.txt
```

The tree is representative rather than exhaustive. Cohort folders are intentionally
near-duplicates: the I/O differs between datasets while the method does not. Names vary
where a dataset demands it — RHUH-GBM uses `normalizeT0_MNI.sh`, and LUMIERE-GBM adds
`EOR.sh` / `compute_eor.py` to derive extent of resection from its longitudinal scans. A
few cohorts also carry their own `demographics__*.ipynb` and `multivariate_survival.ipynb`
for cohort-level checks.

---

## Citation

If you use this code, consider the two methodological papers.

**The L-TDI — definition, validation and survival stratification:**

> Falcó-Roget, J., Basile, G. A., Janus, A., Lillo, S., Politi, L. S., Argasinski, J. K.,
> & Cacciola, A. (2026). A non-local diffusion magnetic resonance imaging tract density
> biomarker to stratify, predict, and interpret survival rates in human glioblastoma.
> *Neuro-Oncology*, 28(2), 564–579. https://doi.org/10.1093/neuonc/noaf234

**Volumetric comparison and the Benjamini–Bogomolov statistical framework:**

> Falcó-Roget, J., Ielo, A., Janus, A., Lillo, S., Fasano, A., Matteoli, M., Pessina, F.,
> Politi, L. S., Coletta, L., Vavassori, L., Sarubbo, S., Avesani, P., Argasinski, J. K.,
> & Cacciola, A. (2026). Beyond tumor volume: connectomic measures of tumor burden provide
> superior prognostic information across tumor compartments in glioblastoma. *medRxiv*.
> https://doi.org/10.64898/2026.09.14.26362015

<details>
<summary>BibTeX</summary>

```bibtex
@article{FalcoRoget2026LTDI,
  author  = {Falc\'{o}-Roget, Joan and Basile, Gianpaolo Antonio and Janus, Anna and
             Lillo, Sara and Politi, Letterio S. and Argasinski, Jan K. and Cacciola, Alberto},
  title   = {A non-local diffusion magnetic resonance imaging tract density biomarker to
             stratify, predict, and interpret survival rates in human glioblastoma},
  journal = {Neuro-Oncology},
  volume  = {28},
  number  = {2},
  pages   = {564--579},
  year    = {2026},
  doi     = {10.1093/neuonc/noaf234}
}

@article{FalcoRoget2026Volume,
  author  = {Falc\'{o}-Roget, Joan and Ielo, Augusto and Janus, Anna and Lillo, Sara and
             Fasano, Alfonso and Matteoli, Michela and Pessina, Federico and
             Politi, Letterio S. and Coletta, Ludovico and Vavassori, Laura and
             Sarubbo, Silvio and Avesani, Paolo and Argasinski, Jan K. and Cacciola, Alberto},
  title   = {Beyond tumor volume: connectomic measures of tumor burden provide superior
             prognostic information across tumor compartments in glioblastoma},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.14.26362015}
}
```
</details>

---

## Acknowledgements

This work would not exist without the groups who made the following resources public. If
you reuse this pipeline, please cite the datasets and atlases you actually use, in addition
to the papers above.

### Patient cohorts

**UCSF-PDGM** — The University of California San Francisco Preoperative Diffuse Glioma MRI dataset. [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)

> Calabrese, E., Villanueva-Meyer, J. E., Rudie, J. D., Rauschecker, A. M., Baid, U.,
> Bakas, S., Cha, S., Mongan, J. T., & Hess, C. P. (2022). The University of California
> San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial
> Intelligence*, 4(6), e220058. https://doi.org/10.1148/ryai.220058

**UPENN-GBM** — Multi-parametric MRI scans for de novo glioblastoma from the University of Pennsylvania Health System. [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)

> Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D.,
> Flores Santamaría, N., Fathi Kazerooni, A., Pati, S., *et al.*, & Davatzikos, C. (2022).
> The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical,
> genomics, & radiomics. *Scientific Data*, 9(1), 453.
> https://doi.org/10.1038/s41597-022-01560-7

**TCGA-GBM** — Segmentation labels and radiomic features for the pre-operative scans of the
TCGA-GBM collection. [Analysis result](https://www.cancerimagingarchive.net/analysis-result/brats-tcga-gbm/)
([10.7937/K9/TCIA.2017.KLXWJJ1Q](https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q)), derived
from the TCGA-GBM collection ([10.7937/K9/TCIA.2016.RNYFUYE9](https://doi.org/10.7937/K9/TCIA.2016.RNYFUYE9)).

> Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J. S.,
> Freymann, J. B., Farahani, K., & Davatzikos, C. (2017). Advancing The Cancer Genome
> Atlas glioma MRI collections with expert segmentation labels and radiomic features.
> *Scientific Data*, 4, 170117. https://doi.org/10.1038/sdata.2017.117
>
> Scarpace, L., Mikkelsen, T., Cha, S., Rao, S., Tekchandani, S., Gutman, D., Saltz, J. H.,
> Erickson, B. J., Pedano, N., Flanders, A. E., Barnholtz-Sloan, J., Ostrom, Q.,
> Barboriak, D., & Pierce, L. J. (2016). The Cancer Genome Atlas Glioblastoma Multiforme
> Collection (TCGA-GBM) [Data set]. *The Cancer Imaging Archive*.
> https://doi.org/10.7937/K9/TCIA.2016.RNYFUYE9

**RHUH-GBM** — The Río Hortega University Hospital Glioblastoma dataset. [10.7937/4545-c905](https://doi.org/10.7937/4545-c905)

> Cepeda, S., García-García, S., Arrese, I., Herrero, F., Escudero, T., Zamora, T., &
> Sarabia, R. (2023). The Río Hortega University Hospital Glioblastoma dataset: A
> comprehensive collection of preoperative, early postoperative and recurrence MRI scans
> (RHUH-GBM). *Data in Brief*, 50, 109617. https://doi.org/10.1016/j.dib.2023.109617

**LUMIERE** — Longitudinal glioblastoma MRI with expert RANO evaluation. [figshare collection](https://doi.org/10.6084/m9.figshare.c.5904905)

> Suter, Y., Knecht, U., Valenzuela, W., Notter, M., Hewer, E., Schucht, P., Wiest, R., &
> Reyes, M. (2022). The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO
> evaluation. *Scientific Data*, 9(1), 768. https://doi.org/10.1038/s41597-022-01881-7

### Normative tractogram

The L-TDI is computed against a normative whole-brain connectome derived from 985 healthy
Human Connectome Project subjects (~12 million streamlines), distributed in MNI space.
[10.6084/m9.figshare.c.6844890](https://doi.org/10.6084/m9.figshare.c.6844890)

> Elias, G. J. B., Germann, J., Joel, S. E., Li, N., Horn, A., Boutet, A., &
> Lozano, A. M. (2024). A large normative connectome for exploring the tractographic
> correlates of focal brain interventions. *Scientific Data*, 11(1), 353.
> https://doi.org/10.1038/s41597-024-03197-0

### Anatomical atlases

Used by `anatomical_TractDensityMarkers.ipynb` to localise the markers to white matter
tracts (XTRACT) and cortical lobes (USCLobes / BCI-DNI).

> Warrington, S., Bryant, K. L., Khrapitchev, A. A., Sallet, J., Charquero-Ballester, M.,
> Douaud, G., Jbabdi, S., Mars, R. B., & Sotiropoulos, S. N. (2020). XTRACT —
> Standardised protocols for automated tractography in the human and macaque brain.
> *NeuroImage*, 217, 116923. https://doi.org/10.1016/j.neuroimage.2020.116923
>
> Joshi, A. A., Choi, S., Liu, Y., Chong, M., Sonkar, G., Gonzalez-Martinez, J., Nair, D.,
> Wisnowski, J. L., Haldar, J. P., Shattuck, D. W., Damasio, H., & Leahy, R. M. (2022). A
> hybrid high-resolution anatomical MRI atlas with sub-parcellation of cortical gyri using
> resting fMRI. *Journal of Neuroscience Methods*, 374, 109566.
> https://doi.org/10.1016/j.jneumeth.2022.109566


## Contact

Questions, problems and reuse: open an issue, contact
[Joan Falcó-Roget](https://github.com/JoanSano), or
send an email joan.falcoroget@gmail.com.
