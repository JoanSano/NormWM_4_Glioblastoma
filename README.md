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

- [What the pipeline does](#what-the-pipeline-does)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [The site correction, in detail](#the-site-correction-in-detail)
- [Data](#data)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## What the pipeline does

The analysis runs in four stages. Stages 1–2 are per-cohort shell/Python pipelines; stages
3–4 are cohort-agnostic and operate on the pooled tables produced upstream.

**1. Spatial normalisation** — `<COHORT>/normalize_MNI.sh`

Each patient's structural MRI is registered to the MNI ICBM 2009b NLIN ASYM template with
ANTs (`antsRegistrationSyNQuick.sh`), using a cost-function mask that excludes the lesion
so the tumour does not bias the warp. Tumour segmentations are carried into template space
with the resulting transforms. `quality-control_registration-MNI.py` flags failed
registrations for visual review.

**2. Tract-density mapping** — `<COHORT>/TDMaps.sh` → `TDMaps-extraction.py`

Each segmentation is split into tumour compartments (whole, core, non-enhancing,
enhancing, core+enhancing). For every compartment, MRtrix3 (`tckedit`) selects the
normative streamlines traversing it and `tckmap` renders them as a **lesion tract-density
map**. Two indices are then extracted per compartment:

| Index | Definition | Interpretation |
|---|---|---|
| **TDI** | Mean normative streamline density *within* the compartment mask | Local white matter density of the tissue the tumour occupies |
| **L-TDI** | Mean density of the map formed by the streamlines the compartment *intercepts* | Whole-brain disruption footprint — the non-local marker |

Both are thresholded at a minimum streamline count per voxel (`-s`, default 0). 

`morphology-extraction.py` records compartment volumes in
parallel, so volume and L-TDI can be compared head to head.

**3. Database assembly** — `createDatabase.py`

Harmonises the per-cohort outputs with clinical and molecular variables (age, sex, KPS,
extent of resection, MGMT, IDH, overall survival and censoring) into the pooled
`data-clinical_*` tables that every downstream analysis consumes. Each cohort reports
these variables under its own schema and coding, so the script applies the per-cohort
inclusion criteria (IDH-wildtype, grade IV, treatment-naive, known survival), converts
survival to days, and recodes sex, MGMT and extent of resection onto common integer keys
(written alongside the table as `keys-maps.json`).

The cohorts are also split into two `site` groups, because survival in UCSF-PDGM is
recorded from a different reference point than in the remaining cohorts. The script
estimates that difference as a Cox log hazard ratio, adds an `OS (days) - corrected`
column rescaled by it, runs a battery of diagnostics on whether the difference is really
about *where patients entered* rather than *who they were*, and closes with an explicit
recommendation of which of the two survival columns to analyse. Every figure and table it
produces is collected into one self-contained HTML report next to the table; the terminal
stays quiet and the line-by-line record goes to a log file beside it.

**This is the least self-explanatory part of the repository, and the part most likely to
change a published number.** It has its own section below:
[The site correction, in detail](#the-site-correction-in-detail).

**4. Statistics and modelling**

| Entry point | Question it answers |
|---|---|
| `TDIndices_stats.py`, `LTDIndices_stats.py`, `Volumes_stats.py` | Per-compartment univariate survival statistics for TDI, L-TDI and volume |
| `LTDI-Volume_comparison.ipynb` | Is L-TDI or tumour volume the better prognostic marker, pre- and post-surgery? |
| `multivariate_survival.ipynb` | Cox models with clinical covariates; equivalence of L-TDI to TDI + volume |
| `anatomical_TractDensityMarkers.ipynb` | Anatomical localisation of the markers; quantifying site effects across cohorts |

Statistical inference is deliberately conservative and lives in `utils/statistics.py`:
selective inference across correlated compartment families via the **Benjamini–Bogomolov
(BB) procedure**, bootstrap confidence intervals and permutation tests for concordance
indices, bootstrapped median-survival differences, DeLong tests for AUC comparison, and
IPCW-corrected concordance for right-censored data.

---

## Repository layout

```
.
├── <COHORT>/                       # UCSF-PDGM, UPENN-GBM, TCGA-GBM, RHUH-GBM, LUMIERE-GBM
│   ├── normalize_MNI.sh            #   1. registration to MNI space
│   ├── quality-control_registration-MNI.py
│   ├── get-modality_space-MNI_*.py #      cohort-specific modality/subtype sorting
│   ├── TDMaps.sh                   #   2. tract-density mapping
│   ├── TDMaps-extraction.py        #      TDI / L-TDI extraction
│   └── morphology-extraction.py    #      compartment volumes
│
├── createDatabase.py               #   3. pooled clinical + imaging tables
│
├── TDIndices_stats.py              #   4. statistics and modelling
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

The tree above is representative rather than exhaustive: cohort folders are intentionally
near-duplicates, because each dataset ships a different directory structure, file naming
and metadata schema, so the I/O differs while the method does not. Names vary slightly
where a dataset demands it — RHUH-GBM registers its baseline timepoint with
`normalizeT0_MNI.sh`, and LUMIERE-GBM adds `EOR.sh` / `compute_eor.py` to derive extent of
resection from its longitudinal scans. A few cohorts also carry their own
`demographics__*.ipynb` and `multivariate_survival.ipynb` for cohort-level checks.

## Installation

**Python.** Python 3.12 is recommended. Everything here was developed and tested inside a
conda environment, which is also the easiest way to get the version floors in
`requirements.txt` satisfied:

```bash
git clone https://github.com/JoanSano/NormWM_4_Glioblastoma.git
cd NormWM_4_Glioblastoma
conda create -n normwm python=3.12
conda activate normwm
pip install -r requirements.txt
```

Optional extras (tumour segmentation, streamline conversion, consensus community
detection) are listed, commented out, at the bottom of `requirements.txt`.

**External neuroimaging software.** The shell pipelines call the following directly; they
are not pip-installable and must be on your `PATH`:

| Software | Used for | Commands |
|---|---|---|
| [ANTs](https://github.com/ANTsX/ANTs) | Registration to MNI, bias correction | `antsRegistrationSyNQuick.sh`, `antsApplyTransforms`, `N4BiasFieldCorrection`, `ImageMath` |
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) | Mask arithmetic, compartment extraction | `fslmaths` |
| [MRtrix3](https://www.mrtrix.org/) | Streamline selection and tract-density mapping | `tckedit`, `tckmap` |

**Template and tractogram.** You will also need, in MNI ICBM 2009b NLIN ASYM space: the
template itself and the normative tractogram (`dTOR_full_tractogram.tck` and its TD map).
Their locations are set by the `MNI_TEMPLATE` variable at the top of each shell script. See
[Acknowledgements](#acknowledgements) for where to obtain them.

## Usage

Per-cohort processing, from a cohort directory. Edit the `MAIN_DIR` / `BASE_DIR` variables
at the top of each script first, then:

```bash
cd UCSF-PDGM

# 1. Register to MNI space and QC the warps
./normalize_MNI.sh
python quality-control_registration-MNI.py

# 2. Tract-density maps and indices
#    -g grade  -n parallel subjects  -s min. streamline density  -k keep .tck files
./TDMaps.sh -g IV -n 4 -s 0 -k 0
```

Database assembly, from the repository root. It takes the directory holding the cohort
folders and the output directory (relative paths are resolved under the former):

```bash
python createDatabase.py /path/to/Glioblastomas RESULTS-GBM_5-cohorts_Tissues \
                         --idh WT --grade IV --stream-th 0 --format pdf
```

To condition the site effect on case-mix — **recommended**, and the basis of the
recommendation the run ends with:

```bash
python createDatabase.py /path/to/Glioblastomas RESULTS-GBM_5-cohorts_Tissues \
                         --adjust-covariates age sex eor mgmt
```

The diagnostics are not optional and there is no flag to skip them: every run computes
them, writes them as CSVs and into the report, and ends by recommending which survival
column to analyse.
`--stream-th` must match the `-s` used by `TDMaps.sh`. See `--help` for the rest and
[The site correction, in detail](#the-site-correction-in-detail) for how to read the
output.

| Flag | What it does |
|---|---|
| `--idh`, `--grade`, `--stream-th` | Which per-cohort pipeline outputs to read: IDH status, WHO grade (UCSF-PDGM only) and the minimum streamline density the indices were extracted at. `--stream-th` must match the `-s` given to `TDMaps.sh`. |
| `--cohorts` | Restricts the pool to a subset (default: all five). A selection that is not the first N cohorts by ID is named after its cohorts, so two subsets cannot overwrite each other. |
| `--adjust-covariates` | Covariates the site model conditions on: any of `age sex eor mgmt kps`. Default: none, i.e. the crude site effect. |
| `--site-reference` | Cohorts forming the reference group (site 0). Default: UCSF alone. |
| `--ladder-covariates` | Covariates the diagnostics ladder walks (default: `--adjust-covariates` if given, else `age sex eor mgmt`). |
| `--truncate-months` | Horizons for the administrative-truncation check (default: 12 24 36 48). |
| `--pairwise` | Also inspect every *pair* of cohorts, not only the two site groups. Slow: it refits the permutation test per pair. |
| `--output-name` | Base name of the assembled table and everything named after it (default: `data-clinical_TD-tissues_<N>-cohorts`). |
| `--log` | Rename the run's log file (default `createDatabase_log.txt`). The run is always logged; this only changes where. |
| `--verbose` | Also print the run to the terminal. Off by default — see [Where the output goes](#where-the-output-goes). |
| `--n-perms`, `--seed` | Permutations for the concordance test, and the seed for them. |
| `--format`, `--show` | Figure format (`pdf`, `svg`, `both`), and whether to display them. |

#### Where the output goes

**The terminal stays quiet.** A run prints two lines when it starts and a short
summary when it finishes — the paths it wrote and the recommended survival column —
and nothing in between:

```
createDatabase.py: pooling UCSF, UPENN, TCGA, RHUH -> /path/RESULTS-GBM_4-cohorts_Tissues
  running quietly; the run is written to .../createDatabase_log.txt (--verbose to watch it here)

Assembled 999 subjects from 4 cohorts.
  table   .../data-clinical_TD-tissues_4-cohorts.csv
  report  .../data-clinical_TD-tissues_4-cohorts_report.html
  log     .../createDatabase_log.txt
  verdict RAW -- analyse 'OS (days)'
```

Everything else — every sample size, fit, test and warning — goes to the log file,
which is always written. `--verbose` mirrors it to the terminal as well. Progress
bars are unaffected either way: they go to stderr, so a long run still shows that it
is alive.

Every run writes, next to the assembled table:

| File | Contents |
|---|---|
| `<stem>.csv` / `.tsv` | The pooled table, including `OS (days)`, `OS (days) - corrected` and `site correction factor` (which makes the rescaling invertible per subject). |
| `<stem>_report.html` | **Start here.** Every figure, every table and the recommendation in one self-contained file. Open it in a browser; print it to PDF from there. |
| `createDatabase_log.txt` | The full run, line by line: sample sizes, censoring, every fit and every warning. The report points at it rather than embedding it. |
| `<stem>_site-correction.json` | Machine-readable provenance: the coefficient, its CI, the design, the resolved site partition and the recommended column. |
| `keys-maps.json` | The categorical encodings used in the table. |
| `OS-stats/` | The figures as `.pdf`/`.svg`, and the diagnostic tables as `Site-diagnostics_*.csv`. |

Pooled statistics, from the repository root. Each script takes a results directory, an
output folder name and the assembled table:

```bash
python LTDIndices_stats.py  /path/to/RESULTS-GBM_4-cohorts_Tissues/ \
                            LesionTract-density_Tissue-types \
                            data-clinical_TD-tissues_4-cohorts.csv \
                            --format pdf --cohort -1
```

`--cohort` selects the cohort to analyse (`-1` for all; see `--help` for the mapping), and
`--format` chooses `pdf` or `svg` figures. `TDIndices_stats.py` and `Volumes_stats.py`
take the same arguments. Remaining analyses run as notebooks, in the order given in
[What the pipeline does](#what-the-pipeline-does).

## The site correction, in detail

Pooling five cohorts creates a problem that has nothing to do with biology: **UCSF-PDGM
records survival from a different reference point than the other four.** Left alone, that
shows up as a survival difference between cohorts and contaminates every downstream
model. Removing it is not optional. Removing *too much* is the risk, and that is what this
section is about.

### The two things `site` could be

A survival gap between two groups of cohorts can come from either of two sources, and they
call for opposite responses:

| Source | What it is | What to do |
|---|---|---|
| **Entry point** | The clock starts at a different event in one cohort. An artefact of record-keeping. | Remove it — rescale the times, or stratify the baseline hazard. |
| **Case-mix** | The cohorts really do hold different patients: more methylated MGMT, more gross-total resections, older patients. A genuine prognostic difference. | **Keep it.** Adjust for the covariates in the downstream model instead. |

Removing case-mix by rescaling the outcome destroys signal you are trying to measure, and
if the downstream model *also* adjusts for those covariates, the same effect is removed
twice. The whole apparatus below exists to tell the two sources apart before deciding.

### What the correction actually does

The script fits a Cox model for the 0/1 `site` indicator and writes

```
OS (days) - corrected = OS (days) × exp(logHR × site)
```

so group 0 (the reference) is untouched and group 1's times are rescaled by a single
constant. `site correction factor` stores that per-subject multiplier, so the operation is
invertible. `--site-reference` chooses which cohorts form group 0; the default is UCSF
alone, but nothing in the method requires that, and naming *every* selected cohort leaves
one group and applies no correction at all.

With `--adjust-covariates`, the coefficient comes from a model that also holds age, sex,
EOR, MGMT and/or KPS fixed. That model is fitted on the subjects reporting every chosen
covariate, and the resulting coefficient is applied to **all** subjects — so the assembled
table never shrinks. Missingness here is severe and cohort-structured (EOR is unrecorded
for all of TCGA, MGMT for all of RHUH, KPS for all of UCSF and LUMIERE), and a
complete-case *table* would cost most of the sample. The transfer assumes the site effect
is the same in complete and incomplete cases; the balance table's missingness columns are
the evidence you weigh that against.

### Reading the diagnostics

Every run produces these, in the HTML report and as CSVs under `OS-stats/`:

| Table | The question it answers |
|---|---|
| **Balance and missingness** | How different are the two groups to begin with? An absolute SMD above 0.10 marks an imbalance worth adjusting for. The two `pct_missing` columns sit next to it because a covariate one group never records cannot be balanced by any adjustment. |
| **Adjustment ladder** | How much of the site effect is case-mix? Every rung is fitted on *one fixed complete-case sample*, so rows differ only in what is adjusted for, never in who is in the model. Watch `logHR` shrink as covariates enter, and read `pct_of_crude_removed`. |
| **Follow-up (reverse KM)** | Were the groups watched for equally long? Median *potential* follow-up, not median observed survival. A log-rank on the censoring distribution is reported beside it. |
| **Administrative truncation** | Is the effect an artefact of unequal follow-up? Everyone is censored at a common horizon, which makes the groups equally observed by construction. An estimate that barely moves across horizons is not a follow-up artefact. |
| **Cohort-stratified Cox** | The alternative to rescaling: leave the baseline hazard free per cohort instead of touching the outcome. Its rows are **mutually adjusted** covariate effects on one complete-case sample — not a univariate effect per covariate. The raw and corrected rows are identical by construction, which is what makes the two approaches alternatives rather than things to do together. |

The forest plot of the ladder is the single most useful picture: if the site log-HR walks
towards zero as covariates enter, the gap was case-mix.

Hazard ratios are reported **per native unit** — one year of age, one KPS point — rather
than rescaled to per-10 units, so a coefficient can be read straight against the column it
came from. That puts the informative digits in the 2nd–3rd decimal, which is why the
stratified table prints six: age reads `1.029380`, not `1.03`. Precision is set per column
rather than per table, so counts stay integers and p-values keep their own notation
(`<0.001`) instead of rounding to `0.000000`.

### Reading the figures

Three kinds of figure are produced, in `OS-stats/` and embedded in the report.

**Cohort survival, before and after correction.** One Kaplan-Meier curve per cohort, with
the omnibus log-rank annotated and pairwise log-rank tests in the log. Beneath each is a

```
No. at risk (right-censored)
 367 (0)    96 (102)   21 (130)    2 (143)    0 (144)    0 (144)
 496 (0)   149 (2)     49 (2)     25 (3)     13 (6)      3 (11)
```

row: the number still under observation at that month, and in brackets the *cumulative*
number right-censored before it. The convention matches the one used by the
`Tract-Density_Components-Survival` repository, so tables from the two read alike. Columns
are thinned to whatever fits the axis at a legible size, and each printed column is centred
on its tick.

**The adjustment ladder forest plot.** The site log-HR with its 95% CI, one row per rung,
labelled with the sample each was fitted on.

**The site-effect figure** (`Site-effects_Survival-times_*`) has three panels, and they do
not all describe the same model:

| Panel | What it shows |
|---|---|
| Top left | Complementary log-log curves per group, with an **unadjusted** log-rank. A Kaplan-Meier curve has no covariates to hold fixed, so this panel is marginal by construction and its annotation says so. |
| Bottom left | Scaled Schoenfeld residuals and the Grambsch-Therneau test **for the model whose coefficient is actually applied** — adjusted when the correction is adjusted. Proportional hazards is a property of a model, not of a pair of groups, so diagnosing the crude fit while applying an adjusted coefficient would vouch for the wrong model. The axis label names the covariates and the complete-case sample the panel rests on. |
| Right | Survival after rescaling group 1's times, with the uncorrected curve overlaid in grey. The legend states the log HR applied and whether it is crude or adjusted. |

`--pairwise` produces the same figure for every *pair* of cohorts. Each pair gets its own
adjusted coefficient, estimated on that pair alone — a coefficient borrowed from the site
model would describe a different contrast — and its diagnostic panel is fitted on that same
model.

### The recommendation

Each run ends with an explicit verdict — printed, in the report, and in the provenance
JSON as `recommended_outcome`. The rule it applies is:

> Fit the site effect adjusted for case-mix. **If its 95% confidence interval covers
> zero, use the raw survival times** and adjust or stratify for cohort downstream. If it
> excludes zero, the corrected column is defensible.

The verdict is computed from the adjustment ladder, which runs on every invocation
regardless of what was applied — so **the recommendation can disagree with the column the
table actually carries**, and it says so when it does. The corrected column is always
written; nothing downstream is obliged to use it.

It also reports what qualifies the verdict: a rejected proportional-hazards test (a single
multiplicative factor is then the wrong description at every follow-up time, and
stratification is safer), the size of the complete-case sample the estimate rests on, the
largest remaining imbalance, and whether the censoring distributions differ.

### What the pooled four-cohort run concludes

For the canonical UCSF + UPENN + TCGA + RHUH pool, adjusted for age, sex, EOR and MGMT:

| | log HR | 95% CI | p | n |
|---|---|---|---|---|
| Crude site effect | 0.2922 | 0.1374 to 0.4471 | <0.001 | 999 |
| Adjusted for case-mix | 0.1381 | −0.0594 to 0.3356 | 0.171 | 617 |

Case-mix accounts for roughly 55% of the site effect, and what remains is not
distinguishable from zero. **The verdict for this pool is RAW**: analyse `OS (days)` and
adjust or stratify for cohort in the downstream model. The gap between UCSF-PDGM and the
rest is mostly its MGMT and EOR distribution, not its entry point, and rescaling the
outcome by it would remove prognostic signal that belongs to those covariates.

Re-run the assembly for any other cohort selection: the verdict is a property of the pool,
not of the method, and a different subset can land the other way.

### Practical rules

- **Use one remedy, not both.** Rescaling the outcome and stratifying the baseline hazard
  by cohort correct the same difference; applying both removes it twice.
- **Read the report before the table.** `<stem>_report.html` holds the figures, the
  diagnostics and the verdict in the order the analysis ran. For the line-by-line
  record of what was fitted, read `createDatabase_log.txt` beside it — the report
  names the file rather than embedding it.
- **Quote the provenance.** `<stem>_site-correction.json` records the coefficient, its CI,
  the design, the resolved site partition and the recommended column — everything needed
  to state in a methods section what was done to the survival times.
- **The correction is invertible.** Divide by `site correction factor` to recover the raw
  times from a corrected table.

---

## Data

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
