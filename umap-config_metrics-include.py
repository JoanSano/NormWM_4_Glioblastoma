import pandas as pd
from joblib import Parallel, delayed 
import os 
import warnings
import numpy as np
from itertools import combinations as iter_combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_TRT_LOGGER"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"


MAIN_DIR = "/home/joan/Desktop/PROJECTS/Glioblastomas"
RESULTS = f"{MAIN_DIR}/RESULTS-GBM_4-cohorts_UMAP-Risks"
os.makedirs(f"{RESULTS}", exist_ok=True)

daysXmonth = 365/12
voxel_size = (0.5**3) * (1/1000)


hcpex_complete = pd.read_csv(f"{RESULTS}/data-clinical_parcellation-hcpex_4-cohorts.csv", sep=",")
aal3_complete  = pd.read_csv(f"{RESULTS}/data-clinical_parcellation-aal3_4-cohorts.csv",  sep=",")


metrics_covariates = [
    'size', 'density', 'avg_degree',
    'avg_clustering', 'modularity', 'global_efficiency', 'avg_local_efficiency', 'avg_shortest_path_length',
    'avg_participation_coef', 'avg_eigenvector_centrality', 'avg_betweenness_centrality', 'avg_closeness_centrality', 'avg_edge_betwenness_centrality', 'avg_percolation_centrality',
    'deg_assortativity', 'powerlaw_exponent', 'LLR_distribution', 's_metric_SF', 'avg_rich_club_coef'
]

clinical_covariates = [
    'age', 'sex', 'eor', 'mgmt', 'kps (preop)'
]

tract_covariates = [
    'Whole TDMap', 'Whole lesion TDMap'
]

morphology_covariates = [
    "Whole tumor size (voxels)"
]

survival_covariates = [
    'cohort', 'site',
    'OS (days) - corrected', 'status'
]


hcpex_metrics = hcpex_complete[
    survival_covariates + metrics_covariates + tract_covariates + morphology_covariates
].copy().dropna()
print(f"HCPEx -- N={len(hcpex_metrics)} ({len(hcpex_complete)-len(hcpex_metrics)} samples with 'NaN' entries discarded)")

aal3_metrics = aal3_complete[
    survival_covariates + metrics_covariates + tract_covariates + morphology_covariates
].copy().dropna()
print(f"AAL3  -- N={len(aal3_metrics)} ({len(aal3_complete)-len(aal3_metrics)} samples with 'NaN' entries discarded)")


from utils.UMAP_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from utils.metrics import clustering_metrics, ranking_OS_topological_risk, topological_risk_concordance_index, topological_risk_kendalltau
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, estimate_bandwidth, HDBSCAN
warnings.filterwarnings('ignore')


##############################
### Split into train and test
custom = True
if custom is True:
    hcpex_train = hcpex_metrics.loc[hcpex_metrics["cohort"]<=1].copy()
    hcpex_test = hcpex_metrics.loc[hcpex_metrics["cohort"]>1].copy()
else:
    from sklearn.model_selection import train_test_split
    hcpex_train, hcpex_test = train_test_split(
        hcpex_metrics,
        test_size=0.2,
        random_state=None,
        shuffle=True,
        stratify=hcpex_metrics["status"]
    )
    hcpex_train = hcpex_train.copy()
    hcpex_test  = hcpex_test.copy()

##############################
### These will be used to define the mapping between cluster labels and topological risk
mask_train = hcpex_train["status"]==1
os_train = hcpex_train["OS (days) - corrected"].values
status_train = hcpex_train["status"].values

##############################
### Standardizing the test sample based on the training distribution
st = 1
if st == 1:
    standard = StandardScaler(with_mean=True, with_std=True)
    hcpex_standard_train = standard.fit_transform(hcpex_train[metrics_covariates])
    hcpex_standard_test = standard.transform(hcpex_test[metrics_covariates])
    data_train, data_test = hcpex_standard_train.copy(), hcpex_standard_test.copy()
elif st == 2:
    minmax = MinMaxScaler(feature_range=(0,1), clip=False)
    hcpex_minmax_train = minmax.fit_transform(hcpex_train[metrics_covariates])
    hcpex_minmax_test = minmax.transform(hcpex_test[metrics_covariates])
    data_train, data_test = hcpex_minmax_train.copy(), hcpex_minmax_test.copy()
elif st == 3:
    robust = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False)
    hcpex_robust_train = robust.fit_transform(hcpex_train[metrics_covariates])
    hcpex_robust_test = robust.transform(hcpex_test[metrics_covariates])
    data_train, data_test = hcpex_robust_train.copy(), hcpex_robust_test.copy()
elif st == 4:
    powerT = PowerTransformer(method='yeo-johnson', standardize=True)
    hcpex_powerT_train = powerT.fit_transform(hcpex_train[metrics_covariates])
    hcpex_powerT_test = powerT.transform(hcpex_test[metrics_covariates])
    data_train, data_test = hcpex_powerT_train.copy(), hcpex_powerT_test.copy()

N_train, N_test = len(data_train), len(data_test)
print("Train:", N_train, "Test:", N_test)    

# Index lookup: metric name → column position in metrics_covariates
metric_index = {m: i for i, m in enumerate(metrics_covariates)}

##############################
### Fixed UMAP configuration
ranking_method = "median-os"
random_seed    = 42
fixed_umap_config = {
    "n_neighbors": 40,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": 57463,
    "min_dist": .1,
    "spread": 1
}

##############################
### Metric subsets to evaluate
# sys.argv[4] (or a hard-coded default) sets how many metrics per combination.
# Pass  -1  to sweep ALL sizes from 2 to len(metrics_covariates).
try:
    Nmetrics = int(sys.argv[4])
except (IndexError, ValueError):
    Nmetrics = len(metrics_covariates) 
combinations = list(iter_combinations(metrics_covariates, Nmetrics))

print(f"Fixed UMAP config           : {fixed_umap_config}")
print(f"Nmetrics                    : {'all sizes' if Nmetrics == -1 else Nmetrics}")
print(f"Number of metric subsets    : {len(combinations)}")

###################################
### Worker function
###################################

def run_metric_subset(
        k, 
        data_train,
        data_test, 
        metrics_subset, 
        metric_index,
        umap_configuration,
        hcpex_train, 
        hcpex_test,
        mask_train, 
        os_train, 
        status_train,
        ranking_method, 
        daysXmonth, 
        random_seed
    ):

    import warnings, sys, io, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    metrics_subset = list(metrics_subset)

    try:
        # ── Column-slice the pre-scaled arrays (no re-fitting needed) ────────
        cols = [metric_index[m] for m in metrics_subset]
        data_train = data_train[:, cols]
        data_test  = data_test[:, cols]

        # This meanshift code is only to reproduce the patent, for the paper the repetition is no longer needed
        risk_ = compute_topological_risk(
            fixed_umap_config,
            data_train,
            data_test,
            MeanShift(), 
            mask_train,
            os_train,
            status_train,
            method=ranking_method,
        )
        clusterizer = MeanShift(
            bandwidth=estimate_bandwidth(
                risk_["umap coordinates: train"], 
                quantile=0.15, 
                n_samples=data_train.shape[0]
            )
        )
        # ── UMAP + clustering ────────────────────────────────────────────────
        umap_risk = compute_topological_risk(
            umap_configuration,
            data_train,
            data_test,
            clusterizer=clusterizer,
            mask_train=mask_train,
            os_train=os_train,
            status_train=status_train,
            method=ranking_method,
            cmap="coolwarm"
        )

        hcpex_train_local = hcpex_train.copy()
        hcpex_test_local  = hcpex_test.copy()

        hcpex_train_local["topological risk"] = umap_risk["topological risk: train"]
        hcpex_test_local["topological risk"]  = umap_risk["topological risk: test"]

        fig, statistics = plot_umap_outcome(
            hcpex_train_local,
            umap_risk["umap coordinates: train"],
            hcpex_test_local,
            umap_risk["umap coordinates: test"],
            daysXmonth,
            umap_risk["color risk"],
            status="status",
            risk="topological risk",
            survival="OS (days) - corrected",
            cmap="coolwarm",
            months=range(0, 110, 10),
            show_median="all",
            plot=False
        )

        statistics = clustering_metrics(umap_risk, statistics)
        statistics = topological_risk_kendalltau(statistics)
        statistics = topological_risk_concordance_index(
            statistics,
            hcpex_train_local,
            hcpex_test_local,
            status="status",
            risk="topological risk",
            survival="OS (days) - corrected",
            alpha_CI=0.05,
            n_permutations=2500,
            n_bootstrap=None,
            random_state=random_seed
        )
        statistics = ranking_OS_topological_risk(
            statistics,
            hcpex_train_local,
            hcpex_test_local,
            status="status",
            risk="topological risk",
            survival="OS (days) - corrected",
        )

        train = statistics["training cohort(s)"]
        test  = statistics["testing cohort(s)"]

        # ── Shared metadata ──────────────────────────────────────────────────
        meta = {
            "combination ID":   k,
            "n_metrics":        len(metrics_subset),
            **umap_configuration,
        }
        dict_metrics = ", ".join(metrics_subset)

        train_result = {
            **meta,
            "km_chi_squared":           train["kaplan-meier"]["chi-squared"],
            "km_p_value":               train["kaplan-meier"]["p-value"],
            "cluster_silhouette":       train["clustering metrics"]["silhouette"],
            "cluster_calinski_harabasz":train["clustering metrics"]["calinski harabasz"],
            "cluster_davies_bouldin":   train["clustering metrics"]["davies bouldin"],
            "kendall_tau":              train["kendall tau"]["tau"],
            "kendall_p_val":            train["kendall tau"]["p_val"],
            "cox_c_index":              train["cox model"]["C-index"],
            "cox_c_index_p_perm":       train["cox model"]["C-index p-value (perm)"],
            "hazard_ratio":             train["cox model"]["Hazard ratio and p-value"][0],
            "hazard_ratio_pval":        train["cox model"]["Hazard ratio and p-value"][1],
            "spearman_rho":             train["rankings OS"]["spearman r"]["rho"],
            "spearman_p":               train["rankings OS"]["spearman r"]["p_val"],
            "os_kendall_tau":           train["rankings OS"]["kendall tau"]["tau"],
            "os_kendall_p":             train["rankings OS"]["kendall tau"]["p_val"],
            "somers_d":                 train["rankings OS"]["somers d"]["d"],
            "somers_d_p":               train["rankings OS"]["somers d"]["p_val"],
        }

        test_result = {
            **meta,
            "km_chi_squared":           test["kaplan-meier"]["chi-squared"],
            "km_p_value":               test["kaplan-meier"]["p-value"],
            "cluster_silhouette":       test["clustering metrics"]["silhouette"],
            "cluster_calinski_harabasz":test["clustering metrics"]["calinski harabasz"],
            "cluster_davies_bouldin":   test["clustering metrics"]["davies bouldin"],
            "kendall_tau":              test["kendall tau"]["tau"],
            "kendall_p_val":            test["kendall tau"]["p_val"],
            "cox_c_index":              test["cox model"]["C-index"],
            "cox_c_index_p_perm":       test["cox model"]["C-index p-value (perm)"],
            "spearman_rho":             test["rankings OS"]["spearman r"]["rho"],
            "spearman_p":               test["rankings OS"]["spearman r"]["p_val"],
            "os_kendall_tau":           test["rankings OS"]["kendall tau"]["tau"],
            "os_kendall_p":             test["rankings OS"]["kendall tau"]["p_val"],
            "somers_d":                 test["rankings OS"]["somers d"]["d"],
            "somers_d_p":               test["rankings OS"]["somers d"]["p_val"],
        }

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return k, train_result, test_result, None, None, dict_metrics

    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return k, None, None, "|".join(combinations), f"FAILED — {e}", dict_metrics


###################################
### Parallelised sweep
###################################

# CLI: python script.py <Ncs0> <Ncs1> <njobs> [<Nmetrics>]
Ncs0   = int(sys.argv[1])
Ncs1   = int(sys.argv[2])
Ncs1   = None if Ncs1==-1 else Ncs1
njobs  = int(sys.argv[3])

subset = combinations[Ncs0:Ncs1]
print(f"Running combinations from index {Ncs0} to {Ncs1} "
      f"(total: {len(subset)}) with {njobs} parallel jobs...")

raw_results = list(tqdm(
    Parallel(n_jobs=njobs, backend="loky", return_as="generator")(
        delayed(run_metric_subset)(
            k,
            data_train,
            data_test, 
            metrics_subset,
            metric_index,
            fixed_umap_config,
            hcpex_train,
            hcpex_test,
            mask_train,
            os_train,
            status_train,
            ranking_method,
            daysXmonth,
            random_seed
        )
        for k, metrics_subset in enumerate(subset, start=Ncs0)
    ),
    total=len(subset),
    desc="Metric subsets",
    unit="combo"
))
print(f"All combinations processed ({len(raw_results)}).")


###################################
### Collect & save results
###################################

results_train, results_test = {}, {}
key_metrics = {}
failed = []

for result in raw_results:
    if result is not None:
        k, train_result, test_result, name, status, dict_metrics = result
        if train_result is not None:
            results_train[k] = train_result
            results_test[k]  = test_result
            key_metrics[k]   = dict_metrics
        else:
            failed.append((k, name, status))

print(f"\nDone: {len(results_train)} succeeded, {len(failed)} failed")
if failed:
    for k, name, status in failed:
        print(f"  [{k}] {name} — {status}")

Nmetrics_tag = "all" if Nmetrics == -1 else str(Nmetrics)
os.makedirs(f"{RESULTS}/UMAP-stats", exist_ok=True)
out_dir = f"{RESULTS}/UMAP-stats/Nmetrics-{Nmetrics_tag}"
os.makedirs(out_dir, exist_ok=True)

df_test  = pd.DataFrame.from_dict(results_test,  orient="index")
df_train = pd.DataFrame.from_dict(results_train, orient="index")

Ncs1   = 'end' if Ncs1 is None else Ncs1
df_test.to_csv( f"{out_dir}/HCPEX__results-test_{Ncs0}-{Ncs1}.csv",  sep=",", index=False)
df_train.to_csv(f"{out_dir}/HCPEX__results-train_{Ncs0}-{Ncs1}.csv", sep=",", index=False)

import json
with open(f"{out_dir}/HCPEX__key-metrics_{Ncs0}-{Ncs1}.json", 'w', encoding='utf-8') as ff:
    json.dump(key_metrics, ff, ensure_ascii=False, indent=4)

print(f"Saved train results → {out_dir}/HCPEX__results-train_{Ncs0}-{Ncs1}.csv")
print(f"Saved test results → {out_dir}/HCPEX__results-test_{Ncs0}-{Ncs1}.csv")