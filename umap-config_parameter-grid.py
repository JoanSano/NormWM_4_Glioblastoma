import pandas as pd
#import matplotlib.pyplot as plt
from joblib import Parallel, delayed 
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from matplotlib.gridspec import GridSpec
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#import seaborn as sns
import os 
import warnings
#import itertools
import numpy as np
#from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, Birch#, estimate_bandwidth
from sklearn.mixture import GaussianMixture
#from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # suppress oneDNN warning
os.environ["TF_TRT_LOGGER"] = "3"                  # suppress TensorRT warning
os.environ["PYTHONWARNINGS"] = "ignore"



MAIN_DIR = "/home/joan/Desktop/PROJECTS/Glioblastomas"
RESULTS = f"{MAIN_DIR}/RESULTS-GBM_4-cohorts_UMAP-Risks"
os.makedirs(f"{RESULTS}", exist_ok=True)

daysXmonth = 365/12
voxel_size = (0.5**3) * (1/1000) # 0.5 (mm³/voxel) X 0.001 (cm³/mm³)   


hcpex_complete = pd.read_csv(f"{RESULTS}/data-clinical_parcellation-hcpex_4-cohorts.csv", sep=",")
aal3_complete = pd.read_csv(f"{RESULTS}/data-clinical_parcellation-aal3_4-cohorts.csv", sep=",")


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
    survival_covariates+metrics_covariates+tract_covariates+morphology_covariates
].copy().dropna()
print(f"HCPEx -- N={len(hcpex_metrics)} ({len(hcpex_complete)-len(hcpex_metrics)} samples with 'NaN' entries discarded)")

aal3_metrics = aal3_complete[
    survival_covariates+metrics_covariates+tract_covariates+morphology_covariates
].copy().dropna()
print(f"AAL3 -- N={len(aal3_metrics)} ({len(aal3_complete)-len(aal3_metrics)} samples with 'NaN' entries discarded)")

from utils.UMAP_utils import *
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)
from utils.metrics import clustering_metrics, ranking_OS_topological_risk, topological_risk_concordance_index, topological_risk_kendalltau
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, estimate_bandwidth, HDBSCAN
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

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


###################################------Umap parameters -------###################################
random_seed = 42

def set_umap():
    parameters = []      
    for n_n in range(10, 61):          
        for n_c in range(2, 5):        
            for m in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:  
                for s in [1, 1.5, 2, 3, 5, 10, 20]:
                    parameters.append({
                        "n_neighbors": n_n,
                        "n_components": n_c,
                        "metric": "euclidean",
                        "random_state": random_seed,
                        "min_dist": m,
                        "spread": s
                    })
    return parameters
umap_args = set_umap() 
ranking_method = "median-os"

###################################------Algorithms-------##########################################


def set_algorithms():
    algorithms = {}
    for k in range(2, 11):
        algorithms[f"KMeans_clusters-{k}"] = KMeans(n_clusters=k, random_state=42)
        algorithms[f"GMM_components-{k}"]    = GaussianMixture(n_components=k, random_state=42)
    for k in range(2, 11):
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            algorithms[f"Birch_clusters-{k}_threshold-{t}"] = Birch(n_clusters=k, threshold=t)
    algorithms[f"MeanShift"] = MeanShift(bandwidth=None)
    return algorithms


algorithms = set_algorithms()

###################################-----create combinations-------###################################


def set_combinations():
    combinations = []
    for algorithm_name, algorithm_variant in algorithms.items(): 
        for umap_configuration in umap_args: 
            combinations.append({
                "algorithm_name": algorithm_name, 
                "algorithm_variant": algorithm_variant, 
                "umap_configuration": umap_configuration,
            })
    
    return combinations 

combinations = set_combinations() 
print(f"Total number of combinations: {len(combinations)} LOL XD")



def run_combination(k, combination, data_train, data_test, mask_train, os_train,
                    status_train, ranking_method, hcpex_train, hcpex_test, daysXmonth,
                    random_seed):
    import warnings, sys, io, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    algorithm_name     = combination["algorithm_name"]
    algorithm_variant  = combination["algorithm_variant"]
    umap_configuration = combination["umap_configuration"]


    try:
        

        umap_risk = compute_topological_risk(
            umap_configuration,
            data_train,
            data_test,
            clusterizer=algorithm_variant,
            mask_train=mask_train,
            os_train=os_train,
            status_train=status_train,
            method=ranking_method,
            cmap="coolwarm"
        )

        # Work on local copies to avoid race conditions across parallel workers
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

        train_result = {
            "configuration ID": k,
            "algorithm name": algorithm_name,
            **umap_configuration,
            "km_chi_squared": train["kaplan-meier"]["chi-squared"],
            "km_p_value": train["kaplan-meier"]["p-value"],
            "cluster_silhouette": train["clustering metrics"]["silhouette"],
            "cluster_calinski_harabasz": train["clustering metrics"]["calinski harabasz"],
            "cluster_davies_bouldin": train["clustering metrics"]["davies bouldin"],
            "kendall_tau": train["kendall tau"]["tau"],
            "kendall_p_val": train["kendall tau"]["p_val"],
            "cox_c_index": train["cox model"]["C-index"],
            "cox_c_index_p_perm": train["cox model"]["C-index p-value (perm)"],
            "hazard_ratio": train["cox model"]["Hazard ratio and p-value"][0],
            "hazard_ratio_pval": train["cox model"]["Hazard ratio and p-value"][1],
            "spearman_rho": train["rankings OS"]["spearman r"]["rho"],
            "spearman_p": train["rankings OS"]["spearman r"]["p_val"],
            "os_kendall_tau": train["rankings OS"]["kendall tau"]["tau"],
            "os_kendall_p": train["rankings OS"]["kendall tau"]["p_val"],
            "somers_d": train["rankings OS"]["somers d"]["d"],
            "somers_d_p": train["rankings OS"]["somers d"]["p_val"],
        }

        test_result = {
            "configuration ID": k,
            "algorithm name": algorithm_name,
            **umap_configuration,
            "km_chi_squared": test["kaplan-meier"]["chi-squared"],
            "km_p_value": test["kaplan-meier"]["p-value"],
            "cluster_silhouette": test["clustering metrics"]["silhouette"],
            "cluster_calinski_harabasz": test["clustering metrics"]["calinski harabasz"],
            "cluster_davies_bouldin": test["clustering metrics"]["davies bouldin"],
            "kendall_tau": test["kendall tau"]["tau"],
            "kendall_p_val": test["kendall tau"]["p_val"],
            "cox_c_index": test["cox model"]["C-index"],
            "cox_c_index_p_perm": test["cox model"]["C-index p-value (perm)"],
            #"cox_ci_lower": test["cox model"]["C-index CI"][0],
            #"cox_ci_upper": test["cox model"]["C-index CI"][1],
            "spearman_rho": test["rankings OS"]["spearman r"]["rho"],
            "spearman_p": test["rankings OS"]["spearman r"]["p_val"],
            "os_kendall_tau": test["rankings OS"]["kendall tau"]["tau"],
            "os_kendall_p": test["rankings OS"]["kendall tau"]["p_val"],
            "somers_d": test["rankings OS"]["somers d"]["d"],
            "somers_d_p": test["rankings OS"]["somers d"]["p_val"],
        }
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return k, train_result, test_result, None, None

    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return k, None, None, algorithm_name, f"FAILED — {e}"


# ── Run in parallel ────────────────────────────────────────────────────────────

Ncs0 = int(sys.argv[1])
Ncs1 = int(sys.argv[2])
Ncs1   = None if Ncs1==-1 else Ncs1
njobs = int(sys.argv[3])

subset = combinations[Ncs0:Ncs1]
print(f"Running combinations from {Ncs0} to {Ncs1} (total: {len(subset)}) with {njobs} parallel jobs...")

raw_results = list(tqdm(
    Parallel(n_jobs=njobs, backend="loky", return_as="generator")(
        delayed(run_combination)(
            k, combination,
            data_train, data_test,
            mask_train, os_train, status_train,
            ranking_method,
            hcpex_train, hcpex_test,
            daysXmonth,
            random_seed
        )
        for k, combination in enumerate(subset, start=Ncs0)
    ),
    total=len(subset),
    desc="Combinations",
    unit="combo"
))
print(f"All combinations processed ({len(raw_results)}).")

# ── Results ────────────────────────────────────────
results_train, results_test = {}, {}
failed = []

for result in raw_results:
    if result is not None:
        k, train_result, test_result, name, status = result
        if train_result is not None:
            results_train[k] = train_result
            results_test[k]  = test_result
        else:
            failed.append((k, name, status))

print(f"\nDone: {len(results_train)} succeeded, {len(failed)} failed")
if failed:
    for k, name, status in failed:
        print(f"  [{k}] {name} — {status}")

Ncs1   = 'end' if Ncs1 is None else Ncs1
os.makedirs(f"{RESULTS}/UMAP-stats", exist_ok=True)
os.makedirs(f"{RESULTS}/UMAP-stats/Config-outcomes", exist_ok=True)

df_test = pd.DataFrame.from_dict(results_test, orient="index")
df_test.to_csv(f"{RESULTS}/UMAP-stats/Config-outcomes/HCPEX__results-test_{Ncs0}-{Ncs1}.csv", sep=",", index=False)

df_train = pd.DataFrame.from_dict(results_train, orient="index")
df_train.to_csv(f"{RESULTS}/UMAP-stats/Config-outcomes/HCPEX__results_train_{Ncs0}-{Ncs1}.csv", sep=",", index=False)

print(f"Saved {len(df_train)} train results → {RESULTS}/UMAP-stats/Config-outcomes/results_train.csv")
print(f"Saved {len(df_test)}  test  results → {RESULTS}/UMAP-stats/Config-outcomes/results_test.csv")

combinations_key = {}
for k, combination in tqdm(enumerate(subset, start=Ncs0), desc="Prepping combination key dicts"):
    combinations_key[k] = {
        "ALGORITHM": combination["algorithm_name"],
        "NAME": combination["umap_configuration"]
    }

import json
with open(f"{RESULTS}/UMAP-stats/Config-outcomes/HCPEX__key-combinations_{Ncs0}-{Ncs1}.json", 'w', encoding='utf-8') as ff:
    json.dump(combinations_key, ff, ensure_ascii=False, indent=4)

print(f"Saved combinations keys → {RESULTS}/UMAP-stats/Config-outcomes/HCPEX__key-combinations_{Ncs0}-{Ncs1}.json")