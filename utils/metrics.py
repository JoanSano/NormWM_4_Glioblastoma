import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau, spearmanr, somersd
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from sksurv.compare import compare_survival
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def topological_risk_kendalltau(statistics):
    groups = np.array([int(tr) for tr in statistics["training cohort(s)"]["topological risk"].keys()])
    medians = np.array([dt[0] for dt in statistics["training cohort(s)"]["topological risk"].values()])
    tau, pval = kendalltau(groups, medians, variant="b")
    statistics["training cohort(s)"]["kendall tau"]={"tau": tau, "p_val": pval}

    groups = np.array([int(tr) for tr in statistics["testing cohort(s)"]["topological risk"].keys()])
    medians = np.array([dt[0] for dt in statistics["testing cohort(s)"]["topological risk"].values()])
    tau, pval = kendalltau(groups, medians, variant="b")
    statistics["testing cohort(s)"]["kendall tau"]={"tau": tau, "p_val": pval}

    return statistics

def compute_quantile_OS(quantiles, status, os, ci=None, alpha=0.05):

    # --- Branch depending on CI request ---
    if ci is None:
        times, survival_prob = kaplan_meier_estimator(
            status == 1,
            os
        )
    else:
        times, survival_prob, conf = kaplan_meier_estimator(
            status == 1,
            os,
            conf_type=ci,
            conf_level=1 - alpha
        )
        conf = conf.T
        lower_surv = conf[:, 0]
        upper_surv = conf[:, 1]

    pcs = []
    cis = []
    for pc in quantiles:

        if not (0 < pc < 1):
            raise ValueError("All quantiles must be in (0,1)")

        # ---- Point estimate ----
        idx = np.flatnonzero(survival_prob <= pc)
        q = times[idx[0]] if idx.size > 0 else np.inf
        pcs.append(q)

        # ---- Confidence interval via band inversion ----
        if ci is not None:

            # Lower CI bound (invert upper survival band)
            idx_low = np.flatnonzero(lower_surv <= pc)
            q_lower = times[idx_low[0]] if idx_low.size > 0 else np.inf

            # Upper CI bound (invert lower survival band)
            idx_up = np.flatnonzero(upper_surv <= pc)
            q_upper = times[idx_up[0]] if idx_up.size > 0 else np.inf

            cis.append([q_lower, q_upper])

    if ci is None:
        return pcs
    else:
        return pcs, cis

def permutation_p_value(observed, null_dist, higher_is_better=True):
    null_dist = np.array(null_dist)
    if higher_is_better:
        return np.sum(null_dist >= observed) / len(null_dist)
    else:
        return np.sum(null_dist <= observed) / len(null_dist)
    
def topological_risk_concordance_index(
    statistics,
    data_train,
    data_test,
    status="status",
    risk="topological risk",
    survival="OS (days) - corrected",
    alpha_CI=0.05,
    n_permutations=10000,
    n_bootstrap=10000,
    random_state=42
):
    
    rng = np.random.default_rng(random_state)
    features = [survival, status, risk]

    ## Cox model on the training set
    cph = CoxPHFitter()
    cph.fit(data_train[features], duration_col=survival, event_col=status)
    cindex_train = cph.score(       # Compute C-index of the training set
        data_train[features], 
        scoring_method="concordance_index"
    )
    llr = cph.log_likelihood_       # Log-likelihood ratio
    hr = cph.hazard_ratios_         # Hazard ratio
    statistics["training cohort(s)"]["cox model"] = {
        "C-index": cindex_train, 
        "Log-likelihood ratio": llr, 
        "Hazard ratio and p-value": (hr.values[0], cph.summary['p'].values[0])
    }

    # Compute C-index of the testing set
    test_pred = cph.predict_partial_hazard(data_test[features])
    cindex_test = concordance_index(
        data_test[survival],
        -test_pred.values.ravel(),   # negative because higher hazard = worse survival
        data_test[status]
    )
    statistics["testing cohort(s)"]["cox model"] = {"C-index": cindex_test}

    ## Permutation testing
    train_null = []
    test_null = []
    for ip in tqdm(range(n_permutations), desc="Permutation procedure"):
        # Training null C-index by training on permuted data
        perm_idx = rng.permutation(len(data_train))
        perm_train = data_train.copy()
        perm_train[survival] = data_train[survival].values[perm_idx]
        perm_train[status] = data_train[status].values[perm_idx]
        cph_perm = CoxPHFitter()
        cph_perm.fit(perm_train[features], duration_col=survival, event_col=status)
        train_null.append(
            cph_perm.score(perm_train[features], scoring_method="concordance_index")
        )

        # Testing null C-index permuting the test data and using the "true" model
        perm_idx = rng.permutation(len(data_test))
        perm_test = data_test.copy()
        perm_test[survival] = data_test[survival].values[perm_idx]
        perm_test[status] = data_test[status].values[perm_idx]
        test_null.append(
            concordance_index(
                perm_test[survival],
                -test_pred.values.ravel(),
                perm_test[status]
            )
        )
    
    p_train = permutation_p_value(cindex_train, train_null)
    statistics["training cohort(s)"]["cox model"]["C-index p-value (perm)"] = p_train

    p_test = permutation_p_value(cindex_test, test_null)
    statistics["testing cohort(s)"]["cox model"]["C-index p-value (perm)"] = p_test

    ## Confidence intervals with bootstrapping
    if n_bootstrap is not None and n_bootstrap > 0:
        cindex_boot = []
        for ib in tqdm(range(n_bootstrap), desc="Bootsrapping procedure"):
            idx = rng.integers(0, len(data_test), len(data_test))
            cidx = concordance_index(data_test[survival].values[idx], -test_pred.values[idx], data_test[status].values[idx])
            cindex_boot.append(cidx)
        cindex_boot = np.array(cindex_boot)
        lower = np.percentile(cindex_boot, 100 * (alpha_CI / 2))
        upper = np.percentile(cindex_boot, 100 * (1 - alpha_CI / 2))
        statistics["testing cohort(s)"]["cox model"]["C-index CI"] = (lower, upper)

    return statistics

def ranking_OS_topological_risk(
    statistics,
    data_train,
    data_test,
    status="status",
    risk="topological risk",
    survival="OS (days) - corrected"
):
    
    statistics["training cohort(s)"]["rankings OS"] = {}
    statistics["testing cohort(s)"]["rankings OS"] = {}

    ## Sperman r
    mask = data_train[status]==1
    srho, pval = spearmanr(data_train[risk].values[mask], data_train[survival].values[mask])
    statistics["training cohort(s)"]["rankings OS"]["spearman r"]={"rho": srho, "p_val": pval}

    mask = data_test[status]==1
    srho, pval = spearmanr(data_test[risk].values[mask], data_test[survival].values[mask])
    statistics["testing cohort(s)"]["rankings OS"]["spearman r"]={"rho": srho, "p_val": pval}

    ## Kendall's tau
    mask = data_train[status]==1
    tau, pval = kendalltau(data_train[risk].values[mask], data_train[survival].values[mask])
    statistics["training cohort(s)"]["rankings OS"]["kendall tau"]={"tau": tau, "p_val": pval}

    mask = data_test[status]==1
    tau, pval = kendalltau(data_test[risk].values[mask], data_test[survival].values[mask])
    statistics["testing cohort(s)"]["rankings OS"]["kendall tau"]={"tau": tau, "p_val": pval}

    ## Somer's D
    mask = data_train[status]==1
    res = somersd(data_train[risk].values[mask], data_train[survival].values[mask])
    sd, pval = res.statistic, res.pvalue
    statistics["training cohort(s)"]["rankings OS"]["somers d"]={"d": sd, "p_val": pval}

    mask = data_test[status]==1
    res = somersd(data_test[risk].values[mask], data_test[survival].values[mask])
    sd, pval = res.statistic, res.pvalue
    statistics["testing cohort(s)"]["rankings OS"]["somers d"]={"d": sd, "p_val": pval}

    return statistics

def clustering_metrics(umap_risk, statistics):
    ##############################
    ### COMPUTE CLUSTERING METRICS

    statistics["training cohort(s)"]["clustering metrics"] = {
        "silhouette": silhouette_score(umap_risk["umap coordinates: train"], umap_risk["cluster labels: train"]),
        "calinski harabasz": calinski_harabasz_score(umap_risk["umap coordinates: train"], umap_risk["cluster labels: train"]),
        "davies bouldin": davies_bouldin_score(umap_risk["umap coordinates: train"], umap_risk["cluster labels: train"])
    }

    statistics["testing cohort(s)"]["clustering metrics"] = {
        "silhouette": silhouette_score(umap_risk["umap coordinates: test"], umap_risk["cluster labels: test"]),
        "calinski harabasz": calinski_harabasz_score(umap_risk["umap coordinates: test"], umap_risk["cluster labels: test"]),
        "davies bouldin":davies_bouldin_score(umap_risk["umap coordinates: test"], umap_risk["cluster labels: test"])
    }

    return statistics

def process_metric(
        data,
        metric,
        survival="OS (days) - corrected",
        status="status",
        random_state=42,
        n_perm=1000,
        round_decimal=6
    ):

    data = data[[metric, survival, status]].copy()
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col=survival, event_col=status)

    ## C-index ##
    cindex = cph.score(       
        data, 
        scoring_method="concordance_index"
    )
    # One-sided p-value
    rng = np.random.default_rng(random_state)
    permuted_cindices = []
    for _ in range(n_perm):
        permuted_data = data.copy()
        perm_idx = rng.permutation(len(data))
        permuted_data[survival] = data[survival].values[perm_idx]
        permuted_data[status] = data[status].values[perm_idx]

        cph_perm = CoxPHFitter()
        cph_perm.fit(permuted_data, duration_col=survival, event_col=status)
        permuted_cindices.append(cph_perm.score(
            permuted_data,
            scoring_method="concordance_index"
        ))
    permuted_cindices = np.array(permuted_cindices)
    p_value = np.mean(permuted_cindices >= cindex) 

    ## Log-rank 50-50
    y = Surv.from_arrays(event=data[status].values.astype(bool),time=data[survival].values)
    threshold = np.percentile(data[metric].values, 50)
    group = np.where(data[metric].values <= threshold, 0, 1)
    chi2_5050, pval_5050 = compare_survival(y, group)

    ## Log-rank 40-60
    metric_values = data[metric].values
    p40 = np.percentile(metric_values, 40)
    p60 = np.percentile(metric_values, 60)
    low_mask  = metric_values <= p40
    high_mask = metric_values >= p60
    extreme_mask = low_mask | high_mask
    data_extreme = data.loc[extreme_mask].copy()
    y = Surv.from_arrays(event=data_extreme[status].values.astype(bool),time=data_extreme[survival].values)
    group = np.where(data_extreme[metric].values <= p40, 0, 1)
    chi2_4060, pval_4060 = compare_survival(y, group)

    ## Log-rank 25-75
    metric_values = data[metric].values
    p25 = np.percentile(metric_values, 25)
    p75 = np.percentile(metric_values, 75)
    low_mask  = metric_values <= p25
    high_mask = metric_values >= p75
    extreme_mask = low_mask | high_mask
    data_extreme = data.loc[extreme_mask].copy()
    y = Surv.from_arrays(event=data_extreme[status].values.astype(bool),time=data_extreme[survival].values)
    group = np.where(data_extreme[metric].values <= p25, 0, 1)
    chi2_2575, pval_2575 = compare_survival(y, group)

    result_text = {
        "metric": metric,
        "C-index (p-value)": f"{cindex.round(round_decimal)} ({p_value.round(round_decimal)})",
        "HR [95% CI]": f"{cph.hazard_ratios_.values[0].round(round_decimal)} [{cph.summary["exp(coef) lower 95%"].values[0].round(round_decimal)}, {cph.summary["exp(coef) upper 95%"].values[0].round(round_decimal)}]",
        "Log-HR (p-value)": f"{cph.params_.values[0].round(round_decimal)} ({cph.summary["p"].values[0].round(round_decimal)})",
        "Log-rank 50-50 (p-value)": f"{chi2_5050.round(round_decimal)} ({pval_5050.round(round_decimal)})",
        "Log-rank 40-60 (p-value)": f"{chi2_4060.round(round_decimal)} ({pval_4060.round(round_decimal)})",
        "Log-rank 25-75 (p-value)": f"{chi2_2575.round(round_decimal)} ({pval_2575.round(round_decimal)})"
    }

    result_numeric = {
        "metric": metric,
        # C-index
        "cindex": float(cindex),
        "cindex_pvalue": float(p_value),
        # Cox
        "cox_hazard_ratio": float(cph.hazard_ratios_.values[0]),
        "cox_hr_ci_lower_95%": float(cph.summary["exp(coef) lower 95%"].values[0]),
        "cox_hr_ci_upper_95%": float(cph.summary["exp(coef) upper 95%"].values[0]),
        "cox_log_hr": float(cph.params_.values[0]),
        "cox_log_hr_pvalue": float(cph.summary["p"].values[0]),
        # Log-rank splits
        "logrank_5050_chi2": float(chi2_5050),
        "logrank_5050_pvalue": float(pval_5050),
        "logrank_4060_chi2": float(chi2_4060),
        "logrank_4060_pvalue": float(pval_4060),
        "logrank_2575_chi2": float(chi2_2575),
        "logrank_2575_pvalue": float(pval_2575),
    }

    return result_numeric, result_text

def test_metric(
        data_train,
        data_test,
        metric,
        survival="OS (days) - corrected",
        status="status",
        random_state=42,
        n_perm=1000,
        round_decimal=6
    ):

    data_train = data_train[[metric, survival, status]].copy()
    data_test = data_test[[metric, survival, status]].copy()
    
    cph = CoxPHFitter()
    cph.fit(data_train, duration_col=survival, event_col=status)

    ## C-index ##
    test_pred = cph.predict_partial_hazard(data_test)
    cindex = concordance_index(
        data_test[survival],
        -test_pred.values.ravel(),   # negative because higher hazard = worse survival
        data_test[status]
    )
    # One-sided p-value
    rng = np.random.default_rng(random_state)
    permuted_cindices = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(len(data_test))
        permuted_cindices.append(concordance_index(
            data_test[survival].values[perm_idx],
            -test_pred.values.ravel(),
            data_test[status].values[perm_idx]
        ))
    permuted_cindices = np.array(permuted_cindices)
    p_value = np.mean(permuted_cindices >= cindex) 

    ## Log-rank 50-50
    y = Surv.from_arrays(event=data_test[status].values.astype(bool),time=data_test[survival].values)
    threshold = np.percentile(data_train[metric].values, 50)
    group = np.where(data_test[metric].values <= threshold, 0, 1)
    chi2_5050, pval_5050 = compare_survival(y, group)

    ## Log-rank 40-60
    p40 = np.percentile(data_train[metric].values, 40)
    p60 = np.percentile(data_train[metric].values, 60)
    low_mask  = data_test[metric].values <= p40
    high_mask = data_test[metric].values >= p60
    extreme_mask = low_mask | high_mask
    data_extreme = data_test.loc[extreme_mask].copy()
    y = Surv.from_arrays(event=data_extreme[status].values.astype(bool),time=data_extreme[survival].values)
    group = np.where(data_extreme[metric].values <= p40, 0, 1)
    if data_extreme.shape[0] == 0 or len(np.unique(group)) < 2:
        chi2_4060 = np.nan
        pval_4060 = np.nan
    else:
        chi2_4060, pval_4060 = compare_survival(y, group)

    ## Log-rank 25-75
    p25 = np.percentile(data_train[metric].values, 25)
    p75 = np.percentile(data_train[metric].values, 75)
    low_mask  = data_test[metric].values <= p25
    high_mask = data_test[metric].values >= p75
    extreme_mask = low_mask | high_mask
    data_extreme = data_test.loc[extreme_mask].copy()
    y = Surv.from_arrays(event=data_extreme[status].values.astype(bool),time=data_extreme[survival].values)
    group = np.where(data_extreme[metric].values <= p25, 0, 1)
    if data_extreme.shape[0] == 0 or len(np.unique(group)) < 2:
        chi2_2575 = np.nan
        pval_2575 = np.nan
    else:
        chi2_2575, pval_2575 = compare_survival(y, group)

    result_text = {
        "metric": metric,
        "C-index (p-value)": f"{cindex.round(round_decimal)} ({p_value.round(round_decimal)})",
        "HR [95% CI]": f"{cph.hazard_ratios_.values[0].round(round_decimal)} [{cph.summary["exp(coef) lower 95%"].values[0].round(round_decimal)}, {cph.summary["exp(coef) upper 95%"].values[0].round(round_decimal)}]",
        "Log-HR (p-value)": f"{cph.params_.values[0].round(round_decimal)} ({cph.summary["p"].values[0].round(round_decimal)})",
        "Log-rank 50-50 (p-value)": f"{chi2_5050.round(round_decimal)} ({pval_5050.round(round_decimal)})",
        "Log-rank 40-60 (p-value)": f"{chi2_4060.round(round_decimal)} ({pval_4060.round(round_decimal)})",
        "Log-rank 25-75 (p-value)": f"{chi2_2575.round(round_decimal)} ({pval_2575.round(round_decimal)})"
    }

    result_numeric = {
        "metric": metric,
        # C-index
        "cindex": float(cindex),
        "cindex_pvalue": float(p_value),
        # Cox
        "cox_hazard_ratio": float(cph.hazard_ratios_.values[0]),
        "cox_hr_ci_lower_95%": float(cph.summary["exp(coef) lower 95%"].values[0]),
        "cox_hr_ci_upper_95%": float(cph.summary["exp(coef) upper 95%"].values[0]),
        "cox_log_hr": float(cph.params_.values[0]),
        "cox_log_hr_pvalue": float(cph.summary["p"].values[0]),
        # Log-rank splits
        "logrank_5050_chi2": float(chi2_5050),
        "logrank_5050_pvalue": float(pval_5050),
        "logrank_4060_chi2": float(chi2_4060),
        "logrank_4060_pvalue": float(pval_4060),
        "logrank_2575_chi2": float(chi2_2575),
        "logrank_2575_pvalue": float(pval_2575),
    }

    return result_numeric, result_text

def repeated_validation(
        data,
        metrics,
        N_val=100,
        test_size=0.25,
        survival="OS (days) - corrected",
        status="status",
        n_perm=1000,
        round_decimal=6,
        random_state=42
    ):
    
    results_dict = {metric: [] for metric in metrics}
    rng = np.random.default_rng(random_state)

    for i in range(N_val):

        # Random split
        train, test = train_test_split(
            data,
            test_size=test_size,
            random_state=rng.integers(0, 1_000_000),
            shuffle=True,
            stratify=data[status]
        )

        for metric in metrics:

            r_num, _ = test_metric(
                train,
                test,
                metric,
                survival=survival,
                status=status,
                random_state=rng.integers(0, 1_000_000),
                n_perm=n_perm,
                round_decimal=round_decimal
            )

            results_dict[metric].append(r_num)

    # Convert lists → DataFrames
    results = {}
    for metric in metrics:
        results[metric] = pd.DataFrame(results_dict[metric])

    return results

def print_model_summary(label, model, cindex_train, ci_boot, ci_perm, uno=None):
    """Pretty-print log-HRs, HRs, p-values and CIs for one model."""
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")

    summary = model.summary  # lifelines DataFrame
    display_cols = ["exp(coef)", "coef", "exp(coef) lower 95%",
                    "exp(coef) upper 95%", "p"]
    rename = {
        "coef":                  "log-HR",
        "exp(coef)":             "HR",
        "exp(coef) lower 95%":  "HR CI lower",
        "exp(coef) upper 95%":  "HR CI upper",
        "p":                     "p-value",
    }
    out = summary[display_cols].rename(columns=rename)
    out["sig"] = out["p-value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    with pd.option_context("display.float_format", "{:.8f}".format):
        print(out.to_string())

    print(f"\n  Train C-index : {cindex_train:.4f}")
    if uno is not None:
        print(f"  Train Uno's C-index : {uno:.4f}")
    print(f"  Bootstrap C-index : {ci_boot[0]:.4f}  "
        f"(95% CI {ci_boot[1]:.4f} – {ci_boot[2]:.4f})")
    print(f"  C-index permutation p-value : {ci_perm[0]:.6f}")
    print(f"  Log-likelihood    : {model.log_likelihood_:.4f}")

if __name__ == "__main__":
    pass