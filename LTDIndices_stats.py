import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

import seaborn as sns

import scipy
from scipy.stats import mannwhitneyu, linregress, pearsonr, PermutationMethod, BootstrapMethod

from statsmodels.stats.multitest import multipletests, fdrcorrection

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.ensemble import RandomSurvivalForest

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedKFold, RepeatedStratifiedKFold,
    cross_val_score, cross_validate, cross_val_predict, permutation_test_score
)
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from utils.statistics import DeLong_Test, benjamini_bogomolov_procedure, bootstrap_median_os_difference, bootstrap_cindex, permutation_cindex
from utils.metrics import compute_quantile_OS, print_model_summary

def pvalue_to_text(p, nd=4):
    if np.isnan(p):
        return ""
    return "<0.0001" if p < 0.0001 else str(round(p, nd))

def make_annot(matrix):
    annot = np.empty(matrix.shape, dtype=object)
    for idx in np.ndindex(matrix.shape):
        annot[idx] = pvalue_to_text(matrix[idx])
    return annot

def sig_marker(p):
    if p < 0.001:   return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"

####################################################################################################################################################################
## General processing
## Example command --> python LTDIndices_stats.py /home/joan/Desktop/PROJECTS/Glioblastomas/RESULTS-GBM_4-cohorts_Tissues/ LesionTract-density_Tissue-types data-clinical_TD-tissues_4-cohorts.csv --format pdf --cohort -1
####################################################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the directory where the TDI and survival data are stored")
parser.add_argument("results_folder", type=str, help="Name of the folder where the results will be stored")
parser.add_argument("data", type=str, help="Name of the CSV file with the data")
parser.add_argument("--format", type=str, default='pdf', choices=['pdf','svg'], help="Output figure format")
parser.add_argument("--cohort", type=int, default=-1, choices=[-1,0,1,2,3], help="{-1: all cohorts, 0: UCSF, 1: UPENN, 2: TCGA, 3: RHUH}")
args = parser.parse_args()

results_folder = args.results_folder
os.makedirs(os.path.join(args.path, results_folder), exist_ok=True)
try:
    data_o = pd.read_csv(os.path.join(args.path, args.data))
except:
    raise FileNotFoundError("Please copy the database inside the directory you will be working with and has been created.")

daysXmonth = 365/12 
ip, dp = 10, 0.1
percentiles2check = [(i.round(1), (100 - i).round(1)) for i in np.arange(ip, 50, dp)]
percentiles2check.append((50, 50))
percentiles2plot = (25,75),(40,60),(50,50)
n_resamples = 5000 # Bottstrapping and permutation of correlation values
n_perms = 5000 # Permutation of Cox Prop Hazard models
months = np.array([6,12,18,24,30,36,42,48])
nrows, ncols = 1, 5
figsize = (5*ncols, 6*nrows)

# Select cohort if applicable
if args.cohort in range(4):
    data_o = data_o.loc[
        data_o["cohort"]==args.cohort
    ].copy()

TDMaps = data_o[
    [
        "OS (days) - corrected",
        "Whole lesion TDMap",
        "Core lesion TDMap",
        "Non-enhancing lesion TDMap",
        "Enhancing lesion TDMap",
        "Core+Enhancing lesion TDMap",
        "status"
    ]
].copy().rename(
    columns={
        "OS (days) - corrected": "OS",
        "Whole lesion TDMap": "W.L-TDI",
        "Core lesion TDMap": "C.L-TDI",
        "Non-enhancing lesion TDMap": "NE.L-TDI",
        "Enhancing lesion TDMap": "E.L-TDI",
        "Core+Enhancing lesion TDMap": "C+E.L-TDI",
        "status": "status"
        }
    )

# To study the common subset of patients with complete segmentations
TDMaps = TDMaps.dropna(subset=["W.L-TDI", "C.L-TDI", "NE.L-TDI", "E.L-TDI", "C+E.L-TDI"])

life = TDMaps["status"].values
TDMaps.drop(columns=["status"], inplace=True)

####################################################################################################################################################################
## General numbers
####################################################################################################################################################################
print("+++++++++++++++++++++++++++++\nNumber of samples per each group")
for i in range(1,len(TDMaps.columns)):
    x = TDMaps[TDMaps.columns[i]]
    y = TDMaps["OS"]
    
    # Remove rows where x or y is NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if i==1:
        print(f"OS: {len(y_clean)} Patients")
    print(f"{TDMaps.columns[i]}: {len(x_clean)} Patients")

a,b,c = np.count_nonzero(~np.isnan(life)), np.nansum(life), int(np.nansum(np.where(life==0,1,np.nan)))
print(f"No. of patients with a registered event (1-dead/0-alive): {a}")
print(f"No. of dead patients (without right censoring): {b} ({round(100*b/a,2)}%)")
print(f"No. of patients with right censoring: {c} ({round(100*c/a,2)}%)")

####################################################################################################################################################################
## Correlation coefficient between TD Maps and OS
####################################################################################################################################################################
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(27, 9))
cross_TD = np.zeros((len(TDMaps.columns)+1, len(TDMaps.columns))) * np.nan
cross_TD_p = np.zeros((len(TDMaps.columns)+1, len(TDMaps.columns))) * np.nan
for i in range(len(TDMaps.columns)):
    y = TDMaps[TDMaps.columns[i]]
    for j in range(i,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[j]]

        # Remove rows where x or y is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        if i==0:
            cross_TD[0,j], cross_TD_p[0,j] = pearsonr(x_clean[life_clean==1], y_clean[life_clean==1], alternative='two-sided')
            cross_TD[-1,j], cross_TD_p[-1,j] = pearsonr(x_clean[life_clean==0], y_clean[life_clean==0], alternative='two-sided')
        else:
            cross_TD[i,j], cross_TD_p[i,j] = pearsonr(x_clean, y_clean, alternative='two-sided')
sns.heatmap(np.round(cross_TD[1:-1,1:],4), annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax1, 
            xticklabels=TDMaps.columns[1:], yticklabels=TDMaps.columns[1:], cbar_kws={"shrink": 0.7})
ax1.set_title('Pearson Correlation Coefficients', fontweight='bold', fontsize=12)
ax1.tick_params(axis='both', length=0) 
sns.heatmap(np.round(cross_TD_p[1:-1,1:],4), annot=make_annot(cross_TD_p[1:-1, 1:]), cmap='viridis', square=True, ax=ax2, 
            xticklabels=TDMaps.columns[1:], yticklabels=TDMaps.columns[1:], cbar_kws={"shrink": 0.7}, fmt='')
ax2.set_title('p-values for Correlation Coefficients', fontweight='bold', fontsize=12)
ax2.tick_params(axis='both', length=0) 
# Method: Benjamin-Hochberg ---> ALL the p-values are used but only the TDmaps are plotted
n_td = len(TDMaps.columns) - 1
r1_cols = np.arange(1, n_td + 1)
r1_p = cross_TD_p[0, r1_cols]
r2_cols = np.arange(1, n_td + 1)
r2_p = cross_TD_p[-1, r2_cols]
triu_rows, triu_cols = np.triu_indices(n_td, k=0)
r3_p = cross_TD_p[1:-1, 1:][triu_rows, triu_cols]
all_p = np.concatenate([r1_p, r2_p, r3_p])
valid = ~np.isnan(all_p)
fdr_all = np.full(len(all_p), np.nan)
if valid.any():
    _, fdr_all[valid] = fdrcorrection(all_p[valid], alpha=0.05, method='p', is_sorted=False)
n1 = len(r1_p)
n2 = len(r2_p)
fdr_r1, fdr_r2, fdr_r3 = fdr_all[:n1], fdr_all[n1:n1+n2], fdr_all[n1+n2:]
cross_TD_p_corrected = np.full_like(cross_TD_p, np.nan)
mask_r1 = ~np.isnan(r1_p)
cross_TD_p_corrected[0, r1_cols[mask_r1]] = fdr_r1[mask_r1]
mask_r2 = ~np.isnan(r2_p)
cross_TD_p_corrected[-1, r2_cols[mask_r2]] = fdr_r2[mask_r2]
mask_r3 = ~np.isnan(r3_p)
cross_TD_p_corrected[1:-1, 1:][triu_rows[mask_r3], triu_cols[mask_r3]] = fdr_r3[mask_r3]
mask_tri = np.tril(np.ones_like(cross_TD[1:-1, 1:], dtype=bool), k=-1)
sns.heatmap(np.round(cross_TD_p_corrected[1:-1,1:],4), annot=make_annot(cross_TD_p_corrected[1:-1, 1:]), cmap='viridis', square=True, ax=ax3, 
            xticklabels=TDMaps.columns[1:], yticklabels=TDMaps.columns[1:], cbar_kws={"shrink": 0.7}, fmt='')
ax3.set_title("FDR Corrected", fontweight='bold', fontsize=12)
ax3.tick_params(axis='both', length=0)
fig.tight_layout()
fig.savefig(os.path.join(args.path, results_folder, f"correlation-TDMaps.{args.format}"), dpi=300, format=args.format)
plt.close()

####################################################################################################################################################################
## Correlation coefficient between OS and TDMetrics --> Using results and corrections from the previous section
####################################################################################################################################################################
for status in [0,1]:
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]    
        # Remove rows where x or y is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        # Plot the scatter plot and regression line with confidence intervals
        sns.regplot(x=x_clean[life_clean==status], y=y_clean[life_clean==status]/daysXmonth, ax=ax[i-1], scatter_kws={'s': 15, 'color': 'black'}, ci=95)    
        # Calculate the linear regression to get the R² value
        slope, intercept, r_value, p_value, std_err = linregress(x_clean[life_clean==status], y_clean[life_clean==status])
        r_squared = r_value**2    
        if status==0:
            ptext = cross_TD_p[-1,i]
            ptextcorr = cross_TD_p_corrected[-1,i]
        else:
            ptext = cross_TD_p[0,i]
            ptextcorr = cross_TD_p_corrected[0,i]
        ax[i-1].text(0.55, 0.95, f'R² = {r_squared:.4f} \n'+r'$\rho$'+f' = {r_value:.4f} \n'+r'$p$'+f" = {pvalue_to_text(ptext)} \n"+r'$p_{corrected}$'+f" = {pvalue_to_text(ptextcorr)}", transform=ax[i-1].transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_value<=0.05 else "black")    
        # Set the labels and clean up the plot
        ax[i-1].set_xlabel(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[i-1].set_ylabel("Overall survival (months)", fontweight="bold", fontsize=12)
        ax[i-1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, results_folder, f"OS-TDMaps_scatter_status-{status}.{args.format}"), dpi=300, format=args.format)
    plt.close()

####################################################################################################################################################################
## Survival analyses
####################################################################################################################################################################
os.makedirs(os.path.join(args.path, results_folder, "survival"), exist_ok=True)
fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
ax = ax.flatten()
pv_s = np.zeros((len(TDMaps.columns)-1, len(months)))
maxsY = np.zeros((len(TDMaps.columns)-1, len(months)))
for i in range(1,len(TDMaps.columns)):
    x = TDMaps[TDMaps.columns[i]]
    y = TDMaps["OS"]    

    # Remove rows where x or y is NaN
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
    x_clean = x[mask]
    y_clean = y[mask]
    life_clean = life[mask]    
    maxY = 0
    nums = np.zeros((len(months),2))
    for j,m in enumerate(months):
        tdi_alive = x_clean[y_clean>=m*daysXmonth] # Subjects with OS higher than cutoff are alive
        tdi_dead = x_clean[(y_clean<=m*daysXmonth) & (life_clean==1)] # Only dead subjects with OS lower than cutoff
        _, pv_s[i-1,j] = mannwhitneyu(tdi_alive, tdi_dead, alternative='two-sided')  
        nums[j,:] = [len(tdi_alive), len(tdi_dead)]
        mx = int(max([tdi_alive.max(), tdi_dead.max()]))
        if mx>maxY:
            maxY = mx+1
        ax[i-1].plot(np.full(len(tdi_alive),m-1),tdi_alive,'o', markersize=1.5, color='forestgreen', label=f"Status = 0 (alive)" if j==0 else None)
        ax[i-1].plot(np.full(len(tdi_dead),m+1),tdi_dead,'o', markersize=1.5, color='darkorange', label=f"Status = 1 (dead)" if j==0 else None)
        ax[i-1].plot([m-1,m+1],[np.median(tdi_alive), np.median(tdi_dead)], '-s', linewidth=2.5, markersize=6, color='black')
        ax[i-1].plot([m-1,m-1],[np.median(tdi_alive), np.percentile(tdi_alive, 75)], '-+', linewidth=1.5, markersize=5, color='black')
        ax[i-1].plot([m+1,m+1],[np.median(tdi_dead), np.percentile(tdi_dead, 75)], '-+', linewidth=1.5, color='black')
        
        ax[i-1].text(m-2, -0.075*maxY, f"{int(nums[j,0])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="forestgreen") # Numbers alive
        ax[i-1].text(m-2, -0.125*maxY, f"{int(nums[j,1])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="darkorange") # Numbers dead
        if pv_s[i-1,j]<=0.001:
            ax[i-1].text(m-1.5, 1.05*maxY, '***', color='black',fontsize=10)
        elif pv_s[i-1,j]<=0.01:
            ax[i-1].text(m-1, 1.05*maxY, '**', color='black',fontsize=10)
        elif pv_s[i-1,j]<=0.05:
            ax[i-1].text(m-.5, 1.05*maxY, '*', color='black',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.05*maxY, 'n.s.', color='black',fontsize=10)
        maxsY[i-1,j] = maxY
    ax[i-1].text(months[0]-2, -0.01*maxY, "No. of samples", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
    
    ax[i-1].spines[["top", "right"]].set_visible(False)
    ax[i-1].set_ylabel(f"{TDMaps.columns[i]} (a.u.)", fontsize=12)
    ax[i-1].set_xlabel("Survival (months)", fontsize=12)
    ax[i-1].set_xlim([months[0]-5, months[-1]+5])
    ax[i-1].set_xticks(months)
    ax[i-1].set_xticklabels(months)
    ax[i-1].set_ylim([-maxY/5,7*maxY/6])
    ax[i-1].set_yticks([0,maxY])
    ax[i-1].set_yticklabels([0,"MAX"])
    ax[i-1].spines['left'].set_bounds(0, maxY)
    ax[i-1].spines['bottom'].set_bounds(months[0], months[-1])
    if (i-1)==0:
        ax[i-1].legend(frameon=True, ncols=1, loc="upper right")
    
BB_outcomes = benjamini_bogomolov_procedure(pv_s, alpha=0.05, global_test='simes', family_method='fdr_bh', inner_method='fdr_bh')  
for i in range(1,len(TDMaps.columns)):
    for j, m in enumerate(months):
        if BB_outcomes.hypotheses_pvals_adj[i-1][j]<=0.001:
            ax[i-1].text(m-1.5, 1.1*maxsY[i-1,j], '***', color='blue',fontsize=10)
        elif BB_outcomes.hypotheses_pvals_adj[i-1][j]<=0.01:
            ax[i-1].text(m-1, 1.1*maxsY[i-1,j], '**', color='blue',fontsize=10)
        elif BB_outcomes.hypotheses_pvals_adj[i-1][j]<=0.05:
            ax[i-1].text(m-.5, 1.1*maxsY[i-1,j], '*', color='blue',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.1*maxsY[i-1,j], 'n.s.', color='blue',fontsize=10)
       
fig.tight_layout()
fig.savefig(os.path.join(args.path, results_folder, f"survival/Survival-TDMaps_step-monthly.{args.format}"), dpi=300, format=args.format)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(6,3))
alpha = 0.05
x = np.arange(len(TDMaps.columns) - 1)

pvals = BB_outcomes.family_pvals
pvals_adj = BB_outcomes.family_pvals_adj

ax.plot(x, np.log10(pvals),'s', markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.5, linestyle='None', label="Uncorrected")
ax.plot(x, np.log10(pvals_adj), 'x', color='blue', markeredgewidth=1.5, linestyle='None', label="Corrected")
ax.plot([0,len(TDMaps.columns)-2], [np.log10(0.05), np.log10(0.05)], linestyle="--", color='red', linewidth=0.75)
ax.plot([0,len(TDMaps.columns)-2], [np.log10(0.01), np.log10(0.01)], linestyle="--", color='gray', linewidth=0.5)
ax.plot([0,len(TDMaps.columns)-2], [np.log10(0.001), np.log10(0.001)], linestyle="--", color='gray', linewidth=0.5)
ax.plot([0,len(TDMaps.columns)-2], [np.log10(0.0001), np.log10(0.0001)], linestyle="--", color='gray', linewidth=0.5)

ax.spines[["top", "right"]].set_visible(False)
ax.spines['left'].set_bounds(np.log10(0.1), np.log10(0.00000001))
ax.spines['bottom'].set_bounds([0,len(TDMaps.columns)-2])
ax.set_xticks(range(len(TDMaps.columns)-1))
ax.set_xticklabels(TDMaps.columns[1:])
ax.set_yticks([np.log10(0.05), np.log10(0.01), np.log10(0.001), np.log10(0.0001)])
ax.set_yticklabels(["5*10E-2", "10E-2", "10E-3", "<10E-4"])
ax.set_title("Family-wise combined p-value", fontweight="bold")
ax.legend(frameon=True)

fig.savefig(os.path.join(args.path, results_folder, f"survival/Survival-TDMaps_step-monthly_family-selection.{args.format}"), dpi=300, format=args.format)
plt.close()

fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

pvals = pv_s
pvals_adj = BB_outcomes.hypotheses_pvals_adj

for i in range(1,len(TDMaps.columns)):
    # U-test of size differences and survival
    ax[i-1].plot(pvals[i-1], "-o", color="black", label="Uncorrected", linewidth=2)
    ax[i-1].plot(pvals_adj[i-1], "-o", color="blue", label="Corrected", linewidth=.75, alpha=.5)
    ax[i-1].axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="p<0.05")
    ax[i-1].axhline(y=0.01, color="gray", linestyle="--", linewidth=1)
    ax[i-1].axhline(y=0.0001, color="gray", linestyle="--", linewidth=.5)
    ax[i-1].set_yscale("log")
    ax[i-1].set_xticks(range(0, len(months)))
    ax[i-1].set_xticklabels(months)
    ax[i-1].spines['bottom'].set_bounds(0, len(months)-1)
    ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
    ax[i-1].set_xlabel("Survival (months)", fontsize=12)
    ax[i-1].set_ylabel(r'$p-value \ (U)$', fontsize=12)
    ax[i-1].spines[["top", "right"]].set_visible(False)
    if i == 1:
        ax[i-1].legend(frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(args.path, results_folder, f"survival/BB-corrected_pvals.{args.format}"), dpi=300, format=args.format)
plt.close()

####################################################################################################################################################################
## Death analyses
####################################################################################################################################################################
print("Death earlier than X months\n---------------------")
os.makedirs(os.path.join(args.path, results_folder, "death"), exist_ok=True)
for p_iter, (plow, phigh) in enumerate(percentiles2plot):
    print(f"Percentiles ({plow},{phigh})")

    ## Death earlier than X months 
    perc = plow
    fig, ax = plt.subplots(nrows*2, ncols, figsize=figsize)
    ax = ax.flatten()
    k_ax = 0 
    rho_s = np.zeros((len(TDMaps.columns)-1, len(months)))
    pv_s_rho = np.zeros((len(TDMaps.columns)-1, len(months)))
    pv_s_u = np.zeros((len(TDMaps.columns)-1, len(months)))
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]    

        # Remove rows where x or y is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life) & life==1
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        ax[k_ax].text(months[0]-2, -0.375, "No. of deaths", transform=ax[k_ax].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
        rs = np.zeros((len(months),4)) # rho, pval, low CI, high CI
        pv_us = np.zeros((len(months),2))
        for j,m in enumerate(months):
            mask_months = y_clean<=(m*daysXmonth)
            x_masked = x_clean[mask_months]
            y_masked = y_clean[mask_months]
            # Correlation
            result_rho = pearsonr(
                x_masked, y_masked, 
                method=PermutationMethod(n_resamples=n_resamples), 
                alternative='two-sided'
            )
            rs[j,0], rs[j,1] = result_rho[0], result_rho[1]
            rs[j,2:] = result_rho.confidence_interval(0.95, method=BootstrapMethod(n_resamples=n_resamples))
            if rs[j,1]<=0.001:
                ax[k_ax].text(m-1.5, .425, '***', color='black',fontsize=10, transform=ax[k_ax].transData)
            elif rs[j,1]<=0.01:
                ax[k_ax].text(m-1, .425, '**', color='black',fontsize=10, transform=ax[k_ax].transData)
            elif rs[j,1]<=0.05:
                ax[k_ax].text(m-.5, .425, '*', color='black',fontsize=10, transform=ax[k_ax].transData)
            else:            
                ax[k_ax].text(m-1.5, .425, 'n.s.', color='black',fontsize=10, transform=ax[k_ax].transData)
            ax[k_ax].text(m-2, -0.525, f"{len(y_masked)}", transform=ax[k_ax].transData, fontsize=12, verticalalignment='top', color="black") # Numbers
            # OS 
            y_masked_small = y_masked[x_masked<=np.percentile(x_masked, perc)]
            y_masked_big = y_masked[x_masked>=np.percentile(x_masked, 100-perc)]
            _, pv = mannwhitneyu(y_masked_small, y_masked_big, alternative='two-sided')
            pv_s_u[i-1,j] = pv
            ax[k_ax+5].plot(np.full(len(y_masked_small),m-1),y_masked_small.values,'o', markersize=2, color='royalblue', label=f"Low L-TDI (P<={perc})" if j==0 else None)
            ax[k_ax+5].plot(np.full(len(y_masked_big),m+1),y_masked_big.values,'o', markersize=2, color='salmon', label=f"High L-TDI (P>={100-perc})" if j==0 else None)
            if pv<=0.001:
                ax[k_ax+5].text(m-1.5, 1450, '***', color='black',fontsize=10)
            elif pv<=0.01:
                ax[k_ax+5].text(m-1, 1450, '**', color='black',fontsize=10)
            elif pv<=0.05:
                ax[k_ax+5].text(m-.5, 1450, '*', color='black',fontsize=10)
            else:            
                ax[k_ax+5].text(m-1.5, 1450, 'n.s.', color='black',fontsize=10)
            ax[k_ax+5].plot([m-1,m+1],[np.median(y_masked_small), np.median(y_masked_big)], '-s', linewidth=3, markersize=5, color='black')
            ax[k_ax+5].plot([m-1,m-1],[np.median(y_masked_small), np.percentile(y_masked_small, 75)], '-+', linewidth=2, markersize=5, color='black')
            ax[k_ax+5].plot([m+1,m+1],[np.median(y_masked_big), np.percentile(y_masked_big, 75)], '-+', linewidth=2, color='black')
            ax[k_ax+5].text(m-2, 2020, f"{len(y_masked_small)}", transform=ax[k_ax+5].transData, fontsize=12, verticalalignment='top', color="royalblue") # Numbers
            ax[k_ax+5].text(m-2, 1820, f"{len(y_masked_big)}", transform=ax[k_ax+5].transData, fontsize=12, verticalalignment='top', color="salmon") # Numbers
        pv_s_rho[i-1,:] = rs[:,1]
        rho_s[i-1,:] = rs[:,0]
        # Set the labels and clean up the plot
        ax[k_ax].plot(months,rs[:,0],'-o', linewidth=3, markersize=15, color='black')
        ax[k_ax].fill_between(months, y1=rs[:,2], y2=rs[:,3], color='black', alpha=.15, edgecolor=None)
        ax[k_ax].hlines(0, months[0]-5, months[-1]+5, color='gray', alpha=.75, linewidth=.75, linestyle='--')
        ax[k_ax].set_xlim([months[0]-5, months[-1]+5])
        ax[k_ax].set_xticks(months)
        ax[k_ax].set_xticklabels([])
        ax[k_ax].tick_params(axis='x', which='both', bottom=False) 
        ax[k_ax].set_ylim([-.5,.6])
        ax[k_ax].set_yticks([-.4,-.2,0,.2,.4,.6])
        ax[k_ax].set_yticklabels([-0.4,-0.2,0,0.2,0.4,0.6])
        ax[k_ax].spines['left'].set_bounds(-.4,.6)
        ax[k_ax].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[k_ax].set_ylabel("Pearson "+r'$\rho$'+" (status=1)", fontsize=12)
        ax[k_ax].spines[["top", "right", "bottom"]].set_visible(False)
        ax[k_ax+5].set_ylabel("Overall survival (months)", fontsize=12)
        ax[k_ax+5].set_xlabel("Death cutoff ("+r'$\leq$'+"months)", fontsize=12)
        ax[k_ax+5].set_xlim([months[0]-5, months[-1]+5])
        ax[k_ax+5].set_xticks(months)
        ax[k_ax+5].set_xticklabels(months)
        ax[k_ax+5].set_ylim([-10,1620])
        ax[k_ax+5].set_yticks([0]+list(months*daysXmonth)+[(months[-1]+6)*daysXmonth])
        ax[k_ax+5].set_yticklabels([0]+list(+months)+[months[-1]+6])
        ax[k_ax+5].spines['bottom'].set_bounds(months[0], months[-1])
        ax[k_ax+5].spines['left'].set_bounds(0,1620)
        ax[k_ax+5].spines[["top", "right"]].set_visible(False)
        if (i-1)==0:
            ax[k_ax+5].legend(frameon=False, ncols=1, loc='center left')
        if (i-1)==4:
            k_ax += 6
        else:
            k_ax += 1
 
    BB_outcomes_rho = benjamini_bogomolov_procedure(pv_s_rho, alpha=0.05, global_test='simes', family_method='fdr_bh', inner_method='fdr_bh')   
    BB_outcomes_u = benjamini_bogomolov_procedure(pv_s_u, alpha=0.05, global_test='simes', family_method='fdr_bh', inner_method='fdr_bh')
    
    k_ax=0
    for i in range(1,len(TDMaps.columns)):
        ax[k_ax].plot(
            months[BB_outcomes_rho.hypotheses_pvals_adj[i-1]<=0.05],
            rho_s[i-1,BB_outcomes_rho.hypotheses_pvals_adj[i-1]<=0.05],
            'o', 
            markersize=5, 
            color='red'
        )
        for j, m in enumerate(months):
            if BB_outcomes_rho.hypotheses_pvals_adj[i-1][j]<=0.001:
                ax[k_ax].text(m-1.5, .5, '***', color='blue',fontsize=10, transform=ax[k_ax].transData)
            elif BB_outcomes_rho.hypotheses_pvals_adj[i-1][j]<=0.01:
                ax[k_ax].text(m-1, .5, '**', color='blue',fontsize=10, transform=ax[k_ax].transData)
            elif BB_outcomes_rho.hypotheses_pvals_adj[i-1][j]<=0.05:
                ax[k_ax].text(m-.5, .5, '*', color='blue',fontsize=10, transform=ax[k_ax].transData)
            else:            
                ax[k_ax].text(m-1.5, .5, 'n.s.', color='blue',fontsize=10, transform=ax[k_ax].transData)
            if BB_outcomes_u.hypotheses_pvals_adj[i-1][j]<=0.001:
                ax[k_ax+5].text(m-1.5, 1550, '***', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
            elif BB_outcomes_u.hypotheses_pvals_adj[i-1][j]<=0.01:
                ax[k_ax+5].text(m-1, 1550, '**', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
            elif BB_outcomes_u.hypotheses_pvals_adj[i-1][j]<=0.05:
                ax[k_ax+5].text(m-.5, 1550, '*', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
            else:            
                ax[k_ax+5].text(m-1.5, 1550, 'n.s.', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
        if (i-1)==4:
            k_ax += 6
        else:
            k_ax += 1
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, results_folder, f"death/OS-TDMaps_death-cutoff_status-1_p-{perc}.{args.format}"), dpi=300, format=args.format)

    fig, ax = plt.subplots(1, 2, figsize=(12,3))
    alpha = 0.05
    x = np.arange(len(TDMaps.columns) - 1)

    pvals = BB_outcomes_rho.family_pvals
    pvals_adj = BB_outcomes_rho.family_pvals_adj

    ax[0].plot(x, np.log10(pvals),'s', markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.5, linestyle='None', label="Uncorrected")
    ax[0].plot(x, np.log10(pvals_adj), 'x', color='blue', markeredgewidth=1.5, linestyle='None', label="Corrected")
    ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.05), np.log10(0.05)], linestyle="--", color='red', linewidth=0.75)
    ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.01), np.log10(0.01)], linestyle="--", color='gray', linewidth=0.5)
    ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.001), np.log10(0.001)], linestyle="--", color='gray', linewidth=0.5)
    ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.0001), np.log10(0.0001)], linestyle="--", color='gray', linewidth=0.5)

    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].spines['left'].set_bounds(np.log10(0.1), np.log10(0.0001))
    ax[0].spines['bottom'].set_bounds([0,len(TDMaps.columns)-2])
    ax[0].set_xticks(range(len(TDMaps.columns)-1))
    ax[0].set_xticklabels(TDMaps.columns[1:])
    ax[0].set_yticks([np.log10(0.05), np.log10(0.01), np.log10(0.001), np.log10(0.0001)])
    ax[0].set_yticklabels(["5*10E-2", "10E-2", "10E-3", "<10E-4"])
    ax[0].set_ylabel(r'$p-value \ (\rho)$')
    ax[0].legend(frameon=True)

    pvals = BB_outcomes_u.family_pvals
    pvals_adj = BB_outcomes_u.family_pvals_adj

    ax[1].plot(x, np.log10(pvals),'s', markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.5, linestyle='None', label="Uncorrected")
    ax[1].plot(x, np.log10(pvals_adj), 'x', color='blue', markeredgewidth=1.5, linestyle='None', label="Corrected")
    ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.05), np.log10(0.05)], linestyle="--", color='red', linewidth=0.75)
    ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.01), np.log10(0.01)], linestyle="--", color='gray', linewidth=0.5)
    ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.001), np.log10(0.001)], linestyle="--", color='gray', linewidth=0.5)
    ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.0001), np.log10(0.0001)], linestyle="--", color='gray', linewidth=0.5)

    ax[1].spines[["top", "right"]].set_visible(False)
    ax[1].spines['left'].set_bounds(np.log10(0.1), np.log10(0.00000001))
    ax[1].spines['bottom'].set_bounds([0,len(TDMaps.columns)-2])
    ax[1].set_xticks(range(len(TDMaps.columns)-1))
    ax[1].set_xticklabels(TDMaps.columns[1:])
    ax[1].set_yticks([np.log10(0.05), np.log10(0.01), np.log10(0.001), np.log10(0.0001)])
    ax[1].set_yticklabels(["5*10E-2", "10E-2", "10E-3", "<10E-4"])
    ax[1].set_ylabel(r'$p-value \ (U)$')
    
    fig.suptitle("Family-wise combined p-value", fontweight="bold")
    fig.savefig(os.path.join(args.path, results_folder, f"death/OS-TDMaps_death-cutoff_status-1_p-{perc}_family-selection.{args.format}"), dpi=300, format=args.format)
    plt.close()
print("++++"*40) 

####################################################################################################################################################################
## Median survival and Kaplan-Meier analyses
####################################################################################################################################################################
KMcurves_ps = np.zeros((len(TDMaps.columns)-1, len(percentiles2check)))
Median_ps = np.zeros((len(TDMaps.columns)-1, len(percentiles2check)))
Median_OS = {(plow, phigh): np.zeros((len(TDMaps.columns)-1, 6)) for (plow, phigh) in percentiles2check}
DiffMedian_OS = {(plow, phigh): np.zeros((len(TDMaps.columns)-1, 3)) for (plow, phigh) in percentiles2check}
os.makedirs(os.path.join(args.path, results_folder, "median-os_uncensored"), exist_ok=True)
os.makedirs(os.path.join(args.path, results_folder, "kaplan-meier"), exist_ok=True)

for p_iter, (plow, phigh) in enumerate(percentiles2check):
    ### Median survival of uncensored patients
    if (plow, phigh) in percentiles2plot:
        print(f"Percentiles ({plow},{phigh})\n---------------------")
        print("Median survival times of uncensored patients [1st, 3rd] quartiles (MONTHS)>>")
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.flatten()

    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]        
        # Remove rows where x or y is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life) & life==1
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        # Calculate the 25th and 75th percentiles
        psmall = y_clean[x_clean<np.percentile(x_clean, plow)]
        pbig = y_clean[x_clean>np.percentile(x_clean, phigh)]
        # Obtain the status of each patient
        lifesmall = life_clean[x_clean<np.percentile(x_clean, plow)]
        lifebig = life_clean[x_clean>np.percentile(x_clean, phigh)]
        # Stats
        average_small, average_large = np.mean(psmall[lifesmall==1]/daysXmonth), np.mean(pbig[lifebig==1]/daysXmonth)
        Ustat, pv = mannwhitneyu(psmall[lifesmall==1], pbig[lifebig==1], alternative='two-sided')
        Median_ps[i-1,p_iter] = pv

        if (plow, phigh) in percentiles2plot:
            print(
                f"\t{TDMaps.columns[i]}: {average_small} [{np.quantile(psmall[lifesmall==1]/daysXmonth, 0.25)},{np.quantile(psmall[lifesmall==1]/daysXmonth, 0.75)}] / {average_large} [{np.quantile(pbig[lifebig==1]/daysXmonth, 0.25)},{np.quantile(pbig[lifebig==1]/daysXmonth, 0.75)}]"
            )
            ax[i-1].boxplot(
                [psmall[lifesmall==1]/daysXmonth, pbig[lifebig==1]/daysXmonth], 
                tick_labels=[f"Low {TDMaps.columns[i]} (P{plow})", f"High {TDMaps.columns[i]} (P{phigh})"],
                positions=[1,2],
                widths=[0.4,0.4]
            )
            ax[i-1].text(0.35, 0.85, f'U = {Ustat} \n'+r'$p$'+f' = {pvalue_to_text(pv)}', transform=ax[i-1].transAxes, 
                        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if pv<=0.05 else "black")        
            # Set the title and labels
            ax[i-1].set_xlim([.5,2.5])
            ax[i-1].set_ylabel("OS (months; status=1)", fontsize=12)
            ax[i-1].spines[["top", "right"]].set_visible(False)
    if (plow, phigh) in percentiles2plot:
        fig.tight_layout()
        fig.savefig(os.path.join(args.path, results_folder, f"median-os_uncensored/OS-TDMaps_percentiles-{plow}-{phigh}.{args.format}"), dpi=300, format=args.format)
        plt.close()

    ### Kaplan-Meier analyses with censoring
    if (plow, phigh) in percentiles2plot:
        print("Median overall survival times [p/m 95% CI] quartiles (MONTHS)>>")
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.flatten()
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]        

        # Remove rows where x, y, or life is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]

        # Calculate the 25th and 75th percentiles
        psmall = y_clean[x_clean<np.percentile(x_clean, plow)]
        pbig = y_clean[x_clean>np.percentile(x_clean, phigh)]

        # Obtain the status of each patient --> True or False indicating whether the entry is right censored (False) or not (True)
        lifesmall = life_clean[x_clean<np.percentile(x_clean, plow)]==1  
        lifebig = life_clean[x_clean>np.percentile(x_clean, phigh)]==1   

        # Low-risk group
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifesmall, psmall, conf_type="log-log"
        )
        med, cis = compute_quantile_OS([0.5], lifesmall, psmall, ci='log-log', alpha=0.05)
        Median_OS[(plow, phigh)][i-1,:3] = [med[0], cis[0][0], cis[0][1]]
        if (plow, phigh) in percentiles2plot:
            ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"Low L-TDI", color="royalblue")
            ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="royalblue")
            for t in psmall[lifesmall==0].values: # Censoring times
                ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='royalblue') 

        # High-risk group
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifebig, pbig, conf_type="log-log"
        )
        med, cis = compute_quantile_OS([0.5], lifebig, pbig, ci='log-log', alpha=0.05)
        Median_OS[(plow, phigh)][i-1,3:] = [med[0], cis[0][0], cis[0][1]]
        if (plow, phigh) in percentiles2plot:
            ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"High L-TDI", color="salmon")
            ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="salmon")
            for t in pbig[lifebig==0].values: # Censoring times
                ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='salmon')

        Median_OS[(plow, phigh)][i-1,:] /= daysXmonth
        if (plow, phigh) in percentiles2plot:
            print(
                f"\t{TDMaps.columns[i]}: {Median_OS[(plow, phigh)][i-1,0]} [{Median_OS[(plow, phigh)][i-1,1]},{Median_OS[(plow, phigh)][i-1,2]}] / {Median_OS[(plow, phigh)][i-1,3]} [{Median_OS[(plow, phigh)][i-1,4]},{Median_OS[(plow, phigh)][i-1,5]}]"
            )
        
        # Bootstrap the difference to obtain the 95% CIs
        bt_diffs = bootstrap_median_os_difference(psmall, lifesmall, pbig, lifebig, n_boot=200, alpha=0.05)
        DiffMedian_OS[(plow, phigh)][i-1,:] = [bt_diffs[0], bt_diffs[1][0], bt_diffs[1][1]]
        DiffMedian_OS[(plow, phigh)] /= daysXmonth
        
        # Numbers
        if (plow, phigh) in percentiles2plot:
            ax[i-1].text(-2, -0.025, "No. at risk", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
            for t in [0,10,20,30,40,50,60,70,80,90,100]:
                num_small = (psmall>=(t*daysXmonth)).sum()
                num_big = (pbig>=(t*daysXmonth)).sum()
                ax[i-1].text(t-2, -0.075, f"{num_small}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="royalblue") 
                ax[i-1].text(t-2, -0.125, f"{num_big}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="salmon") 

        # Stats
        OS_STATS = []
        OS_STATS.extend([(st, os) for st,os in zip(lifesmall,psmall.values)])
        OS_STATS.extend([(st, os) for st,os in zip(lifebig,pbig.values)])
        OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
        TD_STATS = [1 for os in psmall.values]
        TD_STATS.extend([2 for os in pbig.values])
        chisquared, p_val, stats, covariance = compare_survival(OS_STATS, TD_STATS, return_stats=True)
        KMcurves_ps[i-1,p_iter] = p_val

        if (plow, phigh) in percentiles2plot:
            ax[i-1].text(0.70, 0.85, r"$\chi^2 =$"+f"{round(chisquared,4)}, \np = {pvalue_to_text(p_val)}", transform=ax[i-1].transAxes, 
                        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val<=0.05 else "black")        
        
            # Set the title and labels
            ax[i-1].hlines(0,-5,105, color="black", linewidth=.5)
            ax[i-1].set_ylim([-.2,1])
            ax[i-1].set_xlim([-5,105])
            ax[i-1].set_xticks(range(0,110,10))
            ax[i-1].set_xticklabels(range(0,110,10))
            ax[i-1].set_yticks([0,0.2,0.4,0.6,0.8,1])
            ax[i-1].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
            ax[i-1].spines['left'].set_bounds(0,1)
            ax[i-1].spines['bottom'].set_bounds(0,100)
            ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
            ax[i-1].set_xlabel("Time (months)", fontsize=12)
            ax[i-1].set_ylabel("Overall survival", fontsize=12)
            ax[i-1].spines[["top", "right"]].set_visible(False)
            if i==1:
                ax[i-1].legend(frameon=False)        

    if (plow, phigh) in percentiles2plot:
        fig.tight_layout()
        fig.savefig(os.path.join(args.path, results_folder, f"kaplan-meier/KM-curves_percentiles-{plow}-{phigh}.{args.format}"), dpi=300, format=args.format)
        plt.close()
        print("++++"*40)

### Median OS and 95% CIs
from scipy.signal import wiener
w_size = 10
fig, ax = plt.subplots(1, 2, figsize=(10,4))
colors = [["tab:blue","Blues_r"],["tab:orange","Oranges_r"],["tab:green","Greens_r"],["tab:purple","Purples_r"],["tab:brown","copper"]]

for i in range(1,len(TDMaps.columns)):
    x = np.array([Median_OS[(plow, phigh)][i-1,0] for plow, phigh in percentiles2check]) 
    y = np.array([Median_OS[(plow, phigh)][i-1,3] for plow, phigh in percentiles2check])

    x = wiener(x, w_size)
    y = wiener(y, w_size)
    
    ax[0].plot(x, y, label=TDMaps.columns[i], color=colors[i-1][0], linewidth=.75)
    if i == 1:
        x_end, y_end = x[0], y[0]-2
        x_pre, y_pre = x[-1]-1.5, y[-1]-1
        ax[0].annotate('', xy=(x_end, y_end), xytext=(x_pre, y_pre),
                    arrowprops=dict(arrowstyle='->', color="black", lw=1.5))
        ax[0].text(x_end, y_end, f"{int(percentiles2check[-1][0])}/{int(percentiles2check[-1][1])}", fontsize=6, fontweight='bold')
        ax[0].text(x_pre, y_pre, f"{int(percentiles2check[0][0])}/{int(percentiles2check[0][1])}", fontsize=6, fontweight='bold')

    # TODO --> Add the fill between with the 95 CIs (check lines 614 and 628)
    ax[1].plot(range(0,len(percentiles2check)), x-y, label=TDMaps.columns[i], color=colors[i-1][0])

ax[0].plot([18,23], [18,23], '--', color="black", linewidth=0.75)
ax[0].spines[["top", "right"]].set_visible(False)
ax[0].set_xlabel(f"Median OS in the Low L-TDI group (months)")
ax[0].set_ylabel(f"Median OS in the High L-TDI group (months)")
ax[1].spines[["top", "right"]].set_visible(False)
ax[1].legend(frameon=True, loc='upper right')

# Adding map
all_vals = np.concatenate([x, y])
min_val, max_val = all_vals.min() - 2, all_vals.max() + 2
diag_space = np.linspace(min_val, max_val, 100)
for offset in np.linspace(0, 15, 30): 
    ax[0].fill_between(diag_space, diag_space - offset, diag_space - offset-5, color=plt.cm.Reds(offset/15), alpha=0.05, zorder=0)
intervals = [2, 4, 6, 8, 10, 12] 
for c in intervals:
    # Line: y = x - c
    line_x = np.linspace(min_val + c, max_val, 5)
    line_y = line_x - c
    ax[0].plot(line_x, line_y, color='black', lw=0.5, ls='--', alpha=0.2, zorder=1)
    ax[0].text(line_x[-1]-.5, line_y[-1]-.25, f"+{c}m", fontsize=7, alpha=1, va='top', ha='right')
ax[0].plot([min_val, max_val], [min_val, max_val], '--', color="black", linewidth=1, zorder=2)
ax[0].set_xlim(10, 24)
ax[0].set_ylim(10, 24)
ax[0].set_aspect('equal')

# Create the inset axes --> Probably to be deleted
axins = inset_axes(ax[0], width="40%", height="40%", loc='upper left', borderpad=2)
for i in range(1, len(TDMaps.columns)):
    x_data = np.array([Median_OS[(plow, phigh)][i-1, 0] for plow, phigh in percentiles2check])
    y_data = np.array([Median_OS[(plow, phigh)][i-1, 3] for plow, phigh in percentiles2check])
    
    x_data = wiener(x_data, w_size)
    y_data = wiener(y_data, w_size)

    axins.plot(x_data, y_data, color=colors[i-1][0], linewidth=1.5, alpha=.85)

    if i == 1:
        x_end, y_end = x_data[0]-.75, y_data[0]-2
        x_pre, y_pre = x_data[-1]-1.5, y_data[-1]-1
        axins.annotate('', xy=(x_end, y_end), xytext=(x_pre, y_pre),
                    arrowprops=dict(arrowstyle='->', color="black", lw=1.5))
        axins.text(x_end, y_end, f"{int(percentiles2check[-1][0])}/{int(percentiles2check[-1][1])}", fontsize=6, fontweight='bold')
        axins.text(x_pre, y_pre, f"{int(percentiles2check[0][0])}/{int(percentiles2check[0][1])}", fontsize=6, fontweight='bold')

axins.plot([0, 50], [0, 50], '--', color="black", linewidth=0.75)
for offset in np.linspace(0, 15, 30): 
    axins.fill_between(diag_space, diag_space - offset, diag_space - offset-5, color=plt.cm.Reds(offset/15), alpha=0.05, zorder=0)
axins.set_xlim(18, 24)
axins.set_ylim(10, 18)
axins.tick_params(labelsize=8)
axins.spines[["top", "right"]].set_visible(False)
ax[0].indicate_inset_zoom(axins, edgecolor="black", linewidth=0.5, linestyle="-")

# Create a colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=13)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax[1], pad=0.1, shrink=1, alpha=0.5)
cbar.set_label(r'$\Delta OS$ (months)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax[1].set_xticks([0,len(percentiles2check)-1])
ax[1].set_xticklabels([f"{int(percentiles2check[0][0])}/{int(percentiles2check[0][1])}",f"{int(percentiles2check[-1][0])}/{int(percentiles2check[-1][1])}"])#[f"{int(p[0])}/{int(p[1])}" for p in percentiles2check[::4]])
ax[1].set_xlabel("Stratification percentile (p, 100-p)")
ax[1].set_ylabel(r'$\Delta OS_{Low \ L-TDI, High \ L-TDI}$'+ " (months)")

fig.savefig(os.path.join(args.path, results_folder, f"kaplan-meier/Median-OS.{args.format}"), dpi=300, format=args.format)
plt.close()

####################################################################################################################################################################
## Pvalue inspecting across percentiles
####################################################################################################################################################################
#np.save("./median.npy", Median_ps)
#np.save("./km.npy", KMcurves_ps)
#Median_ps = np.load("./median.npy")
#KMcurves_ps = np.load("./km.npy")
BB_outcomes_median_uncensored = benjamini_bogomolov_procedure(Median_ps, alpha=0.05, global_test='simes', family_method='fdr_bh', inner_method='fdr_bh')   
BB_outcomes_logrank_km = benjamini_bogomolov_procedure(KMcurves_ps, alpha=0.05, global_test='simes', family_method='fdr_bh', inner_method='fdr_bh')   

fig, ax = plt.subplots(1, 2, figsize=(12,3))
alpha = 0.05
x = np.arange(len(TDMaps.columns) - 1)

pvals = BB_outcomes_logrank_km.family_pvals
pvals_adj = BB_outcomes_logrank_km.family_pvals_adj

ax[0].plot(x, np.log10(pvals),'s', markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.5, linestyle='None', label="Uncorrected")
ax[0].plot(x, np.log10(pvals_adj), 'x', color='blue', markeredgewidth=1.5, linestyle='None', label="Corrected")
ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.05), np.log10(0.05)], linestyle="--", color='red', linewidth=0.75)
ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.01), np.log10(0.01)], linestyle="--", color='gray', linewidth=0.5)
ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.001), np.log10(0.001)], linestyle="--", color='gray', linewidth=0.5)
ax[0].plot([0,len(TDMaps.columns)-2], [np.log10(0.0001), np.log10(0.0001)], linestyle="--", color='gray', linewidth=0.5)

ax[0].spines[["top", "right"]].set_visible(False)
ax[0].spines['left'].set_bounds(np.log10(0.1), np.log10(0.00001))
ax[0].spines['bottom'].set_bounds([0,len(TDMaps.columns)-2])
ax[0].set_xticks(range(len(TDMaps.columns)-1))
ax[0].set_xticklabels(TDMaps.columns[1:])
ax[0].set_yticks([np.log10(0.05), np.log10(0.01), np.log10(0.001), np.log10(0.0001)])
ax[0].set_yticklabels(["5*10E-2", "10E-2", "10E-3", "<10E-4"])
ax[0].set_ylabel(r'$p-value \ (\chi^2)$')
ax[0].legend(frameon=True)

pvals = BB_outcomes_median_uncensored.family_pvals
pvals_adj = BB_outcomes_median_uncensored.family_pvals_adj

ax[1].plot(x, np.log10(pvals),'s', markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.5, linestyle='None', label="Uncorrected")
ax[1].plot(x, np.log10(pvals_adj), 'x', color='blue', markeredgewidth=1.5, linestyle='None', label="Corrected")
ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.05), np.log10(0.05)], linestyle="--", color='red', linewidth=0.75)
ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.01), np.log10(0.01)], linestyle="--", color='gray', linewidth=0.5)
ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.001), np.log10(0.001)], linestyle="--", color='gray', linewidth=0.5)
ax[1].plot([0,len(TDMaps.columns)-2], [np.log10(0.0001), np.log10(0.0001)], linestyle="--", color='gray', linewidth=0.5)

ax[1].spines[["top", "right"]].set_visible(False)
ax[1].spines['left'].set_bounds(np.log10(0.1), np.log10(0.000001))
ax[1].spines['bottom'].set_bounds([0,len(TDMaps.columns)-2])
ax[1].set_xticks(range(len(TDMaps.columns)-1))
ax[1].set_xticklabels(TDMaps.columns[1:])
ax[1].set_yticks([np.log10(0.05), np.log10(0.01), np.log10(0.001), np.log10(0.0001)])
ax[1].set_yticklabels(["5*10E-2", "10E-2", "10E-3", "<10E-4"])
ax[1].set_ylabel(r'$p-value \ (U)$')

fig.suptitle("Family-wise combined p-value", fontweight="bold")
fig.savefig(os.path.join(args.path, results_folder, f"KM-Chisquared_Median-Uncensored_family-selection.{args.format}"), dpi=300, format=args.format)
plt.close()

fig_u, ax_u = plt.subplots(nrows, ncols, figsize=figsize)
fig_chi, ax_chi = plt.subplots(nrows, ncols, figsize=figsize)

pvals_u_adj = BB_outcomes_median_uncensored.hypotheses_pvals_adj
pvals_chi_adj = BB_outcomes_logrank_km.hypotheses_pvals_adj

for i in range(1,len(TDMaps.columns)):
    # U-test of median uncensored
    ax_u[i-1].plot(Median_ps[i-1], color="black", label="Uncorrected", linewidth=2)
    ax_u[i-1].plot(pvals_u_adj[i-1], color="blue", label="Corrected", linewidth=.75, alpha=.5)
    ax_u[i-1].axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="p<0.05")
    ax_u[i-1].axhline(y=0.01, color="gray", linestyle="--", linewidth=1)
    ax_u[i-1].axhline(y=0.0001, color="gray", linestyle="--", linewidth=.5)
    ax_u[i-1].set_yscale("log")
    ax_u[i-1].set_xticks([0, len(percentiles2check)-1])
    ax_u[i-1].set_xticklabels([f"{int(percentiles2check[0][0])}/{int(percentiles2check[0][1])}",
                                f"{int(percentiles2check[-1][0])}/{int(percentiles2check[-1][1])}"])
    ax_u[i-1].spines['bottom'].set_bounds(0, len(percentiles2check)-1)
    ax_u[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
    ax_u[i-1].set_xlabel("Stratification percentile", fontsize=12)
    ax_u[i-1].set_ylabel(r'$p-value \ (U)$', fontsize=12)
    ax_u[i-1].spines[["top", "right"]].set_visible(False)
    if i == 1:
        ax_u[i-1].legend(frameon=True)

    # Log-rank test of KM
    ax_chi[i-1].plot(KMcurves_ps[i-1], color="black", label="Uncorrected", linewidth=2)
    ax_chi[i-1].plot(pvals_chi_adj[i-1], color="blue", label="Corrected", linewidth=.75, alpha=.5)
    ax_chi[i-1].axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="p<0.05")
    ax_chi[i-1].axhline(y=0.01, color="gray", linestyle="--", linewidth=1)
    ax_chi[i-1].axhline(y=0.0001, color="gray", linestyle="--", linewidth=.5)
    ax_chi[i-1].set_yscale("log")
    if np.any(np.array(KMcurves_ps[i-1]) > 0.05) or np.any(np.array(pvals_chi_adj[i-1]) > 0.05):
        ax_chi[i-1].axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax_chi[i-1].set_xticks([0,len(percentiles2check)-1])
    ax_chi[i-1].set_xticklabels([f"{int(percentiles2check[0][0])}/{int(percentiles2check[0][1])}",f"{int(percentiles2check[-1][0])}/{int(percentiles2check[-1][1])}"])
    ax_chi[i-1].spines['bottom'].set_bounds(0,len(percentiles2check)-1)
    ax_chi[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
    ax_chi[i-1].set_xlabel("Stratification percentile", fontsize=12)
    ax_chi[i-1].set_ylabel(r'$p-value \ (\chi^2)$', fontsize=12)
    ax_chi[i-1].spines[["top", "right"]].set_visible(False)
    if i==1:
        ax_chi[i-1].legend(frameon=True)

fig_u.tight_layout()
fig_u.savefig(os.path.join(args.path, results_folder, f"median-os_uncensored/BB-corrected_pvals.{args.format}"), dpi=300, format=args.format)
plt.close()
fig_chi.tight_layout()
fig_chi.savefig(os.path.join(args.path, results_folder, f"kaplan-meier/BB-corrected_pvals.{args.format}"), dpi=300, format=args.format)
plt.close()

print("FINISHED SURVIVAL AND KAPLAN-MEIER ANALYSES")
print("   ************************   ")

####################################################################################################################################################################
## Computing C-indices and hazard ratios
####################################################################################################################################################################
os.makedirs(os.path.join(args.path, results_folder, "cox-model"), exist_ok=True)
ps_HR = []
ps_Cs = []
HRs, HR_lows, HR_highs = [], [], []
Cs, C_lows, C_highs = [], [], []
labels = []
TDMaps["status"] = life

for i in range(1,len(TDMaps.columns)-1):
    print("\n"+40*"+"+"\n")
    features = ["OS", "status",TDMaps.columns[i]]
    data = TDMaps[features].copy()
    data.dropna(inplace=True)

    cph = CoxPHFitter(baseline_estimation_method="breslow")
    cph.fit(data, duration_col="OS", event_col="status")
    cindex_train = cph.score(data, scoring_method="concordance_index")

    ll = cph.log_likelihood_
    ci_boot, _ = bootstrap_cindex(cph, data, features, status="status", survival="OS", n_bootstrap=n_resamples, alpha_CI=0.05, seed=None)
    ci_perm, _ = permutation_cindex(cph, data, features, status="status", survival="OS", n_permutations=n_perms, alpha_CI=0.05, seed=None)
    ps_HR.append(cph.summary["p"].values[0])   
    ps_Cs.append(ci_perm[0])                  

    print_model_summary(f"Tissue: {TDMaps.columns[i]} (N={len(data)})", cph, cindex_train, ci_boot, ci_perm)

    HRs.append(np.exp(cph.summary["coef"].values[0]))
    HR_lows.append(np.exp(cph.summary["coef lower 95%"].values[0]))
    HR_highs.append(np.exp(cph.summary["coef upper 95%"].values[0]))
    Cs.append(ci_boot[0])
    C_lows.append(ci_boot[1])
    C_highs.append(ci_boot[2])
    labels.append(TDMaps.columns[i])

print("\n"+10*"+"+" Multiple hypothesis correction "+ 10*"+"+"\n")
_, ps_HR_adj, _, _ = multipletests(ps_HR, alpha=0.05, method='fdr_bh')
_, ps_Cs_adj, _, _ = multipletests(ps_Cs, alpha=0.05, method='fdr_bh')
results_df = pd.DataFrame({
    "Tissue":               TDMaps.columns[1:-1],
    "HR p-val":             ps_HR,
    "Adjusted HR p-val":    ps_HR_adj,
    "C-index p-val":        ps_Cs,
    "Adjusted C-index p-val": ps_Cs_adj,
})
print(results_df.to_string(index=False))

n = len(labels)
y = np.arange(n)
fig, (ax_hr, ax_ci) = plt.subplots(1, 2, figsize=(10, 0.8 * n + 1))
# --- Hazard Ratios ---
ax_hr.axvline(x=1.0, color="gray", linestyle="--", linewidth=1)
ax_hr.errorbar(
    HRs, y,
    xerr=[np.array(HRs) - np.array(HR_lows), np.array(HR_highs) - np.array(HRs)],
    fmt="o", color="black", markerfacecolor="white", markeredgecolor="black",
    markeredgewidth=1.5, markersize=8, capsize=5, linewidth=1.5
)
for k in range(n):
    ax_hr.text(HR_highs[k], y[k]+0.05, f"  {sig_marker(ps_HR_adj[k])}", va="center", ha="left", fontsize=10, color="blue")
ax_hr.set_yticks(y)
ax_hr.set_yticklabels(labels, fontsize=11)
ax_hr.set_xlabel("Hazard Ratios", fontsize=10, fontweight="bold")
ax_hr.spines[["top", "right"]].set_visible(False)
ax_hr.spines['bottom'].set_bounds(1,1.3)
ax_hr.set_xticks([1.00,1.05,1.10,1.15,1.20,1.25,1.30])
ax_hr.invert_yaxis()
# --- C-index ---
ax_ci.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)
ax_ci.errorbar(
    Cs, y,
    xerr=[np.array(Cs) - np.array(C_lows), np.array(C_highs) - np.array(Cs)],
    fmt="o", color="black", markerfacecolor="white", markeredgecolor="black",
    markeredgewidth=1.5, markersize=8, capsize=5, linewidth=1.5
)
for k in range(n):
    ax_ci.text(C_highs[k], y[k]+0.05, f"  {sig_marker(ps_Cs_adj[k])}", va="center", ha="left", fontsize=10, color="blue")
ax_ci.set_yticks(y)
ax_ci.set_yticklabels([], fontsize=11)
ax_ci.set_xlabel("C-indices", fontsize=10, fontweight="bold")
ax_ci.spines[["top", "right"]].set_visible(False)
ax_ci.spines['bottom'].set_bounds(0.5,0.6)
ax_ci.set_xticks([0.5,0.52,0.54,0.56,0.58,0.6])
ax_ci.invert_yaxis()

fig.tight_layout()
fig.savefig(os.path.join(args.path, results_folder, f"cox-model/HRs-Cs_forest.{args.format}"), dpi=300, format=args.format)
plt.close()