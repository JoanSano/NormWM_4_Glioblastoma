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

class DeLong_Test():    
    # Adopted from https://github.com/yandexdataschool/roc_comparison 
    # Original ref: https://ieeexplore.ieee.org/document/6851192

    def __init__(self, ground_truths) -> None:
        """
        Computes the DeLong p-value and/or variance of a pair or predictions
            that are related to the same structure (or ground truth)
        Args:
        ground_truths: A flat array containing the positive and negative samples.
        """
        self.ground_truth = ground_truths

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    def __compute_midrank(self, x):
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    def __compute_ground_truth_statistics(self):
        assert np.array_equal(np.unique(self.ground_truth), [0, 1])
        order = (-self.ground_truth).argsort()
        label_1_count = int(self.ground_truth.sum())
        return order, label_1_count

    def fastDeLong(self, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float64)
        ty = np.empty([k, n], dtype=np.float64)
        tz = np.empty([k, m + n], dtype=np.float64)
        for r in range(k):
            tx[r, :] = self.__compute_midrank(positive_examples[r, :])
            ty[r, :] = self.__compute_midrank(negative_examples[r, :])
            tz[r, :] = self.__compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov
    
    def calc_pvalue(self, aucs, sigma, alternative="two-sided"):
        """Computes the p-value.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        z_score, p_value
        """
        """ l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        print(z)
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) """     
        if alternative not in ["two-sided", "greater", "lower"]:
            raise ValueError("Provide a valid hypothesis from two-sided, greater or lower")
        l = np.array([1, -1])
        z = (aucs[0]-aucs[1]) / np.sqrt(np.dot(np.dot(l, sigma), l.T))   
        if alternative=="two-sided":
            return z, scipy.stats.norm.sf(abs(z))*2
        elif alternative=="greater":
            return z, scipy.stats.norm.sf(z)
        else:
            return z, scipy.stats.norm.cdf(z)

    def delong_roc_variance(self, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    def delong_roc_test(self, predictions_one, predictions_two, alternative="two-sided"):
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        return self.calc_pvalue(aucs, delongcov, alternative=alternative)

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the directory where the TDI and survival data are stored")
parser.add_argument("--threshold", type=int, default=0, help="Streamline density threshold")
parser.add_argument("--correction", type=str, choices=["fwer","fdr"], default="fdr", help="Multiple hypotheses correction method")
parser.add_argument("--format", type=str, default='pdf', choices=['pdf','svg'], help="Output figure format")
args = parser.parse_args()

stream_th = args.threshold
fwer = True if args.correction=="fwer" else False
daysXmonth = 365/12 
percentiles2check = (20,80),(25,75),(30,70),(35,65),(40,60),(45,55),(50,50)
n_resamples = 2500 # Bottstrapping and permutation of correlation values
n_perms = 5000 # Permutation of Cox Prop Hazard models
months = np.array([6,12,18,24,30,36,42,48])

nrows, ncols = 2, 5
figsize = (25,12)
figs_folder = f"StreamlineTDThreshold-{stream_th}_GBM-Wildtype"
os.makedirs(os.path.join(args.path, "Figures/TDMaps_Grade-IV",figs_folder), exist_ok=True)

demographics_TD = pd.read_csv(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/demographics-TDMaps_streamTH-{stream_th}.csv"))
TDMaps_all = demographics_TD[
    [
        "OS",
        "Whole TDMap",
        "Whole lesion TDMap",
        "Core TDMap", 
        "Core lesion TDMap",
        "Non-enhancing TDMap",
        "Non-enhancing lesion TDMap",
        "Enhancing TDMap",
        "Enhancing lesion TDMap",
        "Core+Enhancing TDMap",
        "Core+Enhancing lesion TDMap",
        "1-dead 0-alive"
    ]
]

TDMaps_all = TDMaps_all.loc[demographics_TD["Final pathologic diagnosis (WHO 2021)"]=="Glioblastoma  IDH-wildtype"] 
TDMaps_all = TDMaps_all.loc[demographics_TD["OS"].fillna('unknown')!='unknown']
#TDMaps_all = TDMaps_all.loc[demographics_TD["MGMT status"].isin(["positive", "negative"])]
#TDMaps_all = TDMaps_all.loc[demographics_TD["MGMT index"].fillna('unknown')!='unknown']
#TDMaps_all = TDMaps_all.loc[demographics_TD["1p/19q"].fillna('unknown').isin(["intact", "unknown"])]

TDMaps = TDMaps_all#.loc[demographics_TD["Final pathologic diagnosis (WHO 2021)"]=="Glioblastoma  IDH-wildtype"] 
life = TDMaps_all["1-dead 0-alive"].values
TDMaps = TDMaps.drop(columns="1-dead 0-alive")
"""
####################################################################################################################################################################
## General numbers
####################################################################################################################################################################
stats_string = "Number of samples per each group\n+++++++++++++++++++++++++++++\n"
for i in range(1,len(TDMaps.columns)):
    x = TDMaps[TDMaps.columns[i]]
    y = TDMaps["OS"]
    
    # Remove rows where x or y is NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if i==1:
        stats_string += f"OS: {len(y_clean)} Patients\n"
    stats_string += f"{TDMaps.columns[i]}: {len(x_clean)} Patients\n"

a,b,c = np.count_nonzero(~np.isnan(life)), np.nansum(life), int(np.nansum(np.where(life==0,1,np.nan)))
stats_string += f"No. of patients with a registered event (1-dead/0-alive): {a}\n"
stats_string += f"No. of dead patients (without right censoring): {b} ({round(100*b/a,2)}%)\n"
stats_string += f"No. of patients with right censoring: {c} ({round(100*c/a,2)}%)\n"

with open(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/stats.txt"), "w") as stats_file:
    stats_file.write(stats_string)

####################################################################################################################################################################
## Correlation coefficient between TD Maps and OS
####################################################################################################################################################################
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 18))
cross_TD_0 = np.zeros((len(TDMaps.columns), len(TDMaps.columns))) * np.nan
cross_TD_p_0 = np.zeros((len(TDMaps.columns), len(TDMaps.columns))) * np.nan
cross_TD_1 = np.zeros((len(TDMaps.columns), len(TDMaps.columns))) * np.nan
cross_TD_p_1 = np.zeros((len(TDMaps.columns), len(TDMaps.columns))) * np.nan
for i in range(len(TDMaps.columns)):
    y = TDMaps[TDMaps.columns[i]]
    for j in range(i,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[j]]

        # Remove rows where x or y is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]

        cross_TD_0[i,j], cross_TD_p_0[i,j] = pearsonr(x_clean[life_clean==0], y_clean[life_clean==0], alternative='two-sided')
        cross_TD_1[i,j], cross_TD_p_1[i,j] = pearsonr(x_clean[life_clean==1], y_clean[life_clean==1], alternative='two-sided')
sns.heatmap(cross_TD_1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax1, 
            xticklabels=[], yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
sns.heatmap(cross_TD_0, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax4, 
            xticklabels=TDMaps.columns, yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
ax1.set_title('Pearson Correlation Coefficients (status=1)', fontweight='bold', fontsize=12)
ax4.set_title('Pearson Correlation Coefficients (status=0)', fontweight='bold', fontsize=12)
ax1.tick_params(axis='both', length=0) 
sns.heatmap(np.round(cross_TD_p_1,4), annot=True, cmap='viridis', square=True, ax=ax2, 
            xticklabels=[], yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
sns.heatmap(np.round(cross_TD_p_0,4), annot=True, cmap='viridis', square=True, ax=ax5, 
            xticklabels=TDMaps.columns, yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
ax2.set_title('P-values for Correlation Coefficients (status=1)', fontweight='bold', fontsize=12)
ax5.set_title('P-values for Correlation Coefficients (status=0)', fontweight='bold', fontsize=12)
ax2.tick_params(axis='both', length=0) 
if fwer: # Method: Holm's procedure
    _, cross_TD_p_corrected_flat_1, _, _ = multipletests(cross_TD_p_1[np.triu_indices(len(TDMaps.columns))], alpha=0.05, method='holm', is_sorted=False)
    _, cross_TD_p_corrected_flat_0, _, _ = multipletests(cross_TD_p_0[np.triu_indices(len(TDMaps.columns))], alpha=0.05, method='holm', is_sorted=False)
else: # Method: Benjamin-Hochberg
    _, cross_TD_p_corrected_flat_1 = fdrcorrection(cross_TD_p_1[np.triu_indices(len(TDMaps.columns))], alpha=0.05, method='p', is_sorted=False)
    _, cross_TD_p_corrected_flat_0 = fdrcorrection(cross_TD_p_0[np.triu_indices(len(TDMaps.columns))], alpha=0.05, method='p', is_sorted=False)
cross_TD_p_corrected_1 = cross_TD_p_1 * np.nan
cross_TD_p_corrected_0 = cross_TD_p_0 * np.nan
k = 0
for r, c in zip(*np.triu_indices(len(TDMaps.columns))):
    cross_TD_p_corrected_1[r,c] = cross_TD_p_corrected_flat_1[k]
    cross_TD_p_corrected_0[r,c] = cross_TD_p_corrected_flat_0[k]
    k += 1
sns.heatmap(np.round(cross_TD_p_corrected_1,4), annot=True, cmap='viridis', square=True, ax=ax3, 
            xticklabels=[], yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
sns.heatmap(np.round(cross_TD_p_corrected_0,4), annot=True, cmap='viridis', square=True, ax=ax6, 
            xticklabels=TDMaps.columns, yticklabels=TDMaps.columns, cbar_kws={"shrink": 0.7})
ax3.set_title('FWER Corrected (status=1)' if fwer else "FDR Corrected (status=1)", fontweight='bold', fontsize=12)
ax6.set_title('FWER Corrected (status=1)' if fwer else "FDR Corrected (status=0)", fontweight='bold', fontsize=12)
ax3.tick_params(axis='both', length=0) 
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/correlation-TDMaps.{args.format}"), dpi=300, format=args.format)
plt.close()

####################################################################################################################################################################
## Correlation coefficient between OS and TDMetrics
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
            ptext = cross_TD_p_0[0,i]
            ptextcorr = cross_TD_p_corrected_0[0,i]
        else:
            ptext = cross_TD_p_1[0,i]
            ptextcorr = cross_TD_p_corrected_1[0,i]
        ax[i-1].text(0.55, 0.95, f'R² = {r_squared:.4f} \n'+r'$\rho$'+f' = {r_value:.4f} \n'+r'$p$'+f" = {ptext:.4f} \n"+r'$p_{corrected}$'+f" = {ptextcorr:.4f}", transform=ax[i-1].transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_value<=0.05 else "black")    
        # Set the labels and clean up the plot
        ax[i-1].set_xlabel(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[i-1].set_ylabel("Overall survival (months)", fontweight="bold", fontsize=12)
        ax[i-1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/OS-TDMaps_scatter_status-{status}.{args.format}"), dpi=300, format=args.format)
    plt.close()

####################################################################################################################################################################
## Death analyses
####################################################################################################################################################################
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
    pv_s = np.zeros((len(months),2))
    maxY = 0
    nums = np.zeros((len(months),2))
    for j,m in enumerate(months):
        tdi_alive = x_clean[y_clean>=m*daysXmonth] # Subjects with OS higher than cutoff are alive
        tdi_dead = x_clean[(y_clean<=m*daysXmonth) & (life_clean==1)] # Dead subjects with OS lower than cutoff
        nums[j,:] = [len(tdi_alive), len(tdi_dead)]
        mx = int(max([tdi_alive.max(), tdi_dead.max()]))
        if mx>maxY:
            maxY = mx+1
        ax[i-1].plot(np.full(len(tdi_alive),m-1),tdi_alive,'o', markersize=1.5, color='forestgreen', label=f"Status = 0 (alive)" if j==0 else None)
        ax[i-1].plot(np.full(len(tdi_dead),m+1),tdi_dead,'o', markersize=1.5, color='darkorange', label=f"Status = 1 (dead)" if j==0 else None)
        ax[i-1].plot([m-1,m+1],[np.median(tdi_alive), np.median(tdi_dead)], '-s', linewidth=2.5, markersize=6, color='black')
        ax[i-1].plot([m-1,m-1],[np.median(tdi_alive), np.percentile(tdi_alive, 75)], '-+', linewidth=1.5, markersize=5, color='black')
        ax[i-1].plot([m+1,m+1],[np.median(tdi_dead), np.percentile(tdi_dead, 75)], '-+', linewidth=1.5, color='black')
        _, pv_s[j,0] = mannwhitneyu(tdi_alive, tdi_dead, alternative='two-sided')        
    ax[i-1].text(months[0]-2, -0.01*maxY, "No. of samples", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
    if fwer: # Method: Holm's procedure
        _, pv_s[:,1], _, _ = multipletests(pv_s[:,0], alpha=0.05, method='holm', is_sorted=False)
    else: # Method: Benjamin-Hochberg
        _, pv_s[:,1] = fdrcorrection(pv_s[:,0], alpha=0.05, method='p', is_sorted=False)
    for j,m in enumerate(months):
        ax[i-1].text(m-2, -0.075*maxY, f"{int(nums[j,0])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="forestgreen") # Numbers alive
        ax[i-1].text(m-2, -0.125*maxY, f"{int(nums[j,1])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="darkorange") # Numbers dead
        if pv_s[j,0]<=0.001:
            ax[i-1].text(m-1.5, 1.05*maxY, '***', color='black',fontsize=10)
        elif pv_s[j,0]<=0.01:
            ax[i-1].text(m-1, 1.05*maxY, '**', color='black',fontsize=10)
        elif pv_s[j,0]<=0.05:
            ax[i-1].text(m-.5, 1.05*maxY, '*', color='black',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.05*maxY, 'n.s.', color='black',fontsize=10)
        if pv_s[j,1]<=0.001:
            ax[i-1].text(m-1.5, 1.1*maxY, '***', color='blue',fontsize=10)
        elif pv_s[j,1]<=0.01:
            ax[i-1].text(m-1, 1.1*maxY, '**', color='blue',fontsize=10)
        elif pv_s[j,1]<=0.05:
            ax[i-1].text(m-.5, 1.1*maxY, '*', color='blue',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.1*maxY, 'n.s.', color='blue',fontsize=10)
    ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
    ax[i-1].spines[["top", "right"]].set_visible(False)
    ax[i-1].set_ylabel("TDI (a.u.)", fontsize=12)
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
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-TDMaps_step-monthly.{args.format}"), dpi=300, format=args.format)
plt.close()

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
    pv_s = np.zeros((len(months),2))
    maxY = 0
    nums = np.zeros((len(months),2))
    for j,m in enumerate(months):
        tdi_alive = x_clean[y_clean>=m*daysXmonth] # Subjects with OS higher than cutoff are alive
        tdi_dead = x_clean[(y_clean<=m*daysXmonth) & (life_clean==1)] # Dead subjects with OS lower than cutoff
        nums[j,:] = [len(tdi_alive), len(tdi_dead)]
        mx = int(max([tdi_alive.max(), tdi_dead.max()]))
        if mx>maxY:
            maxY = mx+1
        ax[i-1].plot(np.full(len(tdi_alive),m-1),tdi_alive,'o', markersize=1.5, color='forestgreen', label=f"Status = 0 (alive)" if j==0 else None)
        ax[i-1].plot(np.full(len(tdi_dead),m+1),tdi_dead,'o', markersize=1.5, color='darkorange', label=f"Status = 1 (dead)" if j==0 else None)
        ax[i-1].plot([m-1,m+1],[np.median(tdi_alive), np.median(tdi_dead)], '-s', linewidth=2.5, markersize=6, color='black')
        ax[i-1].plot([m-1,m-1],[np.median(tdi_alive), np.percentile(tdi_alive, 75)], '-+', linewidth=1.5, markersize=5, color='black')
        ax[i-1].plot([m+1,m+1],[np.median(tdi_dead), np.percentile(tdi_dead, 75)], '-+', linewidth=1.5, color='black')
        _, pv_s[j,0] = mannwhitneyu(tdi_alive, tdi_dead, alternative='two-sided')        
    ax[i-1].text(months[0]-2, -0.01*maxY, "No. of samples", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
    if fwer: # Method: Holm's procedure
        _, pv_s[:,1], _, _ = multipletests(pv_s[:,0], alpha=0.05, method='holm', is_sorted=False)
    else: # Method: Benjamin-Hochberg
        _, pv_s[:,1] = fdrcorrection(pv_s[:,0], alpha=0.05, method='p', is_sorted=False)
    for j,m in enumerate(months):
        ax[i-1].text(m-2, -0.075*maxY, f"{int(nums[j,0])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="forestgreen") # Numbers alive
        ax[i-1].text(m-2, -0.125*maxY, f"{int(nums[j,1])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="darkorange") # Numbers dead
        if pv_s[j,0]<=0.001:
            ax[i-1].text(m-1.5, 1.05*maxY, '***', color='black',fontsize=10)
        elif pv_s[j,0]<=0.01:
            ax[i-1].text(m-1, 1.05*maxY, '**', color='black',fontsize=10)
        elif pv_s[j,0]<=0.05:
            ax[i-1].text(m-.5, 1.05*maxY, '*', color='black',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.05*maxY, 'n.s.', color='black',fontsize=10)
        if pv_s[j,1]<=0.001:
            ax[i-1].text(m-1.5, 1.1*maxY, '***', color='blue',fontsize=10)
        elif pv_s[j,1]<=0.01:
            ax[i-1].text(m-1, 1.1*maxY, '**', color='blue',fontsize=10)
        elif pv_s[j,1]<=0.05:
            ax[i-1].text(m-.5, 1.1*maxY, '*', color='blue',fontsize=10)
        else:            
            ax[i-1].text(m-1.5, 1.1*maxY, 'n.s.', color='blue',fontsize=10)
    ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
    ax[i-1].spines[["top", "right"]].set_visible(False)
    ax[i-1].set_ylabel("TDI (a.u.)", fontsize=12)
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
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-TDMaps_step-monthly_status-1.{args.format}"), dpi=300, format=args.format)
plt.close()

####################################################################################################################################################################
## Death, Median survival and Kaplan-Meier analyses
####################################################################################################################################################################
KMcurves_ps = np.zeros((len(TDMaps.columns)-1, len(percentiles2check)))
KMcurves_ps_1 = np.zeros((len(TDMaps.columns)-1, len(percentiles2check)))
Median_ps = np.zeros((len(TDMaps.columns)-1, len(percentiles2check)))
for p_iter, (plow, phigh) in enumerate(percentiles2check):
    print(f"Percentiles ({plow},{phigh})")

    ## Death earlier than X months 
    perc = plow
    for status in [0,1]:
        fig, ax = plt.subplots(nrows*2, ncols, figsize=figsize)
        ax = ax.flatten()
        k_ax = 0 
        for i in range(1,len(TDMaps.columns)):
            if (i==3 or i==4) and status==0:
                print(f"Status = {status} --> {TDMaps.columns[i]} no TDI groups available")
                ax[k_ax].set_axis_off()
                ax[k_ax+5].set_axis_off()
            else:
                x = TDMaps[TDMaps.columns[i]]
                y = TDMaps["OS"]    
                # Remove rows where x or y is NaN
                mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life) & life==status
                x_clean = x[mask]
                y_clean = y[mask]
                life_clean = life[mask]
                ax[k_ax].text(months[0]-2, -0.375, "No. of deaths", transform=ax[k_ax].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
                rs = np.zeros((len(months),5)) # rho, pval, low CI, high CI, FDR/FWER pval
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
                    rs[j,2:4] = result_rho.confidence_interval(0.95, method=BootstrapMethod(n_resamples=n_resamples))
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
                    pv_us[j,0] = pv
                    ax[k_ax+5].plot(np.full(len(y_masked_small),m-1),y_masked_small.values,'o', markersize=2, color='royalblue', label=f"Low TDI (P<={perc})" if j==0 else None)
                    ax[k_ax+5].plot(np.full(len(y_masked_big),m+1),y_masked_big.values,'o', markersize=2, color='salmon', label=f"High TDI (P>={100-perc})" if j==0 else None)
                    if pv<=0.001:
                        ax[k_ax+5].text(m-1.5, 1300, '***', color='black',fontsize=10)
                    elif pv<=0.01:
                        ax[k_ax+5].text(m-1, 1300, '**', color='black',fontsize=10)
                    elif pv<=0.05:
                        ax[k_ax+5].text(m-.5, 1300, '*', color='black',fontsize=10)
                    else:            
                        ax[k_ax+5].text(m-1.5, 1300, 'n.s.', color='black',fontsize=10)
                    ax[k_ax+5].plot([m-1,m+1],[np.median(y_masked_small), np.median(y_masked_big)], '-s', linewidth=3, markersize=5, color='black')
                    ax[k_ax+5].plot([m-1,m-1],[np.median(y_masked_small), np.percentile(y_masked_small, 75)], '-+', linewidth=2, markersize=5, color='black')
                    ax[k_ax+5].plot([m+1,m+1],[np.median(y_masked_big), np.percentile(y_masked_big, 75)], '-+', linewidth=2, color='black')
                    ax[k_ax+5].text(m-2, 2000, f"{len(y_masked_small)}", transform=ax[k_ax+5].transData, fontsize=12, verticalalignment='top', color="royalblue") # Numbers
                    ax[k_ax+5].text(m-2, 1800, f"{len(y_masked_big)}", transform=ax[k_ax+5].transData, fontsize=12, verticalalignment='top', color="salmon") # Numbers
                if fwer: # Method: Holm's procedure
                    _, rs[:,4], _, _ = multipletests(rs[:,1], alpha=0.05, method='holm', is_sorted=False)
                    _, pv_us[:,1], _, _ = multipletests(pv_us[:,0], alpha=0.05, method='holm', is_sorted=False)
                else: # Method: Benjamin-Hochberg
                    _, rs[:,4] = fdrcorrection(rs[:,1], alpha=0.05, method='p', is_sorted=False)
                    _, pv_us[:,1] = fdrcorrection(pv_us[:,0], alpha=0.05, method='p', is_sorted=False)
                for j,m in enumerate(months):
                    if rs[j,4]<=0.001:
                        ax[k_ax].text(m-1.5, .5, '***', color='blue',fontsize=10, transform=ax[k_ax].transData)
                    elif rs[j,4]<=0.01:
                        ax[k_ax].text(m-1, .5, '**', color='blue',fontsize=10, transform=ax[k_ax].transData)
                    elif rs[j,4]<=0.05:
                        ax[k_ax].text(m-.5, .5, '*', color='blue',fontsize=10, transform=ax[k_ax].transData)
                    else:            
                        ax[k_ax].text(m-1.5, .5, 'n.s.', color='blue',fontsize=10, transform=ax[k_ax].transData)
                    if pv_us[j,1]<=0.001:
                        ax[k_ax+5].text(m-1.5, 1400, '***', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
                    elif pv_us[j,1]<=0.01:
                        ax[k_ax+5].text(m-1, 1400, '**', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
                    elif pv_us[j,1]<=0.05:
                        ax[k_ax+5].text(m-.5, 1400, '*', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
                    else:            
                        ax[k_ax+5].text(m-1.5, 1400, 'n.s.', color='blue',fontsize=10, transform=ax[k_ax+5].transData)
                # Set the labels and clean up the plot
                ax[k_ax].plot(months,rs[:,0],'-o', linewidth=3, markersize=15, color='black')
                ax[k_ax].plot(months[rs[:,4]<=0.05],rs[rs[:,4]<=0.05,0],'o', markersize=5, color='red')
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
                ax[k_ax].set_ylabel("Pearson "+r'$\rho$'+f" (status={status})", fontsize=12)
                ax[k_ax].spines[["top", "right", "bottom"]].set_visible(False)
                ax[k_ax+5].set_ylabel("Overall survival (months)", fontsize=12)
                ax[k_ax+5].set_xlabel("Death cutoff ("+r'$\leq$'+"months)", fontsize=12)
                ax[k_ax+5].set_xlim([months[0]-5, months[-1]+5])
                ax[k_ax+5].set_xticks(months)
                ax[k_ax+5].set_xticklabels(months)
                ax[k_ax+5].set_ylim([-10,1600])
                ax[k_ax+5].set_yticks([0,180,360,540,720,900,1080,1260,1440])
                ax[k_ax+5].set_yticklabels(np.array([0,180,360,540,720,900,1080,1260,1440])//daysXmonth)
                ax[k_ax+5].spines['bottom'].set_bounds(months[0], months[-1])
                ax[k_ax+5].spines['left'].set_bounds(0,1440)
                ax[k_ax+5].spines[["top", "right"]].set_visible(False)
            if (i-1)==0:
                ax[k_ax+5].legend(frameon=False, ncols=1, loc='center left')
            if (i-1)==4:
                k_ax += 6
            else:
                k_ax += 1
        fig.tight_layout()
        if status==1:
            fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/OS-TDMaps_death-cutoff_status-{status}_p-{perc}.{args.format}"), dpi=300, format=args.format)
        else:
            fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/OS-TDMaps_dropout-cutoff_status-{status}_p-{perc}.{args.format}"), dpi=300, format=args.format)
        plt.close()

    ## Median survival analyses
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
        ax[i-1].boxplot(
            [psmall[lifesmall==1]/daysXmonth, pbig[lifebig==1]/daysXmonth], 
            tick_labels=[f"Low TDI (P{plow})", f"High TDI (P{phigh})"],
            positions=[1,2],
            widths=[0.4,0.4]
        )
        # Stats
        _, pv = mannwhitneyu(psmall[lifesmall==1], pbig[lifebig==1], alternative='two-sided')
        Median_ps[i-1,p_iter] = pv
        ax[i-1].text(0.35, 0.85, r'$p_U$'+f' = {pv:.4f}', transform=ax[i-1].transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if pv<=0.05 else "black")        
        # Set the title and labels
        ax[i-1].set_xlim([.5,2.5])
        ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[i-1].set_ylabel("Overall survival (months)", fontsize=12)
        ax[i-1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/OS-TDMaps_percentiles-{plow}-{phigh}_status-1.{args.format}"), dpi=300, format=args.format)
    plt.close()

    ## Kaplan-Meier analyses with dead subjects
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]        
        # Remove rows where x, y, or life is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life) & life==1
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        # Calculate the 25th and 75th percentiles
        psmall = y_clean[x_clean<np.percentile(x_clean, plow)]
        pbig = y_clean[x_clean>np.percentile(x_clean, phigh)]
        # Obtain the status of each patient --> True or False indicating whether the entry is right censored (False) or not (True) (All should be True here)
        lifesmall = life_clean[x_clean<np.percentile(x_clean, plow)]==1  
        lifebig = life_clean[x_clean>np.percentile(x_clean, phigh)]==1        
        # Overall survivale for the given community
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifesmall, psmall, conf_type="log-log"
        )
        ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"Low TDI", color="royalblue")
        ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="royalblue")
        for t in psmall[lifesmall==0].values: # Censoring times
            ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='royalblue')        
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifebig, pbig, conf_type="log-log"
        )
        ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"High TDI", color="salmon")
        ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="salmon")
        for t in pbig[lifebig==0].values: # Censoring times
            ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='salmon')
        # Numbers
        ax[i-1].text(-2, -0.025, "No. at risk", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
        for t in [0,10,20,30,40,50,60,70]:
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
        KMcurves_ps_1[i-1,p_iter] = p_val
        ax[i-1].text(0.70, 0.85, r"$\chi^2 =$"+f"{round(chisquared,4)}, \np = {round(p_val,4)}", transform=ax[i-1].transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val<=0.05 else "black")        
        # Set the title and labels
        ax[i-1].hlines(0,-5,75, color="black", linewidth=.5)
        ax[i-1].set_ylim([-.2,1])
        ax[i-1].set_xlim([-5,75])
        ax[i-1].set_xticks(range(0,80,10))
        ax[i-1].set_xticklabels(range(0,80,10))
        ax[i-1].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[i-1].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
        ax[i-1].spines['left'].set_bounds(0,1)
        ax[i-1].spines['bottom'].set_bounds(0,70)
        ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[i-1].set_xlabel("Time (months)", fontsize=12)
        ax[i-1].set_ylabel("Overall survival", fontsize=12)
        ax[i-1].spines[["top", "right"]].set_visible(False)
        if i==1:
            ax[i-1].legend(frameon=False)        
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/KM-curves_percentiles-{plow}-{phigh}_status-1.{args.format}"), dpi=300, format=args.format)
    plt.close()

    ## Kaplan-Meier analyses with censoring
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
        # Overall survivale for the given community
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifesmall, psmall, conf_type="log-log"
        )
        ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"Low TDI", color="royalblue")
        ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="royalblue")
        for t in psmall[lifesmall==0].values: # Censoring times
            ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='royalblue')        
        time, survival_prob, conf_int = kaplan_meier_estimator(
            lifebig, pbig, conf_type="log-log"
        )
        ax[i-1].step(time/daysXmonth, survival_prob, where="post", label=f"High TDI", color="salmon")
        ax[i-1].fill_between(time/daysXmonth, conf_int[0], conf_int[1], alpha=0.15, step="post", color="salmon")
        for t in pbig[lifebig==0].values: # Censoring times
            ax[i-1].plot(time[time==t]/daysXmonth, survival_prob[time==t], "|", color='salmon')
        # Numbers
        ax[i-1].text(-2, -0.025, "No. at risk", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
        for t in [0,10,20,30,40,50,60,70]:
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
        ax[i-1].text(0.70, 0.85, r"$\chi^2 =$"+f"{round(chisquared,4)}, \np = {round(p_val,4)}", transform=ax[i-1].transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val<=0.05 else "black")        
        # Set the title and labels
        ax[i-1].hlines(0,-5,75, color="black", linewidth=.5)
        ax[i-1].set_ylim([-.2,1])
        ax[i-1].set_xlim([-5,75])
        ax[i-1].set_xticks(range(0,80,10))
        ax[i-1].set_xticklabels(range(0,80,10))
        ax[i-1].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[i-1].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
        ax[i-1].spines['left'].set_bounds(0,1)
        ax[i-1].spines['bottom'].set_bounds(0,70)
        ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax[i-1].set_xlabel("Time (months)", fontsize=12)
        ax[i-1].set_ylabel("Overall survival", fontsize=12)
        ax[i-1].spines[["top", "right"]].set_visible(False)
        if i==1:
            ax[i-1].legend(frameon=False)        
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/KM-curves_percentiles-{plow}-{phigh}.{args.format}"), dpi=300, format=args.format)
    plt.close()

####################################################################################################################################################################
## Pvalue inspecting across percentiles
####################################################################################################################################################################
fig, ax = plt.subplots(len(TDMaps.columns)-1, 3, figsize=(19,12))
for i in range(len(TDMaps.columns)-1):
    ax[i,0].plot(np.ones((len(percentiles2check),))*np.log(0.05), '--', color='red', linewidth=0.5)
    ax[i,0].plot(np.log(KMcurves_ps[i]), '-o', color='black', linewidth=2, label="Uncorrected")
    ax[i,1].plot(np.ones((len(percentiles2check),))*np.log(0.05), '--', color='red', linewidth=0.5)
    ax[i,1].plot(np.log(KMcurves_ps_1[i]), '-o', color='black', linewidth=2)
    ax[i,2].plot(np.ones((len(percentiles2check),))*np.log(0.05), '--', color='red', linewidth=0.5)
    ax[i,2].plot(np.log(Median_ps[i]), '-o', color='black', linewidth=2)
    if fwer: # Method: Holm's procedure
        _, pcorr_0, _, _ = multipletests(KMcurves_ps[i], alpha=0.05, method='holm', is_sorted=False)
        _, pcorr_1, _, _ = multipletests(KMcurves_ps_1[i], alpha=0.05, method='holm', is_sorted=False)
        _, pcorr_2, _, _ = multipletests(Median_ps[i], alpha=0.05, method='holm', is_sorted=False)
    else: # Method: Benjamin-Hochberg
        _, pcorr_0 = fdrcorrection(KMcurves_ps[i], alpha=0.05, method='p', is_sorted=False)
        _, pcorr_1 = fdrcorrection(KMcurves_ps_1[i], alpha=0.05, method='p', is_sorted=False)
        _, pcorr_2 = fdrcorrection(Median_ps[i], alpha=0.05, method='p', is_sorted=False)
    ax[i,0].plot(np.log(pcorr_0), '-o', color='blue', linewidth=1.25, label="FWER corrected" if fwer else "FDR corrected")
    ax[i,1].plot(np.log(pcorr_1), '-o', color='blue', linewidth=1.25)
    ax[i,2].plot(np.log(pcorr_2), '-o', color='blue', linewidth=1.25)

    ax[i,0].set_xlim([-0.1,len(percentiles2check)-0.9])
    ax[i,1].set_xlim([-0.1,len(percentiles2check)-0.9])
    ax[i,2].set_xlim([-0.1,len(percentiles2check)-0.9])
    ax[i,0].set_ylim(np.log([0.00005,2]))
    ax[i,1].set_ylim(np.log([0.00005,2]))
    ax[i,2].set_ylim(np.log([0.00005,2]))

    ax[i,0].set_yticks(np.round(np.log([0.001,0.01,0.1,1]),2))
    ax[i,0].set_yticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'])
    ax[i,1].set_yticks(np.round(np.log([0.001,0.01,0.1,1]),2))
    ax[i,1].set_yticklabels([])
    ax[i,2].set_yticks(np.round(np.log([0.001,0.01,0.1,1]),2))
    ax[i,2].set_yticklabels([])
    ax[i,0].set_ylabel(TDMaps.columns[i+1][:-5], fontweight='bold', fontsize=5)

    if i==0:
        ax[i,0].set_title("Kaplan-Meier p values", fontweight='bold')
        ax[i,1].set_title("Kaplan-Meier p values (status=1)", fontweight='bold')
        ax[i,2].set_title("Median OS p values (status=1)", fontweight='bold')
    if i==len(TDMaps.columns)-2:
        ax[i,0].legend(frameon=False,ncol=2)

    if i!=len(TDMaps.columns)-2:
        ax[i,0].spines[["top","right","bottom"]].set_visible(False)
        ax[i,1].spines[["top","right","bottom","left"]].set_visible(False)
        ax[i,2].spines[["top","right","bottom","left"]].set_visible(False)
        ax[i,0].set_xticks([])
        ax[i,1].set_xticks([])
        ax[i,2].set_xticks([])
    else:
        ax[i,0].spines[["top","right"]].set_visible(False)
        ax[i,1].spines[["top","right","left"]].set_visible(False)
        ax[i,2].spines[["top","right","left"]].set_visible(False)
        ax[i,0].set_xticks(range(len(percentiles2check)))
        ax[i,1].set_xticks(range(len(percentiles2check)))
        ax[i,2].set_xticks(range(len(percentiles2check)))
        ax[i,0].set_xticklabels([f"{plow}/{phigh}" for plow, phigh in percentiles2check])
        ax[i,1].set_xticklabels([f"{plow}/{phigh}" for plow, phigh in percentiles2check])
        ax[i,2].set_xticklabels([f"{plow}/{phigh}" for plow, phigh in percentiles2check])
        ax[i,0].set_xlabel("Percentiles", fontweight='bold')
        ax[i,1].set_xlabel("Percentiles", fontweight='bold')
        ax[i,2].set_xlabel("Percentiles", fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/p-values_percentiles.{args.format}"), dpi=300, format=args.format)
plt.close()
print("FINISHED SURVIVAL AND KAPLAN-MEIER ANALYSES")
print("   ************************   ")

####################################################################################################################################################################
## FEATURE SELECTION USING INDIVIDUAL COX PROPORTIONAL HAZARD MODELS
####################################################################################################################################################################
# With right censoring
fig, ax = plt.subplots(1,1, figsize=(6,4))
level_CI = 95
HarrellCindex = np.zeros((len(TDMaps.columns)-1,))
HarrellCindex_p = np.zeros((len(TDMaps.columns)-1,))
colors = []
Cmodel_pred = []
stats_string = "Right censoring\n===========================\n"
for i in range(1,len(TDMaps.columns)):
    x = TDMaps[TDMaps.columns[i]]
    y = TDMaps["OS"]    
    # Remove rows where x, y, or life is NaN
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
    x_clean = x[mask]
    y_clean = y[mask]
    life_clean = life[mask]
    # Censoring (based on the status variable)
    OS_STATS = []
    OS_STATS.extend([(st, os) for st,os in zip(life_clean,y_clean.values)])
    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])
    # Model
    Cmodel = CoxPHSurvivalAnalysis(n_iter=200)
    Cmodel.fit(x_clean.values.reshape(-1, 1),OS_STATS)
    HarrellCindex[i-1] = Cmodel.score(x_clean.values.reshape(-1, 1),OS_STATS)
    pop = []
    for _ in tqdm(range(n_perms), desc=f"Cox feature selection for the marker: {TDMaps.columns[i]}"):
        perm_OS_STATS = np.random.permutation(OS_STATS)
        p_Cmodel = CoxPHSurvivalAnalysis()
        p_Cmodel.fit(x_clean.values.reshape(-1, 1),perm_OS_STATS)
        pop.append(p_Cmodel.score(x_clean.values.reshape(-1, 1),perm_OS_STATS))
    HarrellCindex_p[i-1] = np.mean(np.array(pop) >= HarrellCindex[i-1])
    if HarrellCindex_p[i-1]<=0.001:
        colors.append("cornflowerblue")
    elif HarrellCindex_p[i-1]<=0.01:
        colors.append("cornflowerblue")
    elif HarrellCindex_p[i-1]<=0.05:
        colors.append("cornflowerblue")
    else:            
        colors.append("lightgray")
    # Summary
    stats_string += f"Feature: {TDMaps.columns[i]} with a c-index of {HarrellCindex[i-1]} (p={HarrellCindex_p[i-1]})\n"
    # Prediction
    x_new, low_p = {}, 0
    for pc in percentiles2check:
        x_new[f"P({low_p},{pc[0]})"] = x_clean[(x_clean>np.percentile(x_clean, low_p))&(x_clean<=np.percentile(x_clean, pc[0]))].mean()
        x_new[f"P({pc[1]},{100-low_p})"] = x_clean[(x_clean>np.percentile(x_clean, pc[1]))&(x_clean<=np.percentile(x_clean, 100-low_p))].mean()
        low_p = pc[0]
    x_new = pd.DataFrame.from_dict(
            x_new,
            orient="index",
    )
    Cmodel_pred.append(
        Cmodel.predict_survival_function(x_new)
    )
ax.bar(
    range(0,len(TDMaps.columns)-1), 
    np.sort(HarrellCindex)[::-1],
    edgecolor="black",
    color=[colors[ii] for ii in np.argsort(HarrellCindex)[::-1]]
)
for ii, pval in enumerate(HarrellCindex_p[np.argsort(HarrellCindex)[::-1]]):
    if pval<=0.001:
        ax.text(ii-.25, .65, '***', color='black',fontsize=10, transform=ax.transData)
    elif pval<=0.01:
        ax.text(ii-.25, .65, '**', color='black',fontsize=10, transform=ax.transData)
    elif pval<=0.05:
        ax.text(ii-.25, .65, '*', color='black',fontsize=10, transform=ax.transData)
    else:            
        ax.text(ii-.25, .65, 'n.s.', color='black',fontsize=10, transform=ax.transData)
ax.hlines(0.5, -1, len(TDMaps.columns)-1, color='black', linewidth=.75, linestyle='--')
ax.spines[["top","right"]].set_visible(False)
ax.set_ylabel("Harrell's C-index")
ax.set_ylim([0,0.7])
ax.set_xlim([-1,len(TDMaps.columns)])
ax.set_xticks(range(0,len(TDMaps.columns)-1))
ax.set_xticklabels([TDMaps.columns[ii+1] for ii in np.argsort(HarrellCindex)[::-1]], rotation=75)
ax.spines['bottom'].set_bounds(0,len(TDMaps.columns)-2)
ax.spines['left'].set_bounds(0,.7)
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_features-selection.{args.format}"), dpi=300, format=args.format)
plt.close()

# Plot the prediction
start_color, end_color = np.array(to_rgba("royalblue")), np.array(to_rgba("salmon"))
colors = [
    start_color * (1 - cstep) + end_color * cstep
    for cstep in np.linspace(0, 1, len(percentiles2check)*2)
]
cmap = mcolors.LinearSegmentedColormap.from_list("salmon_royalblue", ["salmon", "royalblue"])
norm = plt.Normalize(vmin=0, vmax=1)  # Scale from 0 to 1
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for ScalarMappable
fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
ax = ax.flatten()
time = np.arange(0, np.nanmax(TDMaps["OS"].values)) 
for i in range(0,len(TDMaps.columns)-1):
    k_color = 0
    for j in range(0,len(Cmodel_pred[i]),2):
        survival_pred = Cmodel_pred[i][j]
        ax[i].step(time/daysXmonth, survival_pred(time), where="post", color=colors[k_color])
        survival_pred = Cmodel_pred[i][j+1]
        ax[i].step(time/daysXmonth, survival_pred(time), where="post", color=colors[-1-k_color])
        k_color += 1
    if i==0:
        cbar_ax = inset_axes(ax[i], width="5%", height="15%", loc="upper right", borderpad=2)  
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([r"TDI $P_{(80-100)}$", r"TDI $P_{(0-20)}$"], fontsize=15)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left') 
    ax[i].text(
        0, 
        0.05, 
        r'$C^{H}$'+f" = {round(HarrellCindex[i],4)}\n"+r'$p_{perm}$'+f" = {round(HarrellCindex_p[i],4)}", 
        color='black' if HarrellCindex_p[i]>0.05 else 'red',
        fontsize=12, 
        transform=ax[i].transData
    )
    ax[i].set_ylim([0,1])
    ax[i].set_xlim([-5,75])
    ax[i].set_xticks(range(0,80,10))
    ax[i].set_xticklabels(range(0,80,10))
    ax[i].set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax[i].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
    ax[i].spines['left'].set_bounds(0,1)
    ax[i].spines['bottom'].set_bounds(0,70)
    ax[i].set_title(TDMaps.columns[i+1], fontweight="bold", fontsize=12)
    ax[i].set_xlabel("Time (months)", fontsize=12)
    ax[i].set_ylabel("Overall survival", fontsize=12)
    ax[i].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_features-prediction.{args.format}"), dpi=300, format=args.format)
plt.close()

stats_string += "------------------------------\n"

# Without right censoring
fig, ax = plt.subplots(1,1, figsize=(6,4))
level_CI = 95
HarrellCindex = np.zeros((len(TDMaps.columns)-1,))
HarrellCindex_p = np.zeros((len(TDMaps.columns)-1,))
colors = []
Cmodel_pred = []
stats_string += "No censoring\n===========================\n"
for i in range(1,len(TDMaps.columns)):
    x = TDMaps[TDMaps.columns[i]]
    y = TDMaps["OS"]    
    # Remove rows where x, y, or life is NaN
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life) & life==1
    x_clean = x[mask]
    y_clean = y[mask]
    life_clean = life[mask]
    # Censoring (based on the status variable)
    OS_STATS = []
    OS_STATS.extend([(st, os) for st,os in zip(life_clean,y_clean.values)])
    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])
    # Model
    Cmodel = CoxPHSurvivalAnalysis(n_iter=200)
    Cmodel.fit(x_clean.values.reshape(-1, 1),OS_STATS)
    HarrellCindex[i-1] = Cmodel.score(x_clean.values.reshape(-1, 1),OS_STATS)
    pop = []
    for _ in tqdm(range(n_perms), desc=f"Cox feature selection for the marker (status = 1): {TDMaps.columns[i]}"):
        perm_OS_STATS = np.random.permutation(OS_STATS)
        p_Cmodel = CoxPHSurvivalAnalysis()
        p_Cmodel.fit(x_clean.values.reshape(-1, 1),perm_OS_STATS)
        pop.append(p_Cmodel.score(x_clean.values.reshape(-1, 1),perm_OS_STATS))
    HarrellCindex_p[i-1] = np.mean(np.array(pop) >= HarrellCindex[i-1])
    if HarrellCindex_p[i-1]<=0.001:
        colors.append("cornflowerblue")
    elif HarrellCindex_p[i-1]<=0.01:
        colors.append("cornflowerblue")
    elif HarrellCindex_p[i-1]<=0.05:
        colors.append("cornflowerblue")
    else:            
        colors.append("lightgray")
    # Summary
    stats_string += f"Feature: {TDMaps.columns[i]} with a c-index of {HarrellCindex[i-1]} (p={HarrellCindex_p[i-1]})\n"
    # Prediction
    x_new, low_p = {}, 0
    for pc in percentiles2check:
        x_new[f"P({low_p},{pc[0]})"] = x_clean[(x_clean>np.percentile(x_clean, low_p))&(x_clean<=np.percentile(x_clean, pc[0]))].mean()
        x_new[f"P({pc[1]},{100-low_p})"] = x_clean[(x_clean>np.percentile(x_clean, pc[1]))&(x_clean<=np.percentile(x_clean, 100-low_p))].mean()
        low_p = pc[0]
    x_new = pd.DataFrame.from_dict(
            x_new,
            orient="index",
    )
    Cmodel_pred.append(
        Cmodel.predict_survival_function(x_new)
    )
ax.bar(
    range(0,len(TDMaps.columns)-1), 
    np.sort(HarrellCindex)[::-1],
    edgecolor="black",
    color=[colors[ii] for ii in np.argsort(HarrellCindex)[::-1]]
)
for ii, pval in enumerate(HarrellCindex_p[np.argsort(HarrellCindex)[::-1]]):
    if pval<=0.001:
        ax.text(ii-.25, .65, '***', color='black',fontsize=10, transform=ax.transData)
    elif pval<=0.01:
        ax.text(ii-.25, .65, '**', color='black',fontsize=10, transform=ax.transData)
    elif pval<=0.05:
        ax.text(ii-.25, .65, '*', color='black',fontsize=10, transform=ax.transData)
    else:            
        ax.text(ii-.25, .65, 'n.s.', color='black',fontsize=10, transform=ax.transData)
ax.hlines(0.5, -1, len(TDMaps.columns)-1, color='black', linewidth=.75, linestyle='--')
ax.spines[["top","right"]].set_visible(False)
ax.set_ylabel("Harrell's C-index")
ax.set_ylim([0,0.7])
ax.set_xlim([-1,len(TDMaps.columns)])
ax.set_xticks(range(0,len(TDMaps.columns)-1))
ax.set_xticklabels([TDMaps.columns[ii+1] for ii in np.argsort(HarrellCindex)[::-1]], rotation=75)
ax.spines['bottom'].set_bounds(0,len(TDMaps.columns)-2)
ax.spines['left'].set_bounds(0,.7)
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_features-selection_status-1.{args.format}"), dpi=300, format=args.format)
plt.close()
# Plot the prediction
start_color, end_color = np.array(to_rgba("royalblue")), np.array(to_rgba("salmon"))
colors = [
    start_color * (1 - cstep) + end_color * cstep
    for cstep in np.linspace(0, 1, len(percentiles2check)*2)
]
cmap = mcolors.LinearSegmentedColormap.from_list("salmon_royalblue", ["salmon", "royalblue"])
norm = plt.Normalize(vmin=0, vmax=1)  # Scale from 0 to 1
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for ScalarMappable
fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
ax = ax.flatten()
time = np.arange(0, np.nanmax(TDMaps["OS"].values)) 
for i in range(0,len(TDMaps.columns)-1):
    k_color = 0
    for j in range(0,len(Cmodel_pred[i]),2):
        survival_pred = Cmodel_pred[i][j]
        ax[i].step(time/daysXmonth, survival_pred(time), where="post", color=colors[k_color])
        survival_pred = Cmodel_pred[i][j+1]
        ax[i].step(time/daysXmonth, survival_pred(time), where="post", color=colors[-1-k_color])
        k_color += 1
    if i==0:
        cbar_ax = inset_axes(ax[i], width="5%", height="15%", loc="upper right", borderpad=2)  
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([r"TDI $P_{(0-20)}$", r"TDI $P_{(80-100)}$"], fontsize=15)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left') 
    ax[i].text(
        0, 
        0.05, 
        r'$C^{H}$'+f" = {round(HarrellCindex[i],4)}\n"+r'$p_{perm}$'+f" = {round(HarrellCindex_p[i],4)}", 
        color='black' if HarrellCindex_p[i]>0.05 else 'red',
        fontsize=12, 
        transform=ax[i].transData
    )
    ax[i].set_ylim([0,1])
    ax[i].set_xlim([-5,75])
    ax[i].set_xticks(range(0,80,10))
    ax[i].set_xticklabels(range(0,80,10))
    ax[i].set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax[i].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
    ax[i].spines['left'].set_bounds(0,1)
    ax[i].spines['bottom'].set_bounds(0,70)
    ax[i].set_title(TDMaps.columns[i+1], fontweight="bold", fontsize=12)
    ax[i].set_xlabel("Time (months)", fontsize=12)
    ax[i].set_ylabel("Overall survival", fontsize=12)
    ax[i].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_features-prediction_status-1.{args.format}"), dpi=300, format=args.format)
plt.close()

with open(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/stats-featuresTDI_CoxPHazard.txt"), "w") as stats_file:
    stats_file.write(stats_string)

print("FINISHED FEATURE IMPORTANCE USING COX PH MODELS AND C-index")
print("   ************************   ")

####################################################################################################################################################################
## FEATURE SELECTION USING COX PROPORTIONAL HAZARD MODELS and THE K FIRST/BEST FEATURES SORTED ABOVE
####################################################################################################################################################################
# Features, except the Core labels due to nans which makes the workflow difficult to automatize. Since they do not seem to contribute much, we discard them before hand.
def Harrell_C_index(X, y):
    n_features = X.shape[1]
    Cscores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        Cscores[j] = m.score(Xj, y)
    return Cscores

pipe = Pipeline(
    [
        ("select", SelectKBest(Harrell_C_index)),
        ("model", CoxPHSurvivalAnalysis()),
    ]
)
splits = range(2,11) # 10
runs = 50 # 20

# Include right censored data and discard possible nan values
#feature_labels = [TDMaps.columns[ii+1] for ii in np.argsort(HarrellCindex)[::-1]][:-2] 
feature_labels = [ # Manual sorting for now, based on the C index
    "Whole lesion TDMap",
    "Non-enhancing lesion TDMap",
    "Enhancing lesion TDMap",
    "Core+Enhancing lesion TDMap",
    "Enhancing TDMap",
    "Whole TDMap",
    "Core+Enhancing TDMap",
    "Non-enhancing TDMap",
    #"Core lesion TDMap",
    #"Core TDMap", 
]
TDMaps_final = TDMaps[["OS"]+feature_labels]
mask = TDMaps_final[TDMaps_final.columns].notna().all(axis=1) & ~np.isnan(life)
TDMaps_filtered = TDMaps_final.loc[mask]
life_filtered = life[mask]
features = TDMaps_filtered[TDMaps_filtered.columns[1:]].values
OS_STATS = []
OS_STATS.extend([(st, os) for st,os in zip(life_filtered,TDMaps_filtered["OS"].values)])
OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])

param_grid = {"select__k": np.arange(1, len(TDMaps_filtered.columns))}
mean_test = np.zeros((runs, len(splits)))
k_best_results = np.zeros((runs, len(splits)))
fig, ax = plt.subplots(1,3, figsize=(12,6))
for tt in range(runs):
    print(f"Right censoring: Select K features workflow; run No. {tt+1}")
    for i, spl in enumerate(splits):
        cv = KFold(n_splits=spl, shuffle=True, random_state=None) # Assign a given random state if you want to ensure reproducibility
        gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
        gcv.fit(features, OS_STATS)        
        results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)
        mean_test[tt,i] = results["mean_test_score"].values.max()
        k_best_results[tt,i] = results["param_select__k"].values[0]
        ax[0].plot(spl, mean_test[tt,i], 'o', color='black', alpha=0.25, markersize=5)

best_split = splits[mean_test.mean(axis=0).argmax()]-2
k_feat, feature_counts = np.unique(k_best_results[:,best_split], return_counts=True)
ax[0].errorbar(splits, mean_test.mean(axis=0), yerr=mean_test.std(axis=0)/np.sqrt(splits), linestyle='-', color='blue', linewidth=1, fmt='x',markersize=10, capsize=10)
rect = patches.Rectangle((best_split+1.75, mean_test.mean(axis=0).max()-.0125), 0.5, 0.025, linewidth=2, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
ax[0].spines[["top","right"]].set_visible(False)
ax[0].set_xticks(splits)
ax[0].set_xticklabels(splits)
ax[0].set_xlim([splits[0]-.5, splits[-1]+.5])
ax[0].set_xlabel("No. of splits")
ax[0].set_ylabel("Average Harrell's C index")
ax[0].spines['bottom'].set_bounds(splits[0], splits[-1])
ax[1].bar(
    k_feat, 
    feature_counts,
    edgecolor="black",
    color="gray",
    width=.35
)
ax[1].spines[["top","right"]].set_visible(False)
ax[1].set_xlim([.5,features.shape[-1]+.5])
ax[1].set_xticks(range(1,features.shape[-1]+1))
ax[1].set_xticklabels([f"k={ii}" for ii in range(1,features.shape[-1]+1)])
ax[1].set_ylabel("No. of ocurrences")
ax[1].set_xlabel("No. of selected TD Maps")
ax[1].spines['bottom'].set_bounds(1,features.shape[-1])
ax[1].set_ylim([0, max(feature_counts)+.5])
ax[1].set_yticks(range(0,max(feature_counts)+1))
ax[1].set_yticklabels(range(0,max(feature_counts)+1))
feature_participation = np.zeros((len(feature_labels),)) # Same order as in labels
for i in range(len(k_feat)):
    k_fs = int(k_feat[i])
    for j in range(k_fs):
        feature_participation[j] += feature_counts[i]

ax[2].bar(
    range(1,features.shape[-1]+1), 
    100*feature_participation/runs,
    edgecolor="black",
    color="gray",
    width=.35
)
ax[2].spines[["top","right"]].set_visible(False)
ax[2].set_xlim([.5,features.shape[-1]+.5])
ax[2].set_xticks(range(1,features.shape[-1]+1))
ax[2].set_xticklabels(feature_labels, rotation=75)
ax[2].set_ylabel("Percentage of participation (%)")
ax[2].spines['bottom'].set_bounds(1,features.shape[-1])
ax[2].set_ylim([0, 100])
ax[2].set_yticks([0,20,40,60,80,100])
ax[2].set_yticklabels([0,20,40,60,80,100])
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_feature-importance.{args.format}"), dpi=300, format=args.format)

# Discard right censored data and possible nan values
#feature_labels = [TDMaps.columns[ii+1] for ii in np.argsort(HarrellCindex)[::-1]][:-2] 
feature_labels = [ # Manual sorting for now, based on the C index
    "Whole lesion TDMap",
    "Non-enhancing lesion TDMap",
    "Enhancing lesion TDMap",
    "Core+Enhancing lesion TDMap",
    "Whole TDMap",
    "Core lesion TDMap",
    "Core TDMap", 
    "Non-enhancing TDMap",
    #"Core+Enhancing TDMap",
    #"Enhancing TDMap",
]
TDMaps_final = TDMaps[["OS"]+feature_labels]
mask = TDMaps_final[TDMaps_final.columns].notna().all(axis=1) & ~np.isnan(life) & life==1
TDMaps_filtered = TDMaps_final.loc[mask]
life_filtered = life[mask]
features = TDMaps_filtered[TDMaps_filtered.columns[1:]].values
OS_STATS = []
OS_STATS.extend([(st, os) for st,os in zip(life_filtered,TDMaps_filtered["OS"].values)])
OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])

param_grid = {"select__k": np.arange(1, len(TDMaps_filtered.columns))}
mean_test = np.zeros((runs, len(splits)))
k_best_results = np.zeros((runs, len(splits)))
fig, ax = plt.subplots(1,3, figsize=(12,6))
for tt in range(runs):
    print(f"Without censoring: Select K features workflow; run No. {tt+1}")
    for i, spl in enumerate(splits):
        cv = KFold(n_splits=spl, shuffle=True, random_state=None) # Assign a given random state if you want to ensure reproducibility
        gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
        gcv.fit(features, OS_STATS)        
        results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)
        mean_test[tt,i] = results["mean_test_score"].values.max()
        k_best_results[tt,i] = results["param_select__k"].values[0]
        ax[0].plot(spl, mean_test[tt,i], 'o', color='black', alpha=0.25, markersize=5)

best_split = splits[mean_test.mean(axis=0).argmax()]-2
k_feat, feature_counts = np.unique(k_best_results[:,best_split], return_counts=True)
ax[0].errorbar(splits, mean_test.mean(axis=0), yerr=mean_test.std(axis=0)/np.sqrt(splits), linestyle='-', color='blue', linewidth=1, fmt='x',markersize=10, capsize=10)
rect = patches.Rectangle((best_split+1.75, mean_test.mean(axis=0).max()-.0125), 0.5, 0.025, linewidth=2, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
ax[0].spines[["top","right"]].set_visible(False)
ax[0].set_xticks(splits)
ax[0].set_xticklabels(splits)
ax[0].set_xlim([splits[0]-.5, splits[-1]+.5])
ax[0].set_xlabel("No. of splits")
ax[0].set_ylabel("Average Harrell's C index")
ax[0].spines['bottom'].set_bounds(splits[0], splits[-1])
ax[1].bar(
    k_feat, 
    feature_counts,
    edgecolor="black",
    color="gray",
    width=.35
)
ax[1].spines[["top","right"]].set_visible(False)
ax[1].set_xlim([.5,features.shape[-1]+.5])
ax[1].set_xticks(range(1,features.shape[-1]+1))
ax[1].set_xticklabels([f"k={ii}" for ii in range(1,features.shape[-1]+1)])
ax[1].set_ylabel("No. of ocurrences")
ax[1].set_xlabel("No. of selected TD Maps")
ax[1].spines['bottom'].set_bounds(1,features.shape[-1])
ax[1].set_ylim([0, max(feature_counts)+.5])
ax[1].set_yticks(range(0,max(feature_counts)+1))
ax[1].set_yticklabels(range(0,max(feature_counts)+1))

feature_participation = np.zeros((len(feature_labels),)) # Same order as in labels
for i in range(len(k_feat)):
    k_fs = int(k_feat[i])
    for j in range(k_fs):
        feature_participation[j] += feature_counts[i]
ax[2].bar(
    range(1,features.shape[-1]+1), 
    100*feature_participation/runs,
    edgecolor="black",
    color="gray",
    width=.35
)
ax[2].spines[["top","right"]].set_visible(False)
ax[2].set_xlim([.5,features.shape[-1]+.5])
ax[2].set_xticks(range(1,features.shape[-1]+1))
ax[2].set_xticklabels(feature_labels, rotation=75)
ax[2].set_ylabel("Percentage of participation (%)")
ax[2].spines['bottom'].set_bounds(1,features.shape[-1])
ax[2].set_ylim([0, 100])
ax[2].set_yticks([0,20,40,60,80,100])
ax[2].set_yticklabels([0,20,40,60,80,100])
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/CoxPHazard_feature-importance_status-1.{args.format}"), dpi=300, format=args.format)

print("FINISHED FEATURE IMPORTANCE USING GRID SEARCH AND COX PH MODELS")
print("   ************************   ")

####################################################################################################################################################################
## FEATURE SELECTION USING CLASSICAL ML CLASSIFIERS TO PREDICT SURVIVAL
####################################################################################################################################################################
N_splits = [8,6,2]
runs = 50
times_auc_split = {
    2: np.arange(180,1800,180),
    6: np.arange(180,1100,180),
    8: np.arange(180,1000,180),
}
colors = [
    "brown",
    "black",
    "pink",
    "cyan",
    "orange",
    "green",
    "blue",
    "purple",
    "gray",
    "red"
]

# Using Cox Prop Hazard models
fig, ax = plt.subplots(len(N_splits),2, figsize=(16,5*len(N_splits)), gridspec_kw={'width_ratios': [2, 1]})
for i_ax, splits in enumerate(N_splits):
    print(f"No. of splits: {splits}")
    times_auc = times_auc_split[splits] 
    HarrellCindex = np.zeros((len(TDMaps.columns)-1, int(splits*runs)))
    DynAUC = np.zeros((len(TDMaps.columns)-1, int(splits*runs), len(times_auc))) * np.nan
    DynAUC_average = np.zeros((len(TDMaps.columns)-1, int(splits*runs))) * np.nan
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]    
        # Remove rows where x, y, or life is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        # Censoring (based on the status variable)
        OS_STATS = []
        OS_STATS.extend([(st, os) for st,os in zip(life_clean,y_clean.values)])
        OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])
        # KFold and survival prediction
        RKF = RepeatedKFold(n_splits=splits, n_repeats=runs)
        fails = 0
        for j, (train_index, test_index) in enumerate(RKF.split(x_clean, OS_STATS)):
            # MODEL 1: Cox Proportional Hazard
            Cmodel = CoxPHSurvivalAnalysis(n_iter=200)
            Cmodel.fit(x_clean.values[train_index].reshape(-1,1), OS_STATS[train_index])
            HarrellCindex[i-1,j] = Cmodel.score(
                    x_clean.values[test_index].reshape(-1,1),
                    OS_STATS[test_index]
            )
            try:
                DynAUC[i-1,j,:], DynAUC_average[i-1,j] = cumulative_dynamic_auc(
                    OS_STATS,#[train_index]
                    OS_STATS[test_index],
                    Cmodel.predict(x_clean.values[test_index].reshape(-1,1)),
                    times=times_auc
                )
            except:
                fails += 1
                print(f"The times were not within the follow-up time range of test data. {round(100*fails/(runs*splits),2)}% of the runs have been ignored so far.")

        # TODO: Implement permutation procedure for the calculation of pvalues
        ax[i_ax,0].plot(times_auc/daysXmonth, np.nanmean(DynAUC[i-1], axis=0), '-o', linewidth=0.5, markersize=8, label=TDMaps.columns[i], color=colors[i-1], alpha=.5)
    ax[i_ax,0].hlines(0.5, -3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3, color='black', linewidth=.75, linestyle='--', alpha=.5)
    ax[i_ax,0].spines[["top","right"]].set_visible(False)
    ax[i_ax,0].set_xlim([-3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3])
    ax[i_ax,0].set_xticks(times_auc/daysXmonth)
    ax[i_ax,0].set_xticklabels(np.int32(times_auc/daysXmonth))
    ax[i_ax,0].set_ylim([0.4,0.75])
    ax[i_ax,0].set_yticks([0.4,.5,.6,.7])
    ax[i_ax,0].set_ylabel("Cumulative dynamic AUC(t)")
    ax[i_ax,0].set_title(f"No. of splits: {splits}", fontweight='bold')
    ax[i_ax,0].set_xlabel("Survival at time (months)")
    ax[i_ax,1].bar(
        TDMaps.columns[1:],
        np.nanmean(DynAUC_average, axis=1),
        edgecolor="black",
        color=["blue" if i in np.nanmean(DynAUC_average, axis=1).argsort()[-3:]  else "lightgrey" for i in range(len(TDMaps.columns)-1)]
    )
    ax[i_ax,1].hlines(0.5, -1, len(TDMaps.columns), color='black', linewidth=.75, linestyle='--', alpha=.5)
    ax[i_ax,1].spines[["top","right","left"]].set_visible(False)
    ax[i_ax,1].set_xlim([-1, len(TDMaps.columns)-1])
    ax[i_ax,1].set_xticks(range(0, len(TDMaps.columns)-1))
    ax[i_ax,1].set_ylim([0.4,0.75])
    ax[i_ax,1].set_yticks([0.4,.5,.6,.7])
    ax[i_ax,1].set_yticklabels([])
    ax[i_ax,1].tick_params(axis='y', length=0) 
    if i_ax==(len(N_splits)-1):
        ax[i_ax,1].set_xticklabels(TDMaps.columns[1:], rotation=75, fontsize=8)
    elif i_ax==0:
        ax[i_ax,1].set_title("Average AUC(t)", fontweight='bold')
        ax[i_ax,0].legend(ncols=5, fontsize=7.5, frameon=False)
        ax[i_ax,1].set_xticklabels([])
        ax[i_ax,1].tick_params(axis='x', length=0) 
    else:
        ax[i_ax,1].set_xticklabels([])
        ax[i_ax,1].tick_params(axis='x', length=0) 
fig.tight_layout()
fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-prediction_model-Cox.{args.format}"), dpi=300, format=args.format)
plt.close()

# Using Random survival forest
# INFO on the minimum number of samples per leaf:
#    - For 6 and 7 the performance is bad (barely above .5)
#    - For 15 the performance increase and the results are more similar to the ones obtained by the Cox models
#    - For 25 the performance increase slightly more and the results are more similar to the ones obtained by the Cox models
#    - For 40 the performance is similar to the case of 15 and 20
for min_samples_leaf in [30]:
    fig, ax = plt.subplots(len(N_splits),2, figsize=(16,5*len(N_splits)), gridspec_kw={'width_ratios': [2, 1]}) # TODO add a row for each split tried
    for i_ax, splits in enumerate(N_splits):
        print(f"No. of splits: {splits}")
        times_auc = times_auc_split[splits] 
        print(f"         {TDMaps.columns[i]}")
        DynAUC = np.zeros((len(TDMaps.columns)-1, int(splits*runs), len(times_auc))) * np.nan
        DynAUC_average = np.zeros((len(TDMaps.columns)-1, int(splits*runs))) * np.nan
        for i in range(1,len(TDMaps.columns)):
            x = TDMaps[TDMaps.columns[i]]
            y = TDMaps["OS"]    
            # Remove rows where x, y, or life is NaN
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
            x_clean = x[mask]
            y_clean = y[mask]
            life_clean = life[mask]
            # Censoring (based on the status variable)
            OS_STATS = []
            OS_STATS.extend([(st, os) for st,os in zip(life_clean,y_clean.values)])
            OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])
            # KFold and survival prediction
            RKF = RepeatedKFold(n_splits=splits, n_repeats=runs)
            fails = 0
            for j, (train_index, test_index) in enumerate(RKF.split(x_clean, OS_STATS)):
                # MODEL 2: Random survival forest
                RSFmodel = RandomSurvivalForest(n_estimators=100, min_samples_leaf=min_samples_leaf)
                RSFmodel.fit(x_clean.values[train_index].reshape(-1,1), OS_STATS[train_index])
                RSFmodel.predict_cumulative_hazard_function(x_clean.values[test_index].reshape(-1,1), return_array=False)
                try:
                    RSFpreds = rsf_risk_scores = np.vstack(
                        [chf(times_auc) for chf in RSFmodel.predict_cumulative_hazard_function(x_clean.values[test_index].reshape(-1,1), return_array=False)]
                    )
                    DynAUC[i-1,j,:], DynAUC_average[i-1,j] = cumulative_dynamic_auc(
                        OS_STATS,#[train_index]
                        OS_STATS[test_index],
                        RSFpreds,
                        times=times_auc
                    )
                except:
                    fails += 1
                    print(f"The times were not within the follow-up time range of test data. {round(100*fails/(runs*splits),2)}% of the runs have been ignored so far.")

            # TODO: Implement permutation procedure for the calculation of pvalues
            ax[i_ax,0].plot(times_auc/daysXmonth, np.nanmean(DynAUC[i-1], axis=0), '-o', linewidth=0.5, markersize=8, label=TDMaps.columns[i], color=colors[i-1], alpha=.5)
        ax[i_ax,0].hlines(0.5, -3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3, color='black', linewidth=.75, linestyle='--', alpha=.5)
        ax[i_ax,0].spines[["top","right"]].set_visible(False)
        ax[i_ax,0].set_xlim([-3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3])
        ax[i_ax,0].set_xticks(times_auc/daysXmonth)
        ax[i_ax,0].set_xticklabels(np.int32(times_auc/daysXmonth))
        ax[i_ax,0].set_ylim([.3,.7])
        ax[i_ax,0].set_yticks([.3,.4,.5,.6,.7])
        ax[i_ax,0].set_ylabel("Cumulative dynamic AUC(t)")
        ax[i_ax,0].set_title(f"No. of splits: {splits}", fontweight='bold')
        ax[i_ax,0].set_xlabel("Survival at time (months)")
        ax[i_ax,1].bar(
            TDMaps.columns[1:],
            np.nanmean(DynAUC_average, axis=1),
            edgecolor="black",
            color=["blue" if i in np.nanmean(DynAUC_average, axis=1).argsort()[-3:]  else "lightgrey" for i in range(len(TDMaps.columns)-1)]
        )
        ax[i_ax,1].hlines(0.5, -1, len(TDMaps.columns), color='black', linewidth=.75, linestyle='--', alpha=.5)
        ax[i_ax,1].spines[["top","right","left"]].set_visible(False)
        ax[i_ax,1].set_xlim([-1, len(TDMaps.columns)-1])
        ax[i_ax,1].set_xticks(range(0, len(TDMaps.columns)-1))
        ax[i_ax,1].set_ylim([0.3,0.7])
        ax[i_ax,1].set_yticks([.3,.4,.5,.6,.7])
        ax[i_ax,1].set_yticklabels([])
        ax[i_ax,1].tick_params(axis='y', length=0) 
        if i_ax==(len(N_splits)-1):
            ax[i_ax,1].set_xticklabels(TDMaps.columns[1:], rotation=75, fontsize=8)
        elif i_ax==0:
            ax[i_ax,1].set_title("Average AUC(t)", fontweight='bold')
            ax[i_ax,0].legend(ncols=5, fontsize=7.5, frameon=False)
            ax[i_ax,1].set_xticklabels([])
            ax[i_ax,1].tick_params(axis='x', length=0) 
        else:
            ax[i_ax,1].set_xticklabels([])
            ax[i_ax,1].tick_params(axis='x', length=0) 
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-prediction_model-RSF_minLeaf-{min_samples_leaf}.{args.format}"), dpi=300, format=args.format)
    plt.close()
    print("*****************************")
"""
""" # Using an SVClassifier to predict survival
splits = 8 # Determine whether 8 is a good number or we should experiment more
colors = ["black", "forestgreen", "purple", "cornflowerblue"]
for model in ["linear","rbf"]:# 
    for plow, phigh in [(25,75),(50,50)]:
        print(f"{model} SVC: Percentiles ({plow},{phigh})")
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        fig_bis, ax_bis = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.flatten()
        ax_bis = ax_bis.flatten()
        for i in range(1,len(TDMaps.columns)):
            x = TDMaps[TDMaps.columns[i]]
            y = TDMaps["OS"]    
            # Remove rows where x or y is NaN
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
            x_clean = x[mask]
            y_clean = y[mask]
            life_clean = life[mask]    
            # Results
            metrics = {
                "Balanced_accuracy": np.zeros((len(months),)) * np.nan, 
                "Accuracy": np.zeros((len(months),)) * np.nan, 
                #"PPV": np.zeros((len(months),)) * np.nan, 
                "Recall": np.zeros((len(months),)) * np.nan, 
            }
            scoring = {
                "Balanced_accuracy": make_scorer(balanced_accuracy_score), 
                "Accuracy": make_scorer(accuracy_score),
                #"PPV": make_scorer(precision_score),
                "Recall": make_scorer(recall_score)
            }
            pv_s = {
                "Balanced_accuracy": np.zeros((len(months),)) * np.nan, 
                "Accuracy": np.zeros((len(months),)) * np.nan, 
                #"PPV": np.zeros((len(months),)) * np.nan, 
                "Recall": np.zeros((len(months),)) * np.nan, 
            }
            nums = np.zeros((len(months),2))
            for j,m in enumerate(months):
                print(f"         {TDMaps.columns[i]}: {m} months")
                month_mask = (y_clean<=m*daysXmonth) & (life_clean==0) # Discard censored data points with censoring times smaller than cutoff
                tdi = x_clean[~month_mask]
                y_month_mask = y_clean[~month_mask]
                tdi_mask = (tdi>np.percentile(tdi, plow)) & (tdi<np.percentile(tdi, phigh)) # Discard value between percentiles (to emualte Salvalaggio, et al. 2023)
                tdi = tdi[~tdi_mask]
                y_month_mask = y_month_mask[~tdi_mask]
                died = np.where(y_month_mask>=m*daysXmonth, 0, 1)
                nums[j,:] = [(1-died).sum(), died.sum()]
                ax[i-1].text(m-2, -0.175, f"{int(nums[j,0])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="forestgreen") # Numbers alive
                ax[i-1].text(m-2, -0.225, f"{int(nums[j,1])}", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="darkorange") # Numbers dead
                tdi = tdi / max(tdi)
                for kk, metric in enumerate(scoring.keys()):
                    RSKF = RepeatedStratifiedKFold(n_splits=splits, n_repeats=runs)
                    classifier = SVC(C=1, kernel=model, class_weight='balanced')
                    cv_results, _, pv_results = permutation_test_score(#cross_validate(
                        classifier,
                        tdi.values.reshape(-1,1),
                        y=died,
                        scoring=scoring[metric],
                        cv=RSKF,
                        n_permutations=500,
                        n_jobs=8
                    )
                    metrics[metric][j] = cv_results
                    pv_s[metric][j] = pv_results   
            for kk, metric in enumerate(scoring.keys()):
                ax[i-1].plot(
                    months+kk-2, 
                    metrics[metric], 
                    'o',
                    label=metric, 
                    linestyle='--', 
                    color=colors[kk],
                    alpha=.75
                )
                ax_bis[i-1].plot(
                    months+kk-2, 
                    pv_s[metric], 
                    'o',
                    label=metric, 
                    linestyle='-', 
                    color=colors[kk],
                    alpha=.75
                )
                ax_bis[i-1].plot(
                    months+kk-2, # if fwer: Holm's procedure // else: Benjamin-Hochberg
                    multipletests(pv_s[metric], alpha=0.05, method='holm', is_sorted=False)[1] if fwer else fdrcorrection(pv_s[metric], alpha=0.05, method='p', is_sorted=False)[1],
                    'o',
                    label=metric+r"$_{ FWER}$" if fwer else metric+r"$_{ FDR}$", 
                    linestyle='--', 
                    color=colors[kk],
                    alpha=.5
                )
            # Metrics
            ax[i-1].text(months[0]-2, -0.125, "No. of samples", transform=ax[i-1].transData, fontsize=12, verticalalignment='top', color="black", fontweight='bold') 
            ax[i-1].hlines(0.5, months[0]-5, months[-1]+5, color='black', linewidth=.75, linestyle='--', alpha=.5)
            ax[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
            ax[i-1].spines[["top", "right"]].set_visible(False)
            ax[i-1].set_ylabel("Metric (a.u.)", fontsize=12)
            ax[i-1].set_xlabel("Death (months)", fontsize=12)
            ax[i-1].set_xlim([months[0]-5, months[-1]+5])
            ax[i-1].set_xticks(months)
            ax[i-1].set_xticklabels(months)
            ax[i-1].set_ylim([0,1])
            ax[i-1].set_yticks([0, 0.25,0.5,0.6,0.7,0.8,1])
            ax[i-1].set_yticklabels([0, 0.25,0.5,0.6,0.7,0.8,1])
            ax[i-1].spines['bottom'].set_bounds(months[0], months[-1])
            # Significance 
            ax_bis[i-1].hlines(0.05, months[0]-5, months[-1]+5, color='red', linewidth=.75, linestyle='--', alpha=.5)
            ax_bis[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
            ax_bis[i-1].spines[["top", "right"]].set_visible(False)
            ax_bis[i-1].set_ylabel("P-value (a.u.)", fontsize=12)
            ax_bis[i-1].set_xlabel("Death (months)", fontsize=12)
            ax_bis[i-1].set_xlim([months[0]-5, months[-1]+5])
            ax_bis[i-1].set_xticks(months)
            ax_bis[i-1].set_xticklabels(months)
            ax_bis[i-1].set_ylim([0.001,1])
            ax_bis[i-1].set_yticks([0.001, 0.01, 0.05, 1])
            ax_bis[i-1].set_yticklabels([0.001, 0.01, 0.05, 1])
            ax_bis[i-1].spines['bottom'].set_bounds(months[0], months[-1])
            ax_bis[i-1].set_yscale('log')
            if (i-1)==0:
                ax[i-1].legend(frameon=False, ncols=2, loc="upper right")
                ax_bis[i-1].legend(frameon=False, ncols=1, loc="lower right")
        fig.tight_layout()
        fig_bis.tight_layout()
        fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-prediction_model-{model}SVC_percTDI-{plow}.{args.format}"), dpi=300, format=args.format)
        fig_bis.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-prediction_model-{model}SVC_percTDI_significance-{plow}.{args.format}"), dpi=300, format=args.format)
        plt.close(fig)
        plt.close(fig_bis) """
"""
# Using the TDI classifiers to compute the time-dependent AUC(t)
for plow, phigh in [(25,75),(50,50)]:
    print(f"TDI classifier: Percentiles ({plow},{phigh})")
    fig, ax = plt.subplots(len(N_splits),2, figsize=(16,5*len(N_splits)), gridspec_kw={'width_ratios': [2, 1]})
    for i_ax, splits in enumerate(N_splits):
        print(f"No. of splits: {splits}")
        times_auc = times_auc_split[splits] 
        print(f"         {TDMaps.columns[i]}")
        DynAUC = np.zeros((len(TDMaps.columns)-1, int(splits*runs), len(times_auc))) * np.nan
        DynAUC_average = np.zeros((len(TDMaps.columns)-1, int(splits*runs))) * np.nan
        for i in range(1,len(TDMaps.columns)):
            x = TDMaps[TDMaps.columns[i]]
            y = TDMaps["OS"]    
            # Remove rows where x, y, or life is NaN
            mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
            x_clean = x[mask]
            y_clean = y[mask]
            life_clean = life[mask]
            # Censoring (based on the status variable)
            OS_STATS = []
            OS_STATS.extend([(st, os) for st,os in zip(life_clean,y_clean.values)])
            OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', '<f4')])
            # KFold and survival prediction
            RKF = RepeatedKFold(n_splits=splits, n_repeats=runs)
            fails = 0
            for j, (train_index, test_index) in enumerate(RKF.split(x_clean, OS_STATS)):
                # MODEL 4: TDI features
                try:
                    DynAUC[i-1,j,:], DynAUC_average[i-1,j] = cumulative_dynamic_auc(
                        OS_STATS,#[train_index]
                        OS_STATS[test_index],
                        x_clean.values[test_index],
                        times=times_auc
                    )
                except:
                    fails += 1
                    print(f"The times were not within the follow-up time range of test data. {round(100*fails/(runs*splits),2)}% of the runs have been ignored so far.")

            # TODO: Implement permutation procedure for the calculation of pvalues
            ax[i_ax,0].plot(times_auc/daysXmonth, np.nanmean(DynAUC[i-1], axis=0), '-o', linewidth=0.5, markersize=8, label=TDMaps.columns[i], color=colors[i-1], alpha=.5)
        ax[i_ax,0].hlines(0.5, -3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3, color='black', linewidth=.75, linestyle='--', alpha=.5)
        ax[i_ax,0].spines[["top","right"]].set_visible(False)
        ax[i_ax,0].set_xlim([-3+times_auc[0]/daysXmonth, times_auc[-1]/daysXmonth+3])
        ax[i_ax,0].set_xticks(times_auc/daysXmonth)
        ax[i_ax,0].set_xticklabels(np.int32(times_auc/daysXmonth))
        ax[i_ax,0].set_ylim([.3,.7])
        ax[i_ax,0].set_yticks([.3,.4,.5,.6,.7])
        ax[i_ax,0].set_ylabel("Cumulative dynamic AUC(t)")
        ax[i_ax,0].set_title(f"No. of splits: {splits}", fontweight='bold')
        ax[i_ax,0].set_xlabel("Survival at time (months)")
        ax[i_ax,1].bar(
            TDMaps.columns[1:],
            np.nanmean(DynAUC_average, axis=1),
            edgecolor="black",
            color=["blue" if i in np.nanmean(DynAUC_average, axis=1).argsort()[-3:]  else "lightgrey" for i in range(len(TDMaps.columns)-1)]
        )
        ax[i_ax,1].hlines(0.5, -1, len(TDMaps.columns), color='black', linewidth=.75, linestyle='--', alpha=.5)
        ax[i_ax,1].spines[["top","right","left"]].set_visible(False)
        ax[i_ax,1].set_xlim([-1, len(TDMaps.columns)-1])
        ax[i_ax,1].set_xticks(range(0, len(TDMaps.columns)-1))
        ax[i_ax,1].set_ylim([0.3,0.7])
        ax[i_ax,1].set_yticks([.3,.4,.5,.6,.7])
        ax[i_ax,1].set_yticklabels([])
        ax[i_ax,1].tick_params(axis='y', length=0) 
        if i_ax==(len(N_splits)-1):
            ax[i_ax,1].set_xticklabels(TDMaps.columns[1:], rotation=75, fontsize=8)
        elif i_ax==0:
            ax[i_ax,1].set_title("Average AUC(t)", fontweight='bold')
            ax[i_ax,0].legend(ncols=5, fontsize=7.5, frameon=False)
            ax[i_ax,1].set_xticklabels([])
            ax[i_ax,1].tick_params(axis='x', length=0) 
        else:
            ax[i_ax,1].set_xticklabels([])
            ax[i_ax,1].tick_params(axis='x', length=0) 
    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Survival-prediction_model-TDI_percTDI-{plow}.{args.format}"), dpi=300, format=args.format)
    plt.close(fig)
"""

# Using the TDI classifiers to compute the AUC at different survival times
months = [12,24,36,48]
colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]
TRUE_DEATHS, TDI_data = dict(), dict() # Record ground truths and indices to perform delong test between pairs of ROCs
for plow, phigh in percentiles2check:    
    TRUE_DEATHS[f"({plow}, {phigh})"], TDI_data[f"({plow}, {phigh})"] = dict(), dict()
    print(f"TDI classifier: Percentiles ({plow},{phigh})")
    fig_roc, ax_roc = plt.subplots(nrows, ncols, figsize=figsize)
    ax_roc = ax_roc.flatten()
    fig_auc, ax_auc = plt.subplots(nrows, ncols, figsize=figsize)
    ax_auc = ax_auc.flatten()
    fig_acc, ax_acc = plt.subplots(nrows, ncols, figsize=figsize)
    ax_acc = ax_acc.flatten()
    fig_bacc, ax_bacc = plt.subplots(nrows, ncols, figsize=figsize)
    ax_bacc = ax_bacc.flatten()
    fig_ppv, ax_ppv = plt.subplots(nrows, ncols, figsize=figsize)
    ax_ppv = ax_ppv.flatten()
    fig_npv, ax_npv = plt.subplots(nrows, ncols, figsize=figsize)
    ax_npv = ax_npv.flatten()
    fig_fdr, ax_fdr = plt.subplots(nrows, ncols, figsize=figsize)
    ax_fdr = ax_fdr.flatten()
    for i in range(1,len(TDMaps.columns)):
        x = TDMaps[TDMaps.columns[i]]
        y = TDMaps["OS"]    
        # Remove rows where x, y, or life is NaN
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(life)
        x_clean = x[mask]
        y_clean = y[mask]
        life_clean = life[mask]
        # pvals
        perm_auc, perm_acc, perm_bacc = np.zeros((len(months),)), np.zeros((len(months),)), np.zeros((len(months),))
        perm_ppv, perm_npv, perm_fdr = np.zeros((len(months),)), np.zeros((len(months),)), np.zeros((len(months),))
        p_auc, p_acc, p_bacc = np.zeros((len(months),)), np.zeros((len(months),)), np.zeros((len(months),))
        p_ppv, p_npv, p_fdr = np.zeros((len(months),)), np.zeros((len(months),)), np.zeros((len(months),))
        TRUE_DEATHS[f"({plow}, {phigh})"][TDMaps.columns[i]], TDI_data[f"({plow}, {phigh})"][TDMaps.columns[i]] = dict(), dict()
        for j,m in enumerate(months):
            print(f"         {TDMaps.columns[i]}: {m} months")
            month_mask = (y_clean<=m*daysXmonth) & (life_clean==0) # Discard censored data points with censoring times smaller than cutoff
            tdi = x_clean[~month_mask]
            y_month_mask = y_clean[~month_mask]
            tdi_mask = (tdi>np.percentile(tdi, plow)) & (tdi<np.percentile(tdi, phigh)) # Discard value between percentiles (to emualte Salvalaggio, et al. 2023)
            tdi = tdi[~tdi_mask]
            y_month_mask = y_month_mask[~tdi_mask]
            died = np.where(y_month_mask>=m*daysXmonth, 0, 1)
            fpr, tpr, thresholds = roc_curve(died, tdi)
            TRUE_DEATHS[f"({plow}, {phigh})"][TDMaps.columns[i]][m] = died
            TDI_data[f"({plow}, {phigh})"][TDMaps.columns[i]][m] = tdi.to_numpy()
            # Metrics
            optimal_idx = np.argmax(tpr - fpr)
            optimal_th = thresholds[optimal_idx]
            death_preds = np.where(tdi>=optimal_th, 1, 0)
            auc = roc_auc_score(died, tdi)
            acc = accuracy_score(died, death_preds)
            bacc = balanced_accuracy_score(died, death_preds)
            tn, fp, fn, tp = confusion_matrix(died, death_preds).ravel()
            ppv, npv, fdr = tp / (tp + fp), tn / (tn + fn), fp / (tp+fp)
            # Permute labels to compute the significance of the metrics
            pop_auc, pop_acc, pop_bacc, pop_ppv, pop_npv, pop_fdr = [], [], [], [], [], []
            for _ in range(n_perms):
                perm_deaths = np.random.permutation(died)
                pop_auc.append(roc_auc_score(perm_deaths, tdi))
                pop_acc.append(accuracy_score(perm_deaths, death_preds))
                pop_bacc.append(balanced_accuracy_score(perm_deaths, death_preds))
                perm_tn, perm_fp, perm_fn, perm_tp = confusion_matrix(perm_deaths, death_preds).ravel()
                pop_ppv.append(perm_tp / (perm_tp + perm_fp))
                pop_npv.append(perm_tn / (perm_tn + perm_fn))
                pop_fdr.append(perm_fp / (perm_tp+perm_fp))
            p_auc[j], p_acc[j], p_bacc[j] = np.mean(np.array(pop_auc) >= auc), np.mean(np.array(pop_acc) >= acc), np.mean(np.array(pop_bacc) >= bacc)
            p_ppv[j], p_npv[j], p_fdr[j] = np.mean(np.array(pop_ppv) >= ppv), np.mean(np.array(pop_npv) >= npv), np.mean(np.array(pop_fdr) <= fdr)
            perm_auc[j], perm_acc[j], perm_bacc[j] = np.mean(np.array(pop_auc)), np.mean(np.array(pop_acc)), np.mean(np.array(pop_bacc))
            perm_ppv[j], perm_npv[j], perm_fdr[j] = np.mean(np.array(pop_ppv)), np.mean(np.array(pop_npv)), np.mean(np.array(pop_fdr))
            # Plots
            ax_roc[i-1].plot(fpr, tpr, linestyle='-', linewidth=2.5, color=colors[j], label=f"Death at {m} months", alpha=.8) # Plot ROC
            ax_roc[i-1].plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=15, color=colors[j], markerfacecolor=None, markeredgecolor='black', markeredgewidth=2.5, alpha=0.5)
            ax_roc[i-1].plot([fpr[optimal_idx],fpr[optimal_idx]],[fpr[optimal_idx],tpr[optimal_idx]], linestyle='--', linewidth=1.5, color="gray")           
            ax_auc[i-1].bar(m, auc, color="cornflowerblue" if p_auc[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot AUC           
            ax_acc[i-1].bar(m, acc, color="cornflowerblue" if p_acc[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot Accuracy          
            ax_bacc[i-1].bar(m, bacc, color="cornflowerblue" if p_bacc[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot Balanced accuracy 
            ax_ppv[i-1].bar(m, ppv, color="cornflowerblue" if p_ppv[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot Positive predictive value
            ax_npv[i-1].bar(m, npv, color="cornflowerblue" if p_npv[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot Negative predictive value
            ax_fdr[i-1].bar(m, fdr, color="cornflowerblue" if p_fdr[j]<=0.05 else "lightgray", edgecolor="black", width=5) # Plot False discovery rate
        if fwer: # Method: Holm's procedure
            _, p_auc_corr, _, _ = multipletests(p_auc, alpha=0.05, method='holm', is_sorted=False)
            _, p_acc_corr, _, _ = multipletests(p_acc, alpha=0.05, method='holm', is_sorted=False)
            _, p_bacc_corr, _, _ = multipletests(p_bacc, alpha=0.05, method='holm', is_sorted=False)
            _, p_ppv_corr, _, _ = multipletests(p_ppv, alpha=0.05, method='holm', is_sorted=False)
            _, p_npv_corr, _, _ = multipletests(p_npv, alpha=0.05, method='holm', is_sorted=False)
            _, p_fdr_corr, _, _ = multipletests(p_fdr, alpha=0.05, method='holm', is_sorted=False)
        else: # Method: Benjamin-Hochberg
            _, p_auc_corr = fdrcorrection(p_auc, alpha=0.05, method='p', is_sorted=False)
            _, p_acc_corr = fdrcorrection(p_acc, alpha=0.05, method='p', is_sorted=False)
            _, p_bacc_corr = fdrcorrection(p_bacc, alpha=0.05, method='p', is_sorted=False)
            _, p_ppv_corr = fdrcorrection(p_ppv, alpha=0.05, method='p', is_sorted=False)
            _, p_npv_corr = fdrcorrection(p_npv, alpha=0.05, method='p', is_sorted=False)
            _, p_fdr_corr = fdrcorrection(p_fdr, alpha=0.05, method='p', is_sorted=False)
        # Axes ROC
        ax_roc[i-1].plot([0,1],[0,1], linestyle='--', linewidth=.5, color="k")
        ax_roc[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_roc[i-1].spines[["top", "right"]].set_visible(False)
        ax_roc[i-1].set_xlim([-.05,1.05])
        ax_roc[i-1].set_xticks([0,0.5,1])
        ax_roc[i-1].set_xlabel("False positive rate (FPR)", fontsize=12)
        ax_roc[i-1].spines['bottom'].set_bounds(0,1)
        ax_roc[i-1].set_ylim([0,1])
        ax_roc[i-1].set_yticks([0,0.5,1])
        ax_roc[i-1].set_ylabel("True positive rate (TPR)", fontsize=12)
        ax_roc[i-1].spines['left'].set_bounds(0,1)
        # Axes AUC
        ax_auc[i-1].plot(months, perm_auc, 'o', linestyle='--', linewidth=.5, color="k")
        ax_auc[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_auc[i-1].spines[["top", "right"]].set_visible(False)
        ax_auc[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_auc[i-1].set_xticks(months)
        ax_auc[i-1].set_xlabel("Death (months)")
        ax_auc[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_auc[i-1].set_ylim([0.25,1])
        ax_auc[i-1].set_yticks([0.25,0.5,0.6,0.7,0.8,1])
        ax_auc[i-1].set_ylabel("Area Under the Curve (AUC)")
        # Axes ACC
        ax_acc[i-1].plot(months, perm_acc, 'o', linestyle='--', linewidth=.5, color="k")
        ax_acc[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_acc[i-1].spines[["top", "right"]].set_visible(False)
        ax_acc[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_acc[i-1].set_xticks(months)
        ax_acc[i-1].set_xlabel("Death (months)")
        ax_acc[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_acc[i-1].set_ylim([0.25, 1])
        ax_acc[i-1].set_yticks([0.25, 0.5, 0.6, 0.7, 0.8, 1])
        ax_acc[i-1].set_ylabel("Accuracy")
        # Axes BACC
        ax_bacc[i-1].plot(months, perm_bacc, 'o', linestyle='--', linewidth=.5, color="k")
        ax_bacc[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_bacc[i-1].spines[["top", "right"]].set_visible(False)
        ax_bacc[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_bacc[i-1].set_xticks(months)
        ax_bacc[i-1].set_xlabel("Death (months)")
        ax_bacc[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_bacc[i-1].set_ylim([0.25, 1])
        ax_bacc[i-1].set_yticks([0.25, 0.5, 0.6, 0.7, 0.8, 1])
        ax_bacc[i-1].set_ylabel("Balanced Accuracy")    
        # Axes PPV
        ax_ppv[i-1].plot(months, perm_ppv, 'o', linestyle='--', linewidth=.5, color="k")
        ax_ppv[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_ppv[i-1].spines[["top", "right"]].set_visible(False)
        ax_ppv[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_ppv[i-1].set_xticks(months)
        ax_ppv[i-1].set_xlabel("Death (months)")
        ax_ppv[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_ppv[i-1].set_ylim([0.25, 1])
        ax_ppv[i-1].set_yticks([0.25, 0.5, 0.6, 0.7, 0.8, 1])
        ax_ppv[i-1].set_ylabel("Positive Predictive Value")  
        # Axes NPV
        ax_npv[i-1].plot(months, perm_npv, 'o', linestyle='--', linewidth=.5, color="k")
        ax_npv[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_npv[i-1].spines[["top", "right"]].set_visible(False)
        ax_npv[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_npv[i-1].set_xticks(months)
        ax_npv[i-1].set_xlabel("Death (months)")
        ax_npv[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_npv[i-1].set_ylim([0.25, 1])
        ax_npv[i-1].set_yticks([0.25, 0.5, 0.6, 0.7, 0.8, 1])
        ax_npv[i-1].set_ylabel("Negative Predictive Value")        
        # Axes NPV
        ax_fdr[i-1].plot(months, perm_fdr, 'o', linestyle='--', linewidth=.5, color="k")
        ax_fdr[i-1].set_title(TDMaps.columns[i], fontweight="bold", fontsize=12)
        ax_fdr[i-1].spines[["top", "right"]].set_visible(False)
        ax_fdr[i-1].set_xlim([months[0]-5, months[-1]+5])
        ax_fdr[i-1].set_xticks(months)
        ax_fdr[i-1].set_xlabel("Death (months)")
        ax_fdr[i-1].spines['bottom'].set_bounds(months[0]-2.5, months[-1]+2.5)
        ax_fdr[i-1].set_ylim([0, 1])
        ax_fdr[i-1].set_yticks([0.25, 0.5, 0.75, 1])
        ax_npv[i-1].set_ylabel("False discovery rate")        
        # Significance and adjusted significance
        for j,m in enumerate(months):
            # AUC
            if p_auc[j]<=0.001:
                ax_auc[i-1].text(m-1.5, 0.9, '***', color='black',fontsize=10)
            elif p_auc[j]<=0.01:
                ax_auc[i-1].text(m-1, 0.9, '**', color='black',fontsize=10)
            elif p_auc[j]<=0.05:
                ax_auc[i-1].text(m-.5, 0.9, '*', color='black',fontsize=10)
            else:            
                ax_auc[i-1].text(m-1.5, 0.9, 'n.s.', color='black',fontsize=10)
            if p_auc_corr[j]<=0.001:
                ax_auc[i-1].text(m-1.5, 0.95, '***', color='blue',fontsize=10)
            elif p_auc_corr[j]<=0.01:
                ax_auc[i-1].text(m-1, 0.95, '**', color='blue',fontsize=10)
            elif p_auc_corr[j]<=0.05:
                ax_auc[i-1].text(m-.5, 0.95, '*', color='blue',fontsize=10)
            else:            
                ax_auc[i-1].text(m-1.5, 0.95, 'n.s.', color='blue',fontsize=10)  
            # Accuracy significance labels
            if p_acc[j] <= 0.001:
                ax_acc[i-1].text(m-1.5, 0.9, '***', color='black', fontsize=10)
            elif p_acc[j] <= 0.01:
                ax_acc[i-1].text(m-1, 0.9, '**', color='black', fontsize=10)
            elif p_acc[j] <= 0.05:
                ax_acc[i-1].text(m-0.5, 0.9, '*', color='black', fontsize=10)
            else:
                ax_acc[i-1].text(m-1.5, 0.9, 'n.s.', color='black', fontsize=10)            
            if p_acc_corr[j] <= 0.001:
                ax_acc[i-1].text(m-1.5, 0.95, '***', color='blue', fontsize=10)
            elif p_acc_corr[j] <= 0.01:
                ax_acc[i-1].text(m-1, 0.95, '**', color='blue', fontsize=10)
            elif p_acc_corr[j] <= 0.05:
                ax_acc[i-1].text(m-0.5, 0.95, '*', color='blue', fontsize=10)
            else:
                ax_acc[i-1].text(m-1.5, 0.95, 'n.s.', color='blue', fontsize=10)
            # Balanced Accuracy significance labels
            if p_bacc[j] <= 0.001:
                ax_bacc[i-1].text(m-1.5, 0.9, '***', color='black', fontsize=10)
            elif p_bacc[j] <= 0.01:
                ax_bacc[i-1].text(m-1, 0.9, '**', color='black', fontsize=10)
            elif p_bacc[j] <= 0.05:
                ax_bacc[i-1].text(m-0.5, 0.9, '*', color='black', fontsize=10)
            else:
                ax_bacc[i-1].text(m-1.5, 0.9, 'n.s.', color='black', fontsize=10)               
            if p_bacc_corr[j] <= 0.001:
                ax_bacc[i-1].text(m-1.5, 0.95, '***', color='blue', fontsize=10)
            elif p_bacc_corr[j] <= 0.01:
                ax_bacc[i-1].text(m-1, 0.95, '**', color='blue', fontsize=10)
            elif p_bacc_corr[j] <= 0.05:
                ax_bacc[i-1].text(m-0.5, 0.95, '*', color='blue', fontsize=10)
            else:
                ax_bacc[i-1].text(m-1.5, 0.95, 'n.s.', color='blue', fontsize=10)
            # positive predictive value significance labels
            if p_ppv[j] <= 0.001:
                ax_ppv[i-1].text(m-1.5, 0.9, '***', color='black', fontsize=10)
            elif p_ppv[j] <= 0.01:
                ax_ppv[i-1].text(m-1, 0.9, '**', color='black', fontsize=10)
            elif p_ppv[j] <= 0.05:
                ax_ppv[i-1].text(m-0.5, 0.9, '*', color='black', fontsize=10)
            else:
                ax_ppv[i-1].text(m-1.5, 0.9, 'n.s.', color='black', fontsize=10)               
            if p_ppv_corr[j] <= 0.001:
                ax_ppv[i-1].text(m-1.5, 0.95, '***', color='blue', fontsize=10)
            elif p_ppv_corr[j] <= 0.01:
                ax_ppv[i-1].text(m-1, 0.95, '**', color='blue', fontsize=10)
            elif p_ppv_corr[j] <= 0.05:
                ax_ppv[i-1].text(m-0.5, 0.95, '*', color='blue', fontsize=10)
            else:
                ax_ppv[i-1].text(m-1.5, 0.95, 'n.s.', color='blue', fontsize=10)
            # negative predictive value significance labels
            if p_npv[j] <= 0.001:
                ax_npv[i-1].text(m-1.5, 0.9, '***', color='black', fontsize=10)
            elif p_npv[j] <= 0.01:
                ax_npv[i-1].text(m-1, 0.9, '**', color='black', fontsize=10)
            elif p_npv[j] <= 0.05:
                ax_npv[i-1].text(m-0.5, 0.9, '*', color='black', fontsize=10)
            else:
                ax_npv[i-1].text(m-1.5, 0.9, 'n.s.', color='black', fontsize=10)               
            if p_npv_corr[j] <= 0.001:
                ax_npv[i-1].text(m-1.5, 0.95, '***', color='blue', fontsize=10)
            elif p_npv_corr[j] <= 0.01:
                ax_npv[i-1].text(m-1, 0.95, '**', color='blue', fontsize=10)
            elif p_npv_corr[j] <= 0.05:
                ax_npv[i-1].text(m-0.5, 0.95, '*', color='blue', fontsize=10)
            else:
                ax_npv[i-1].text(m-1.5, 0.95, 'n.s.', color='blue', fontsize=10)
            # false discovery rate significance labels
            if p_fdr[j] <= 0.001:
                ax_fdr[i-1].text(m-1.5, 0.9, '***', color='black', fontsize=10)
            elif p_fdr[j] <= 0.01:
                ax_fdr[i-1].text(m-1, 0.9, '**', color='black', fontsize=10)
            elif p_fdr[j] <= 0.05:
                ax_fdr[i-1].text(m-0.5, 0.9, '*', color='black', fontsize=10)
            else:
                ax_fdr[i-1].text(m-1.5, 0.9, 'n.s.', color='black', fontsize=10)               
            if p_fdr_corr[j] <= 0.001:
                ax_fdr[i-1].text(m-1.5, 0.95, '***', color='blue', fontsize=10)
            elif p_fdr_corr[j] <= 0.01:
                ax_fdr[i-1].text(m-1, 0.95, '**', color='blue', fontsize=10)
            elif p_fdr_corr[j] <= 0.05:
                ax_fdr[i-1].text(m-0.5, 0.95, '*', color='blue', fontsize=10)
            else:
                ax_fdr[i-1].text(m-1.5, 0.95, 'n.s.', color='blue', fontsize=10)
        if i==1:
            ax_roc[i-1].legend(frameon=False)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-ROC.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_roc)          
    fig_auc.tight_layout()
    fig_auc.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-AUC.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_auc)          
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-ACC.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_acc)          
    fig_bacc.tight_layout()
    fig_bacc.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-BACC.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_bacc)    
    fig_ppv.tight_layout()  
    fig_ppv.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-PPV.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_ppv)  
    fig_npv.tight_layout() 
    fig_npv.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-NPV.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_npv)   
    fig_fdr.tight_layout()
    fig_fdr.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_p-{plow}_metric-FDR.{args.format}"), dpi=300, format=args.format)
    plt.close(fig_fdr)   
    print("**************************")

for i in tqdm(range(1,len(TDMaps.columns)), desc="Plotting prediction histograms"):
    feature_TDI = TDMaps.columns[i]
    fig, ax = plt.subplots(len(percentiles2check), len(months), figsize=(6*len(months),5*len(percentiles2check)))
    for k, (plow, phigh) in enumerate(percentiles2check):    
        for j, m in enumerate(months):
            death = TRUE_DEATHS[f"({plow}, {phigh})"][feature_TDI][m]
            scaled = TDI_data[f"({plow}, {phigh})"][feature_TDI][m]
            scaled = (scaled - scaled.min())/(scaled.max()-scaled.min())
            fpr, tpr, thresholds = roc_curve(death, scaled)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_th = thresholds[optimal_idx]
            death_preds = np.where(scaled>=optimal_th, 1, 0)

            scaled_survived = scaled[death==0]
            low_scaled_tn = scaled_survived[scaled_survived<=optimal_th]
            low_scaled_fp = scaled_survived[scaled_survived>optimal_th]
            scaled_died = scaled[death==1]
            low_scaled_fn = scaled_died[scaled_died<=optimal_th]
            low_scaled_tp = scaled_died[scaled_died>optimal_th]

            ax[k,j].hist(low_scaled_tn, histtype="bar", density=False, cumulative=False, bins=20, color="forestgreen", label="Survived")
            ax[k,j].hist(low_scaled_fp, histtype="step", density=False, cumulative=False, bins=20, color="forestgreen", linewidth=2)
            ax[k,j].hist(low_scaled_tp, histtype="bar", density=False, cumulative=False, bins=20, color="darkorange", weights=-np.ones_like(low_scaled_tp), label="Died")
            ax[k,j].hist(low_scaled_fn, histtype="step", density=False, cumulative=False, bins=20, color="darkorange", weights=-np.ones_like(low_scaled_fn), linewidth=2)

            ax[k,j].hlines(0, -.05, 1051, linestyle='-', linewidth=1, color='black')
            ax[k,j].vlines(optimal_th, -12, 12, linestyle='--', linewidth=2, color="black")
            ax[k,j].set_xlim([-0.1,1.1])
            
            ax[k,j].set_xticks([])
            ax[k,j].set_yticks([])
            if "lesion" in feature_TDI:
                ax[k,j].set_xlabel("L-TDI (a.u.)", fontsize=15)
            else:
                ax[k,j].set_xlabel("TDI (a.u.)", fontsize=15)
            if j==0:
                ax[k,j].spines[["top", "right", "bottom"]].set_visible(False)
                ax[k,j].set_ylabel(f"Stratification threshold {plow}/{phigh}"+r'$^{th}$'+" percentiles", fontsize=15)
            else:
                ax[k,j].spines[["top", "right", "bottom","left"]].set_visible(False)
            if k==0:
                ax[k,j].set_title(f"Prediction at {m} months", fontweight='bold', fontsize=12)
            ax[k,j].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_metric-histograms_{feature_TDI}.{args.format}"), dpi=300, format=args.format)
    plt.close(fig)

percentile = "(50, 50)"
for i in range(1, len(TDMaps.columns),2):
    fig, ax = plt.subplots(1, len(months), figsize=(6*len(months),5 ))
    tissue1, tissue2 = TDMaps.columns[i], TDMaps.columns[i+1]
    Zs, ps_DL, ps_DL_corrected = [], [], []
    for j, m in enumerate(months):
        death = TRUE_DEATHS[percentile][tissue1][m] # Since the percentile is 50, there is a unique ground truth (no data has been discarded for any of the two features -- not true for the other stratification thresholds)
        # TDI
        tdis_pred = TDI_data[percentile][tissue1][m].to_numpy()
        fpr_tdi, tpr_tdi, thresholds = roc_curve(death, tdis_pred)
        optimal_idx_tdi = np.argmax(tpr_tdi - fpr_tdi)
        optimal_th_tdi = thresholds[optimal_idx_tdi]
        # LDT
        ltdis_pred = TDI_data[percentile][tissue2][m].to_numpy()
        fpr_ltdi, tpr_ltdi, thresholds = roc_curve(death, ltdis_pred)
        optimal_idx_ltdi = np.argmax(tpr_ltdi - fpr_ltdi)
        optimal_th_ltdi = thresholds[optimal_idx_ltdi]
        # De Long test
        DeLong = DeLong_Test(death) 
        Z, pDL = DeLong.delong_roc_test(ltdis_pred, tdis_pred)
        Zs.append(Z), ps_DL.append(pDL)
        # Plot
        ax[j].plot(fpr_ltdi, tpr_ltdi, linestyle='-', linewidth=2.5, label='L-TDI', alpha=.8) # Plot ROC
        ax[j].plot(fpr_ltdi[optimal_idx_ltdi], tpr_ltdi[optimal_idx_ltdi], 'o', markersize=15, color="gray", markerfacecolor=None, markeredgecolor='black', markeredgewidth=2.5, alpha=0.5)
        ax[j].plot([fpr_ltdi[optimal_idx_ltdi],fpr_ltdi[optimal_idx_ltdi]],[fpr_ltdi[optimal_idx_ltdi],tpr_ltdi[optimal_idx_ltdi]], linestyle='--', linewidth=1.5, color="gray") 
        ax[j].plot(fpr_tdi, tpr_tdi, linestyle='-', linewidth=2.5, label='TDI', alpha=.8) # Plot ROC
        ax[j].plot(fpr_tdi[optimal_idx_tdi], tpr_tdi[optimal_idx_tdi], 'o', markersize=15, color="gray", markerfacecolor=None, markeredgecolor='black', markeredgewidth=2.5, alpha=0.5)
        ax[j].plot([fpr_tdi[optimal_idx_tdi],fpr_tdi[optimal_idx_tdi]],[fpr_tdi[optimal_idx_tdi],tpr_tdi[optimal_idx_tdi]], linestyle='--', linewidth=1.5, color="gray")           
        ax[j].plot([0,1],[0,1], linestyle='--', linewidth=.5, color="k")
        
        ax[j].legend(frameon=False)
        ax[j].set_title(f"Death at {m} months", fontweight="bold", fontsize=15)
        ax[j].spines[["top", "right"]].set_visible(False)
        ax[j].set_xlim([-.05,1.05])
        ax[j].set_xticks([0,0.5,1])
        ax[j].set_xlabel("False positive rate (FPR)", fontsize=15)
        ax[j].spines['bottom'].set_bounds(0,1)
        ax[j].set_ylim([0,1])
        ax[j].set_yticks([0,0.5,1])
        ax[j].set_ylabel("True positive rate (TPR)", fontsize=15)
        ax[j].spines['left'].set_bounds(0,1)

        if fwer: # Method: Holm's procedure
            _, ps_DL_corrected, _, _ = multipletests(ps_DL, alpha=0.05, method='holm', is_sorted=False)
        else: # Method: Benjamin-Hochberg
            _, ps_DL_corrected = fdrcorrection(ps_DL, alpha=0.05, method='p', is_sorted=False)

    for j, m in enumerate(months):
        ax[j].text(0.65, 0.15, r"$Z =$"+f"{round(Zs[j],4)} \np = {round(ps_DL[j],4)} \np"+r'$_{corrected}$'+f" = {round(ps_DL_corrected[j],4)}", transform=ax[j].transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if ps_DL[j]<=0.05 else "black")      

    fig.tight_layout()
    fig.savefig(os.path.join(args.path, f"Figures/TDMaps_Grade-IV/{figs_folder}/Death-prediction_model-TDI_metric-DeLong_features-{TDMaps.columns[i].split(" ")[0]}.{args.format}"), dpi=300, format=args.format)
    plt.close(fig)