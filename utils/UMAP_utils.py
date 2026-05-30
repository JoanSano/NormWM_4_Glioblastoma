import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import kendalltau, spearmanr, somersd
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from utils.metrics import compute_quantile_OS

#######################################################################################################
################################## PLOTING ############################################################

def color_topological_risk(labels, cmap="viridis"):
    n = len(np.unique(labels))

    if cmap not in plt.colormaps():
        raise ValueError(f"Unknown colormap '{cmap}'. Try one of: {plt.colormaps()[:10]} ...")

    cmap_obj = plt.get_cmap(cmap)
    return cmap_obj(np.linspace(0, 1, n))

def plot_umap_outcome(
    training_dataframe,
    umap_X_train,
    testing_dataframe,
    umap_X_test,
    daysXmonth,
    colors_groups,
    status="status",
    risk="topological risk",
    survival="OS (days) - corrected",
    months = range(0,110,10),
    cmap="viridis",
    show_median="all",
    plot=True
):
    is_3d = (umap_X_train.shape[1] == 3)
    if (umap_X_train.shape[1] > 3) and (umap_X_test.shape[1] > 3):
        umap_X_train = umap_X_train[:,:3]
        umap_X_test = umap_X_test[:,:3]
        is_3d = True

    # Create figure
    if plot:
        fig = plt.figure(figsize=(20,12))
        gs = GridSpec(3, 4, figure=fig)

        # ---- First row: 4 normal axes ----
        ax0 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 2])
        # Axes: 1 and 3 can be 2D or 3D
        if is_3d:
            ax1 = fig.add_subplot(gs[0, 1], projection="3d")#1, 4, 2, projection="3d")
            ax3 = fig.add_subplot(gs[0, 3], projection="3d")#1, 4, 4, projection="3d")
        else:
            ax1 = fig.add_subplot(gs[0, 1])#1, 4, 2)
            ax3 = fig.add_subplot(gs[0, 3])#1, 4, 4)

        # ---- Second row: 2 wide axes (each spans 2 columns) ----
        ax4 = fig.add_subplot(gs[1:, 0:2])  # spans columns 0 and 1
        ax5 = fig.add_subplot(gs[1:, 2:4])  # spans columns 2 and 3

        ax = [ax0, ax1, ax2, ax3, ax4, ax5]

    ##################################################
    # ------ Recording statistical outcomes -------- # 
    stats = {}
    stats["training cohort(s)"] = {}
    stats["training cohort(s)"]["topological risk"] = {}
    stats["training cohort(s)"]["kaplan-meier"] = {}
    stats["testing cohort(s)"] = {}
    stats["testing cohort(s)"]["topological risk"] = {}
    stats["testing cohort(s)"]["kaplan-meier"] = {}

    groups = np.unique(training_dataframe[risk].values)

    for tr in range(len(groups)): 
        ##############################################
        # -------- TRAIN -------- #
        if show_median == "uncensored":
            mask = (training_dataframe[risk] == tr) & (training_dataframe[status] == 1)
            data = training_dataframe[survival].values[mask] / daysXmonth

            med = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)

        elif show_median == "all":
            mask = (training_dataframe[risk] == tr)
            data = training_dataframe[survival].values[mask] / daysXmonth
            status_data = training_dataframe[status].values[mask]
            med, q1, q3 = compute_quantile_OS([0.5, 0.75, 0.25], status_data, data)
        
        else:
            raise ValueError("Show median option undefined")
        
        yerr = np.array([[med - q1], [q3 - med]])
        stats["training cohort(s)"]["topological risk"][tr] = [med, q1, q3]

        if plot:
            ax[0].barh(
                tr, med, xerr=yerr, capsize=5,
                color=colors_groups[tr], alpha=.75,
                edgecolor="black", linewidth=2
            )
            ax[0].plot(med, tr, "s", markeredgecolor="black",
                    markerfacecolor="white", markersize=10, markeredgewidth=2.5)
            ax[0].spines[["left","top","right"]].set_visible(False)
            ax[0].set_yticks([])

        ##########################
        # -------- TEST -------- #
        if show_median == "uncensored":
            mask = (testing_dataframe[risk] == tr) & (testing_dataframe[status] == 1)
            data = testing_dataframe[survival].values[mask] / daysXmonth

            med = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)

        elif show_median == "all":
            mask = (testing_dataframe[risk] == tr)
            data = testing_dataframe[survival].values[mask] / daysXmonth
            status_data = testing_dataframe[status].values[mask]
            med, q1, q3 = compute_quantile_OS([0.5, 0.75, 0.25], status_data, data)
        else:
            raise ValueError("Show median option undefined")

        yerr = np.array([[med - q1], [q3 - med]])
        stats["testing cohort(s)"]["topological risk"][tr] = [med, q1, q3]

        if plot:
            ax[2].barh(
                tr, med, xerr=yerr, capsize=5,
                color=colors_groups[tr], alpha=.75,
                edgecolor="black", linewidth=2
            )
            ax[2].plot(med, tr, "s", markeredgecolor="black",
                    markerfacecolor="white", markersize=10, markeredgewidth=2.5)
            ax[2].spines[["left","top","right"]].set_visible(False)
            ax[2].set_yticks([])


    if plot:
        if show_median == "uncensored":
            ax[0].set_xlabel("Median OS (months; status: dead)")
        elif show_median == "all":
            ax[0].set_xlabel("Median OS (months)")
        else:
            raise ValueError("Show median option undefined")
        ax[0].set_title("Training cohort(s)", fontweight="bold")
    
    ################################
    # -------- UMAP TRAIN -------- #
    if plot:
        if is_3d:
            ax[1].scatter(
                umap_X_train[:,0], umap_X_train[:,1], umap_X_train[:,2],
                c=training_dataframe[risk], alpha=.75, cmap=cmap, edgecolors="black" if "gray" in cmap else None, linewidths=0.5 if "gray" in cmap else None
            )
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_zticks([])
            ax[1].set_xlabel("UMAP 1")
            ax[1].set_ylabel("UMAP 2")
            ax[1].set_zlabel("UMAP 3")
        else:
            ax[1].scatter(
                umap_X_train[:,0], umap_X_train[:,1],
                c=training_dataframe[risk], alpha=.75, cmap=cmap, edgecolors="black" if "gray" in cmap else None, linewidths=0.5 if "gray" in cmap else None
            )
            ax[1].spines[["top","right"]].set_visible(False)
            ax[1].set_xlim([umap_X_train[:,0].min()-1, umap_X_train[:,0].max()+2])
            ax[1].spines["bottom"].set_bounds(umap_X_train[:,0].min()-1, umap_X_train[:,0].min()+5)
            ax[1].set_ylim([umap_X_train[:,1].min()-1, umap_X_train[:,1].max()+1])
            ax[1].spines["left"].set_bounds(umap_X_train[:,1].min()-1, umap_X_train[:,1].min()+3)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_xlabel("UMAP 1", loc="left")
            ax[1].set_ylabel("UMAP 2", loc="bottom")

        ax[2].set_xlabel("Median OS (months; status: dead)")
        ax[2].set_title("Testing cohort(s)", fontweight="bold")

        #######################################
        # -------- UMAP TRAIN + TEST -------- #
        if is_3d:
            ax[3].scatter(
                umap_X_train[:,0], umap_X_train[:,1], umap_X_train[:,2],
                c="black", s=2, alpha=.25, label="Training samples"
            )
            sc = ax[3].scatter(
                umap_X_test[:,0], umap_X_test[:,1], umap_X_test[:,2],
                c=testing_dataframe[risk], alpha=.75, cmap=cmap, edgecolors="black" if "gray" in cmap else None, linewidths=0.5 if "gray" in cmap else None
            )
            ax[3].set_xlabel("UMAP 1")
            ax[3].set_ylabel("UMAP 2")
            ax[3].set_zlabel("UMAP 3")
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[3].set_zticks([])
        else:
            ax[3].scatter(
                umap_X_train[:,0], umap_X_train[:,1],
                c="black", s=2, alpha=.25, label="Training samples"
            )
            sc = ax[3].scatter(
                umap_X_test[:,0], umap_X_test[:,1],
                c=testing_dataframe[risk], alpha=.75, cmap=cmap, edgecolors="black" if "gray" in cmap else None, linewidths=0.5 if "gray" in cmap else None
            )
            ax[3].spines[["top","right"]].set_visible(False)
            ax[3].set_xlim([umap_X_train[:,0].min()-1, umap_X_train[:,0].max()+2])
            ax[3].spines["bottom"].set_bounds(umap_X_train[:,0].min()-1, umap_X_train[:,0].min()+5)
            ax[3].set_ylim([umap_X_train[:,1].min()-1, umap_X_train[:,1].max()+1])
            ax[3].spines["left"].set_bounds(umap_X_train[:,1].min()-1, umap_X_train[:,1].min()+3)
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[3].set_xlabel("UMAP 1", loc="left")
            ax[3].set_ylabel("UMAP 2", loc="bottom")

        ax[3].legend(frameon=True, loc="upper left")
        cbar = fig.colorbar(sc, ax=ax[3], shrink=0.7, pad=0.2, orientation="horizontal", location="bottom")
        cbar.set_label("Topological risk")
        cbar.set_ticks(groups)
    ##############################################

    ##############################################
    ## ------- Training KM curves ------------- ##
    ##############################################
    OS_STATS = []
    GROUP_STATS = []

    ## Low topological risk
    top_risk_low = np.min(groups)
    invalid = (training_dataframe[status].isnull()) * (training_dataframe[survival].isnull())
    low_risk = (training_dataframe[risk] == top_risk_low) * (~invalid)
    os_low_risk = training_dataframe[survival].values[low_risk] / daysXmonth
    status_low_risk =  training_dataframe[status].values[low_risk]
    time_low_risk, survival_prob_low_risk, conf_int_low_risk = kaplan_meier_estimator(
        status_low_risk==1, 
        os_low_risk, 
        conf_type="log-log"
    )
    time_low_risk = np.insert(time_low_risk, 0, 0)
    survival_prob_low_risk = np.insert(survival_prob_low_risk, 0, 1)
    conf_int_low_risk = np.insert(conf_int_low_risk, 0, 1, axis=1)
    if plot:
        ax[4].step(time_low_risk, survival_prob_low_risk, where="post",  linewidth=2, color=colors_groups[top_risk_low], label="Low topological risk")
        ax[4].fill_between(time_low_risk, conf_int_low_risk[0], conf_int_low_risk[1], alpha=0.10, step="post", color=colors_groups[top_risk_low], edgecolor="black" if "gray" in cmap else None, linewidth=2 if "gray" in cmap else None)

    ## High topological risk
    top_risk_high = np.max(groups)
    invalid = (training_dataframe[status].isnull()) * (training_dataframe[survival].isnull())
    high_risk = (training_dataframe[risk] == top_risk_high) * (~invalid)
    os_high_risk = training_dataframe[survival].values[high_risk] / daysXmonth
    status_high_risk =  training_dataframe[status].values[high_risk]
    time_high_risk, survival_prob_high_risk, conf_int_high_risk = kaplan_meier_estimator(
        status_high_risk==1, 
        os_high_risk, 
        conf_type="log-log"
    )
    time_high_risk = np.insert(time_high_risk, 0, 0)
    survival_prob_high_risk = np.insert(survival_prob_high_risk, 0, 1)
    conf_int_high_risk = np.insert(conf_int_high_risk, 0, 1, axis=1)    
    if plot:
        ax[4].step(time_high_risk, survival_prob_high_risk, where="post",  linewidth=2, color="gray" if "gray" in cmap else colors_groups[top_risk_high], label="High topological risk", linestyle="dashed" if "gray" in cmap else "-")
        ax[4].fill_between(time_high_risk, conf_int_high_risk[0], conf_int_high_risk[1], alpha=0.10, step="post", color=colors_groups[top_risk_high], edgecolor="gray" if "gray" in cmap else None, linewidth=2 if "gray" in cmap else None)
    
        ### Censoring times
        for t_low, t_high in zip(os_low_risk[status_low_risk==0], os_high_risk[status_high_risk==0]): # Censoring times
            ax[4].plot(time_low_risk[time_low_risk==t_low], survival_prob_low_risk[time_low_risk==t_low], "|", color=colors_groups[top_risk_low])
            ax[4].plot(time_high_risk[time_high_risk==t_high], survival_prob_high_risk[time_high_risk==t_high], "|", color="gray" if "gray" in cmap else colors_groups[top_risk_high])

        ### Number at risk
        ax[4].text(-2, -0.01, "No. at risk", transform=ax[4].transData, fontsize=11, verticalalignment='top', color="black", fontweight='bold') 
        for i,t in enumerate(months):
            ax[4].text(t-2, -0.07, f"{(os_low_risk>=t).sum()}", transform=ax[4].transData, fontsize=11, verticalalignment='top', color=colors_groups[top_risk_low]) 
            ax[4].text(t-2, -0.13, f"{(os_high_risk>=t).sum()}", transform=ax[4].transData, fontsize=11, verticalalignment='top', color="gray" if "gray" in cmap else colors_groups[top_risk_high])

    ### Log-rank test
    OS_STATS.extend([(st, ovs) for st, ovs in zip(status_low_risk==1, os_low_risk)])
    GROUP_STATS.extend([1 for ovs in os_low_risk])
    OS_STATS.extend([(st, ovs) for st, ovs in zip(status_high_risk==1, os_high_risk)])
    GROUP_STATS.extend([2 for ovs in os_high_risk])
    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
    chisquared, p_val, _, _ = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
    stats["training cohort(s)"]["kaplan-meier"] = {"chi-squared": chisquared, "p-value": p_val}
    if plot:
        tx = "<0.0001" if p_val<0.0001 else round(p_val,4)
        ax[4].text(0.85, 0.85, r"$\chi^2 =$"+f"{round(chisquared,4)} \np = {tx}", transform=ax[4].transAxes, 
                            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val<=0.05 else "black")

        ax[4].hlines(0,-5,months[-1]+5, color="black", linewidth=.5)
        ax[4].set_ylim([-.2,1.1])
        ax[4].set_xlim([-5,75])
        ax[4].set_xticks(range(0,months[-1]+10,10))
        ax[4].set_xticklabels(range(0,months[-1]+10,10))
        ax[4].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[4].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
        ax[4].spines['left'].set_bounds(0,1)
        ax[4].spines['bottom'].set_bounds(0,months[-1])
        ax[4].set_xlabel("Time (months)", fontsize=12)
        ax[4].set_ylabel("Overall survival", fontsize=12)
        ax[4].spines[["top", "right"]].set_visible(False)
        ax[4].legend(frameon=False)
    ##############################################

    #############################################
    ## ------- Testing KM curves ------------- ##
    #############################################
    OS_STATS = []
    GROUP_STATS = []

    ## Low topological risk
    top_risk_low = np.min(groups)
    invalid = (testing_dataframe[status].isnull()) * (testing_dataframe[survival].isnull())
    low_risk = (testing_dataframe[risk] == top_risk_low) * (~invalid)
    os_low_risk = testing_dataframe[survival].values[low_risk] / daysXmonth
    status_low_risk =  testing_dataframe[status].values[low_risk]
    time_low_risk, survival_prob_low_risk, conf_int_low_risk = kaplan_meier_estimator(
        status_low_risk==1, 
        os_low_risk, 
        conf_type="log-log"
    )
    time_low_risk = np.insert(time_low_risk, 0, 0)
    survival_prob_low_risk = np.insert(survival_prob_low_risk, 0, 1)
    conf_int_low_risk = np.insert(conf_int_low_risk, 0, 1, axis=1)
    if plot:
        ax[5].step(time_low_risk, survival_prob_low_risk, where="post",  linewidth=2, color=colors_groups[top_risk_low], label="Low topological risk")
        ax[5].fill_between(time_low_risk, conf_int_low_risk[0], conf_int_low_risk[1], alpha=0.10, step="post", color=colors_groups[top_risk_low], edgecolor="black" if "gray" in cmap else None, linewidth=2 if "gray" in cmap else None)

    ## High topological risk
    top_risk_high = np.max(groups)
    invalid = (testing_dataframe[status].isnull()) * (testing_dataframe[survival].isnull())
    high_risk = (testing_dataframe[risk] == top_risk_high) * (~invalid)
    os_high_risk = testing_dataframe[survival].values[high_risk] / daysXmonth
    status_high_risk =  testing_dataframe[status].values[high_risk]
    time_high_risk, survival_prob_high_risk, conf_int_high_risk = kaplan_meier_estimator(
        status_high_risk==1, 
        os_high_risk, 
        conf_type="log-log"
    )
    time_high_risk = np.insert(time_high_risk, 0, 0)
    survival_prob_high_risk = np.insert(survival_prob_high_risk, 0, 1)
    conf_int_high_risk = np.insert(conf_int_high_risk, 0, 1, axis=1)
    if plot:
        ax[5].step(time_high_risk, survival_prob_high_risk, where="post",  linewidth=2, color="gray" if "gray" in cmap else colors_groups[top_risk_high], label="High topological risk", linestyle="dashed" if "gray" in cmap else "-")
        ax[5].fill_between(time_high_risk, conf_int_high_risk[0], conf_int_high_risk[1], alpha=0.10, step="post", color=colors_groups[top_risk_high], edgecolor="gray" if "gray" in cmap else None, linewidth=2 if "gray" in cmap else None)
        
        ### Censoring times
        for t_low, t_high in zip(os_low_risk[status_low_risk==0], os_high_risk[status_high_risk==0]): # Censoring times
            ax[5].plot(time_low_risk[time_low_risk==t_low], survival_prob_low_risk[time_low_risk==t_low], "|", color=colors_groups[top_risk_low])
            ax[5].plot(time_high_risk[time_high_risk==t_high], survival_prob_high_risk[time_high_risk==t_high], "|", color="gray" if "gray" in cmap else colors_groups[top_risk_high])

        ### Number at risk
        ax[5].text(-2, -0.01, "No. at risk", transform=ax[5].transData, fontsize=11, verticalalignment='top', color="black", fontweight='bold') 
        for i,t in enumerate(months):
            ax[5].text(t-2, -0.07, f"{(os_low_risk>=t).sum()}", transform=ax[5].transData, fontsize=11, verticalalignment='top', color=colors_groups[top_risk_low]) 
            ax[5].text(t-2, -0.13, f"{(os_high_risk>=t).sum()}", transform=ax[5].transData, fontsize=11, verticalalignment='top', color="gray" if "gray" in cmap else colors_groups[top_risk_high])

    ### Log-rank test
    OS_STATS.extend([(st, ovs) for st, ovs in zip(status_low_risk==1, os_low_risk)])
    GROUP_STATS.extend([1 for ovs in os_low_risk])
    OS_STATS.extend([(st, ovs) for st, ovs in zip(status_high_risk==1, os_high_risk)])
    GROUP_STATS.extend([2 for ovs in os_high_risk])
    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
    chisquared, p_val, _, _ = compare_survival(OS_STATS, GROUP_STATS, return_stats=True)
    stats["testing cohort(s)"]["kaplan-meier"] = {"chi-squared": chisquared, "p-value": p_val}
    if plot:
        tx = "<0.0001" if p_val<0.0001 else round(p_val,4)
        ax[5].text(0.85, 0.85, r"$\chi^2 =$"+f"{round(chisquared,4)} \np = {tx}", transform=ax[5].transAxes, 
                            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1), color="red" if p_val<=0.05 else "black")

        ax[5].hlines(0,-5,months[-1]+5, color="black", linewidth=.5)
        ax[5].set_ylim([-.2,1.1])
        ax[5].set_xlim([-5,75])
        ax[5].set_xticks(range(0,months[-1]+10,10))
        ax[5].set_xticklabels(range(0,months[-1]+10,10))
        ax[5].set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax[5].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
        ax[5].spines['left'].set_bounds(0,1)
        ax[5].spines['bottom'].set_bounds(0,months[-1])
        ax[5].set_xlabel("Time (months)", fontsize=12)
        ax[5].set_ylabel("Overall survival", fontsize=12)
        ax[5].spines[["top", "right"]].set_visible(False)
        ax[5].legend(frameon=False)
    #############################################

    return fig if plot else None, stats

#######################################################################################################
################################## RISK MAPPERS #######################################################

def median_uncensored_mapper(labels, os, status, mask):
    ### BASED ON MEDIAN SURVIVAL OF UNCENSORED PATIENTS
    #       to map an increased risk based on the median survival of each of the found groups
    Ngroups = np.unique(labels)
    labels, os = labels[mask], os[mask]
    OS_criterion = np.array([np.median(os[labels==l]) for l in Ngroups])
    return {int(l): int(g) for (l,g) in zip(OS_criterion.argsort(), Ngroups[::-1])} 

def mean_uncensored_mapper(labels, os, status, mask):
    ### BASED ON MEAN SURVIVAL OF UNCENSORED PATIENTS
    #       to map an increased risk based on the median survival of each of the found groups
    Ngroups = np.unique(labels)
    labels, os = labels[mask], os[mask]
    OS_criterion = np.array([np.mean(os[labels==l]) for l in Ngroups])
    return {int(l): int(g) for (l,g) in zip(OS_criterion.argsort(), Ngroups[::-1])}

def median_OS_mapper(labels, os, status, mask):
    ### BASED ON MEADIAN SURVIVAL OF THE SURVIVAL PROBABILITY
    #       to map an increased risk based on the median survival of each of the found groups
    Ngroups = np.unique(labels)
    OS_criterion = np.zeros((len(Ngroups), ))
    for l in Ngroups:
        OS_criterion[l] = compute_quantile_OS([0.5], status[labels==l]==1, os[labels==l])[0]
    return {int(l): int(g) for (l,g) in zip(OS_criterion.argsort(), Ngroups[::-1])}

def map_to_topological_risk(
        method="median-os", 
        **kwargs
    ):
    methods = {
        "median-os": median_OS_mapper,
        "median-uncensored": median_uncensored_mapper,
        "mean-uncensored": mean_uncensored_mapper,
    }

    if method not in methods:
        raise ValueError(
            f"Method '{method}' not implemented. Available: {list(methods.keys())}"
        )

    return methods[method](**kwargs)


#######################################################################################################
################################## TOPOLOGICAL RISK ###################################################

def compute_topological_risk(
    umap_args,
    data_train,
    data_test,
    clusterizer,
    mask_train,
    os_train,
    status_train,
    method="median-uncensored",
    cmap="viridis"
):
    ###############################
    ### UMAP STEP
    umap_model = umap.UMAP(**umap_args)
    umap_X_train = umap_model.fit_transform(data_train)
    umap_X_test = umap_model.transform(data_test)

    ###############################
    ### CLUSTERING STEP
    labels_train = clusterizer.fit_predict(umap_X_train)
    labels_test = clusterizer.predict(umap_X_test)

    ###############################
    ### TOPOLOGICAL RISK ASSIGNMENT
    mapper = map_to_topological_risk(method, labels=labels_train, os=os_train, status=status_train, mask=mask_train)
    topological_risk_train = np.array([mapper[l] for l in labels_train])    
    topological_risk_test = np.array([mapper[l] for l in labels_test])   

    ####################################
    ### VISUALIZATION AND CORRESPONDANCE
    colors_groups = color_topological_risk(labels_train, cmap=cmap)
    
    return {
        "mapper": mapper,
        "topological risk: train": topological_risk_train,
        "topological risk: test": topological_risk_test,
        "umap coordinates: train": umap_X_train,
        "umap coordinates: test": umap_X_test,
        "color risk": colors_groups, 
        "cluster labels: train": labels_train,
        "cluster labels: test": labels_test,
    }

#######################################################################################################
#######################################################################################################

if __name__ == "__main__":
    pass