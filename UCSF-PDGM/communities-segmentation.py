import glob
import os
from tqdm import tqdm
import argparse
import nilearn.plotting as plotting
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
import networkx as nx   
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

#from Qommunity.samplers.hierarchical.advantage_sampler import AdvantageSampler
#from Qommunity.samplers.hierarchical.gurobi_sampler import GurobiSampler
from Qommunity.samplers.regular.leiden_sampler import LeidenSampler
from Qommunity.samplers.regular.louvain_sampler import LouvainSampler
from iterative_searcher.iterative_searcher import IterativeSearcher

parser = argparse.ArgumentParser()
parser.add_argument("--mni_dir", type=str, help="Location of the MNI registration template", default="/home/sano/Documents/Joan/Data/MNI_ICBM_2009b_NLIN_ASYM")
parser.add_argument("--th", type=float, help="Threshold to count the overlap as real", default=0.1)
parser.add_argument("--tissue", type=str, help="Tissue type to analyze", choices=["Whole-tumor", "Non-enhancing", "Enhancing", "Core"], default="Whole-tumor")
parser.add_argument("--min_size_community", type=int, help="Minimum size of the community to include in the analysis", default=20)
parser.add_argument("--threshold", type=int, help="Threshold to plot the lesions", default=1)
parser.add_argument("--plot_comms", action="store_true", help="Whether to plot the communities")
parser.add_argument("--algorithm", type=str, choices=["Louvain", "Leiden"], default="Louvain", help="Algorithm to check")
parser.add_argument("--metric", type=str, choices=["Jaccard", "Dice"], default="Jaccard", help="Similarity metric to check")
args = parser.parse_args()

os.makedirs(f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}", exist_ok=True) 
th = args.th
tissue = args.tissue
min_size_community = args.min_size_community
threshold = args.threshold 
plot_comms = args.plot_comms

def rewrite_subjectID(subject_ID, digits_to_write=3):
    if digits_to_write == 3:
        subject_4digits = subject_ID.split("-")
        subject = "-".join(subject_4digits[:-1])
        digits = str(subject_4digits[-1][1:])
        return subject + "-" + digits
    elif digits_to_write == 4:
        subject_3digits = subject_ID.split("-")
        subject = "-".join(subject_3digits[:-1])
        digits = str(subject_3digits[-1]).zfill(4)
        return subject + "-" + digits
    else:
        raise ValueError("Only rewrites in 3 or 4 digits")
    
# Function to compute the Jaccard index
def jaccard_index(img1, img2):
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    return intersection / union if union != 0 else 0

# Function to compute the Dice score
def dice_score(img1, img2):
    intersection = np.logical_and(img1, img2).sum()
    return 2 * intersection / (img1.sum() + img2.sum()) if (img1.sum() + img2.sum()) != 0 else 0

def extract_tissue(img, tissue):
    if tissue=="Whole-tumor":
        return np.where(img>=1, 1, 0)
    elif tissue=="Enhancing":
        return np.where(img==4, 1, 0)
    elif tissue=="Non-enhancing": # Synonim for edema
        return np.where(img==2, 1, 0)
    elif tissue=="Core":
        return np.where(img==1, 1, 0)
    else:
        raise ValueError("Invalid type of tumor tissue provided")

def plot_lesions(lesions, mni_img, fig_folder, title, Nslices=20, cmap="Reds", threshold=1):
    fig, ax = plt.subplots(3, 1, figsize=(20, 5), facecolor="k")
    fig.subplots_adjust(hspace=0, left=0, right=1, top=1, bottom=0)
    lesions = np.where(lesions>=threshold, lesions, 0)
    max_overlap = int(lesions.max())
    lesions = nib.Nifti1Image(lesions, mni_img.affine, mni_img.header)

    display_x = plotting.plot_anat(
        mni_img, 
        axes=ax[0], 
        display_mode="x", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_x.add_overlay(
        lesions,
        cmap = cmap,
        colorbar=True,
        cbar_vmin=threshold,
        cbar_vmax=max_overlap,
        cbar_tick_format="%i"
    )

    display_y = plotting.plot_anat(
        mni_img, 
        axes=ax[1], 
        display_mode="y", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_y.add_overlay(
        lesions,
        cmap = cmap,
        colorbar=False,
        cbar_vmin=threshold,
        cbar_vmax=max_overlap,
        cbar_tick_format="%i"
    )

    display_z = plotting.plot_anat(
        mni_img, 
        axes=ax[2], 
        display_mode="z", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_z.add_overlay(
        lesions,
        cmap = cmap,
        colorbar=False,
        cbar_vmin=threshold,
        cbar_vmax=max_overlap,
        cbar_tick_format="%i"
    )

    fig.suptitle(title, fontweight='bold', color='white', fontsize=20)
    fig.savefig(f"{fig_folder}/{title}.pdf", format='pdf', dpi=100)
    plt.close()

if __name__ == "__main__":  
    # Segmentations
    lesions_files = glob.glob("../*segmentation*/*/*")
    N = len(lesions_files)

    # Load data
    try:
        jaccard = np.load(f"../data/Arrays-NPY_tumor-segmentations/jaccard_{tissue}.npy")
        dice = np.load(f"../data/Arrays-NPY_tumor-segmentations/dice_{tissue}.npy")
        names = np.load(f"../data/Arrays-NPY_tumor-segmentations/ordered-names_{tissue}.npy", allow_pickle=True)
        print("Spatial overlap precomputed, loading data ...")
    except:
        print("Spatial overlap NOT found, computing ...")
        print("Calculating ...")
        jaccard = np.zeros((N,N))
        dice = np.zeros((N,N))
        data_redy = False

        # Create ordered list of subjects for reproducibility
        os.makedirs("../data/Arrays-NPY_tumor-segmentations", exist_ok=True)
        names = []
        for i,f in tqdm(enumerate(lesions_files)):
            name = f.split("/")[-1].split(".")[0]
            names.append(name)
            if not os.path.exists(f"../data/Arrays-NPY_tumor-segmentations/{name}.npy"):
                img = nib.load(f).get_fdata()
                np.save(f"../data/Arrays-NPY_tumor-segmentations/{name}.npy", img, allow_pickle=True)
        np.save(f"../data/Arrays-NPY_tumor-segmentations/ordered-names_{tissue}.npy", names, allow_pickle=True)

        # Compute overlaps
        """
        INFO: Each integer is associated with a tumor tissue
            whole_tumor --> label>=1
            enhancing_tumor --> label==4
            nonenhancing_tumor --> label==2
            core_tumor --> label==1
        """
        for i, f1 in tqdm(enumerate(names), desc="First image"):
            f1 = f"../data/Arrays-NPY_tumor-segmentations/{f1}.npy"
            img1 = np.load(f1, allow_pickle=True)
            img1 = extract_tissue(img1, tissue)
            for j in range(i+1,N):
                f2 = names[j]
                f2 = f"../data/Arrays-NPY_tumor-segmentations/{f2}.npy"
                img2 = np.load(f2, allow_pickle=True)
                img2 = extract_tissue(img2, tissue)
                jaccard[i, j] = jaccard_index(img1, img2)
                dice[i, j] = dice_score(img1, img2)
        jaccard = jaccard + jaccard.T
        dice = dice + dice.T
        np.save(f"../data/Arrays-NPY_tumor-segmentations/jaccard_{tissue}.npy", jaccard)
        np.save(f"../data/Arrays-NPY_tumor-segmentations/dice_{tissue}.npy", dice)
        print("Done!")

    # Prepare demographics following the logs file from the registration
    try: 
        demographics = pd.read_csv(f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}/UCSF-PDGM-metadata_v4_{tissue}_th-{th}.csv")
        print("Updated demographics available. Loading ...")
    except:
        print("Updating demographics with spatial overlaps.")
        demographics = pd.read_csv(f"../data/UCSF-PDGM-metadata_v3.csv")
        logs_msg = ""
        with open("../Figures/QC_registration-MNI/Logs-registration.txt", "r") as logs:
            for i, l in enumerate(logs.readlines()):
                logs_msg += l
                if i>=2:
                    # Delete the row directly in the DataFrame
                    demographics.drop(demographics[demographics["ID"] == rewrite_subjectID(l.split(" ")[0][:-1])].index, inplace=True)
        print(logs_msg)

        ### COMMUNITY DETECTION BASED ON SPATIAL LOCATIONS ###
        # Prepping community detection
        G_jaccard = nx.from_numpy_array(np.where(jaccard>=th, 1, 0), create_using=nx.Graph)  # Already weighted graphs
        G_dice = nx.from_numpy_array(np.where(dice>=th, 1, 0), create_using=nx.Graph)

        # Parameters
        resolution = 1
        num_runs = 50

        # Louvain - Jaccard
        print("Communities using the Louvain algorithm (Jaccard):")
        louv_sampler_J = LouvainSampler(G_jaccard, resolution=resolution)
        louv_cs_J, louv_mod_J, _ = IterativeSearcher(louv_sampler_J).run(num_runs=num_runs, save_results=False)
        max_louv_mod_J, max_louv_cs_J = louv_mod_J.max(), louv_cs_J[louv_mod_J.argmax()]
        print(max_louv_mod_J, len(max_louv_cs_J))

        # Leiden - Jaccard
        print("Communities using the Leiden algorithm (Jaccard):")
        leid_sampler_J = LeidenSampler(G_jaccard, resolution=resolution)
        leid_cs_J, leid_mod_J, _ = IterativeSearcher(leid_sampler_J).run(num_runs=num_runs, save_results=False)
        max_leid_mod_J, max_leid_cs_J = leid_mod_J.max(), leid_cs_J[leid_mod_J.argmax()]
        print(max_leid_mod_J, len(max_leid_cs_J))

        """ # Quantum Hierarchical (H. annealing algorithm) for Jaccard
        print("Communities using the H. annealing algorithm (Jaccard):")
        adv_sampler_J = AdvantageSampler(G_jaccard, resolution=resolution, num_reads=100, use_clique_embedding=False)
        adv_cs_J, adv_mod_J, _ = IterativeSearcher(adv_sampler_J).run(num_runs=num_runs, save_results=False)
        print(adv_mod_J.max(), len(adv_cs_J[adv_mod_J.argmax()]))

        # Gurobi Hierarchical (H. classical algorithm) for Jaccard
        print("Communities using the H. classical algorithm (Jaccard):")
        g_sampler_J = GurobiSampler(G_jaccard, resolution=resolution)
        g_cs_J, g_mod_J, _ = IterativeSearcher(g_sampler_J).run(num_runs=num_runs, save_results=False)
        print(g_mod_J.max(), len(g_cs_J[g_mod_J.argmax()])) """

        # Louvain - Dice
        print("Communities using the Louvain algorithm (Dice):")
        louv_sampler_D = LouvainSampler(G_dice, resolution=resolution)
        louv_cs_D, louv_mod_D, _ = IterativeSearcher(louv_sampler_D).run(num_runs=num_runs, save_results=False)
        max_louv_mod_D, max_louv_cs_D = louv_mod_D.max(), louv_cs_D[louv_mod_D.argmax()]
        print(max_louv_mod_D, len(max_louv_cs_D))

        # Leiden - Dice
        print("Communities using the Leiden algorithm (Dice):")
        leid_sampler_D = LeidenSampler(G_dice, resolution=resolution)
        leid_cs_D, leid_mod_D, _ = IterativeSearcher(leid_sampler_D).run(num_runs=num_runs, save_results=False)
        max_leid_mod_D, max_leid_cs_D = leid_mod_D.max(), leid_cs_D[leid_mod_D.argmax()]
        print(max_leid_mod_D, len(max_leid_cs_D))

        """ # Quantum Hierarchical (H. annealing algorithm) for Dice
        print("Communities using the H. annealing algorithm (Dice):")
        adv_sampler_D = AdvantageSampler(G_dice, resolution=resolution, num_reads=100, use_clique_embedding=False)
        adv_cs_D, adv_mod_D, _ = IterativeSearcher(adv_sampler_D).run(num_runs=num_runs, save_results=False)
        print(adv_mod_D.max(), len(adv_cs_D[adv_mod_D.argmax()]))

        # Gurobi Hierarchical (H. classical algorithm) for Dice
        print("Communities using the H. classical algorithm (Dice):")
        g_sampler_D = GurobiSampler(G_dice, resolution=resolution)
        g_cs_D, g_mod_D, _ = IterativeSearcher(g_sampler_D).run(num_runs=num_runs, save_results=False)
        print(g_mod_D.max(), len(g_cs_D[g_mod_D.argmax()])) """

        # Dropping results
        demographics["Community Louvain (Jaccard)"] = ""
        demographics["Community Leiden (Jaccard)"] = ""
        demographics["Community Louvain (Dice)"] = ""
        demographics["Community Leiden (Dice)"] = ""

        # Assigning community labels for Jaccard graph
        for i, cLouv in enumerate(max_louv_cs_J):
            for j in cLouv:
                name = names[j]
                ID = rewrite_subjectID(name.split("/")[-1].split("_")[0])
                demographics.loc[demographics["ID"] == ID, "Community Louvain (Jaccard)"] = i + 1

        for i, cLeid in enumerate(max_leid_cs_J):
            for j in cLeid:
                name = names[j]
                ID = rewrite_subjectID(name.split("/")[-1].split("_")[0])
                demographics.loc[demographics["ID"] == ID, "Community Leiden (Jaccard)"] = i + 1

        # Assigning community labels for Dice graph
        for i, cLouv in enumerate(max_louv_cs_D):
            for j in cLouv:
                name = names[j]
                ID = rewrite_subjectID(name.split("/")[-1].split("_")[0])
                demographics.loc[demographics["ID"] == ID, "Community Louvain (Dice)"] = i + 1

        for i, cLeid in enumerate(max_leid_cs_D):
            for j in cLeid:
                name = names[j]
                ID = rewrite_subjectID(name.split("/")[-1].split("_")[0])
                demographics.loc[demographics["ID"] == ID, "Community Leiden (Dice)"] = i + 1

        # Saving the results
        demographics.to_csv(f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}/UCSF-PDGM-metadata_v4_{tissue}_th-{th}.csv", sep=",")

    # Seggregate OS by spatial location
    T1 = nib.load(os.path.join(args.mni_dir,"T1_0.5mm.nii"))
    metric_name = f"Community {args.algorithm} ({args.metric})"
    cmaps = ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"]
    colors = ["black","purple","blue","green","orange","red","salmon","cyan","royalblue"]
    k = 0

    fig1, ax1 = plt.subplots(1,1)

    Cs = np.array(demographics[metric_name])
    NCs = np.unique(Cs)
    OS = np.array(demographics["OS"])
    SUBJECTS = np.array(demographics["ID"])
    cs_file = f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}/Communities-tumors_{tissue}_th-{th}_minsize-{min_size_community}.npy"
    if os.path.exists(cs_file) and (plot_comms is True):
        lesion_communities = np.load(cs_file, allow_pickle=True)[()]
        comms_ready = True
        print("The spatial overlap computed with the desired parameters was already available. Loading ...")
    else:
        lesion_communities = {}
        comms_ready = False
        print("The spatial overlap computed with the desired parameters was NOT already available.")
    OS_STATS = []
    CS_STATS = []

    for i,cn in enumerate(NCs):
        OS_C = OS[Cs==cn]
        OS_C = OS_C[~np.isnan(OS_C)]
        
        # Only plot sufficiently big communities
        if len(OS_C)>=min_size_community: 
            # Overall survivale for the given community
            time, survival_prob, conf_int = kaplan_meier_estimator(
                OS_C>0, OS_C, conf_type="log-log"
            )
            ax1.step(time, survival_prob, where="post", label=f"C{cn}", color=colors[k])
            ax1.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post", color=colors[k])

            if plot_comms:
                # Spatial position of the given community
                if comms_ready is False:
                    mask = np.zeros(T1.get_fdata().shape)
                    for j, sid in enumerate(SUBJECTS[Cs==cn]):
                        # Here we can load numpy arrays instead of nifti images and it should be faster
                        data_lesion = nib.load(
                            glob.glob(f"../UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM_segmentation/*/{rewrite_subjectID(sid, 4)}*")[0]
                        ).get_fdata()
                        mask += extract_tissue(data_lesion, tissue)                
                    lesion_communities[cn] = mask
                
                # Plot the spatial location the given community
                plot_lesions(
                    lesion_communities[cn], 
                    T1, 
                    f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}", 
                    title=tissue+f"_community-{cn}_th-{th}_minsize-{min_size_community}", 
                    Nslices=15,
                    cmap=cmaps[k],
                    threshold=threshold
                )            
            k += 1
            OS_STATS.extend([(True, float(day)) for day in OS_C])
            CS_STATS.extend([k for day in OS_C])

    OS_STATS = np.array(OS_STATS, dtype=[('event', 'bool'),('time', 'float')])
    chisquared, p_val, stats, covariance = compare_survival(OS_STATS, CS_STATS, return_stats=True)

    ax1.spines[["top","right"]].set_visible(False)
    ax1.set_ylim([0,1])
    ax1.set_ylabel(r"Probability of survival $\hat{S}(t)$")
    ax1.set_xlabel("Days")
    ax1.set_title(r"$\chi^2 =$"+f"{round(chisquared,4)}, p = {round(p_val,4)}", fontweight="bold")
    ax1.legend(frameon=False)
    fig1.savefig(f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}/OS_{tissue}_th-{th}_minsize-{min_size_community}.pdf")

    # We save for spatial location faster computations and reproducibilities
    if (comms_ready is False) and (plot_comms is True):
        np.save(f"../Figures/Spatial-overlap/{args.algorithm}_{args.metric}/Communities-tumors_{tissue}_th-{th}_minsize-{min_size_community}.npy", lesion_communities, allow_pickle=True)
    
    print("=====================================================")
    print(f"Results --> {tissue}, th {th}, minsize {min_size_community}, {args.algorithm}, {args.metric}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Chi-squared statistic: ", chisquared)
    print("p value: ", p_val)
    print("Stats table: \n", stats)
    print("Covariance: \n", covariance)