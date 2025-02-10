import nibabel as nib
import numpy as np
import pandas as pd
import nilearn.plotting as plotting
import matplotlib.pylab as plt
import scipy
import os
import argparse
import glob

def rewrite_subjectID(subject_ID):
    subject_3digits = subject_ID.split("-")
    subject = "-".join(subject_3digits[:-1])
    digits = str(subject_3digits[-1]).zfill(4)
    return subject + "-" + digits

def MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = scipy.stats.chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi # units: nats

def plot_registration(fixed_img, registered_img, iso_levels, fig_folder, modality, title, Nslices=20):
    fig, ax = plt.subplots(6, 1, figsize=(40, 20), facecolor="k")
    fig.subplots_adjust(hspace=0, left=0, right=1, top=1, bottom=0)

    display_x = plotting.plot_anat(
        fixed_img, 
        axes=ax[0], 
        display_mode="x", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_x.add_overlay(
        registered_img,
        cmap = 'Reds',
        alpha=1
    )

    display_y = plotting.plot_anat(
        fixed_img, 
        axes=ax[1], 
        display_mode="y", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_y.add_overlay(
        registered_img,
        cmap = 'Reds',
        alpha=1
    )

    display_z = plotting.plot_anat(
        fixed_img, 
        axes=ax[2], 
        display_mode="z", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_z.add_overlay(
        registered_img,
        cmap = 'Reds',
        alpha=1
    )

    display_x = plotting.plot_anat(
        fixed_img, 
        axes=ax[3], 
        display_mode="x", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_x.add_contours(
        registered_img,
        levels=iso_levels,
        cmap = 'Reds',
        linewidths=.5,
        antialiased=False
    )

    display_y = plotting.plot_anat(
        fixed_img, 
        axes=ax[4], 
        display_mode="y", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_y.add_contours(
        registered_img,
        levels=iso_levels,
        cmap = 'Reds',
        linewidths=.5,
        antialiased=False
    )

    display_z = plotting.plot_anat(
        fixed_img, 
        axes=ax[5], 
        display_mode="z", 
        cut_coords=np.linspace(-80,80,Nslices), 
        draw_cross=False, 
        black_bg=True,
        annotate=False
    )
    display_z.add_contours(
        registered_img,
        levels=iso_levels,
        cmap = 'Reds',
        linewidths=.5,
        antialiased=False
    )
    fig.suptitle(title, fontweight='bold', color='white', fontsize=20)
    fig.savefig(f"{fig_folder}/{modality}_registration.pdf", format='pdf', dpi=100)
    plt.close()

def plot_lesions(lesions, mni_img, fig_folder, title, Nslices=20):
    fig, ax = plt.subplots(3, 1, figsize=(20, 5), facecolor="k")
    fig.subplots_adjust(hspace=0, left=0, right=1, top=1, bottom=0)
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
        cmap = 'Reds',
        colorbar=True,
        cbar_vmin=0,
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
        cmap = 'Reds',
        colorbar=False,
        cbar_vmin=0,
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
        cmap = 'Reds',
        colorbar=False,
        cbar_vmin=0,
        cbar_vmax=max_overlap,
        cbar_tick_format="%i"
    )

    fig.suptitle(title, fontweight='bold', color='white', fontsize=20)
    fig.savefig(f"{fig_folder}/{title}.pdf", format='pdf', dpi=100)
    plt.close()
 
parser = argparse.ArgumentParser()
parser.add_argument("modalities", nargs="+", help="Modalities to check")
parser.add_argument("--main_dir", type=str, help="Main data directory", default="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024")
parser.add_argument("--modality_prefix", type=str, help="Directory where the data to collect is stored", default="UPENN-GBM_MNI-ICBM-2009b-NLIN-ASYM")
parser.add_argument("--mni_dir", type=str, help="Location of the MNI registration template", default="/home/sano/Documents/Joan/Data/MNI_ICBM_2009b_NLIN_ASYM")
parser.add_argument("--test", action='store_true', help="Include this flag for testing on a single subject")
parser.add_argument("--metadata", type=str, help="Name of the clinical data file", default="UPENN-GBM_clinical_info_v2.1.csv")
args = parser.parse_args()

modalities = args.modalities
MAIN_DIR = args.main_dir
modality_prefix = args.modality_prefix
MNI_DIR = args.mni_dir
T1 = nib.load(os.path.join(MNI_DIR, "T1_0.5mm.nii"))
T2 = nib.load(os.path.join(MNI_DIR, "T2_0.5mm.nii"))
MASK = nib.load(os.path.join(MNI_DIR, "T1_0.5mm_brain_mask.nii.gz"))
T1_BRAIN = MASK.get_fdata()*T1.get_fdata()

demographics = pd.read_csv(f"{MAIN_DIR}/data/{args.metadata}")
subjects = demographics["ID"]

figures_dir = os.path.join(args.main_dir, "Figures", "QC_registration-MNI")
os.makedirs(figures_dir, exist_ok=True)

# Logs
logs = open(f"{figures_dir}/Logs-registration.txt", "w")
logs.write(f"Logs for the normalization of {MAIN_DIR}\n")
logs.write("-----------------------------------------------------------------\n")

dice_on_T1 = {}
error_on_T1 = {}
error_on_T1_average = {}
whole_tumor = np.zeros(T1.get_fdata().shape)
enhancing_tumor = np.zeros(T1.get_fdata().shape)
nonenhancing_tumor = np.zeros(T1.get_fdata().shape)
core_tumor = np.zeros(T1.get_fdata().shape)
iso_levels = {
    "automated_approx_segm": [1,2,3,4],
    "corrected_segm": [1,2,3,4],
    "FLAIR": np.linspace(200, 3000, 5),
    "T2": np.linspace(200, 1000, 5),
    "T1": np.linspace(200, 1000, 5),
    "T1GD": np.linspace(200, 8000, 5),
}
total_s = 0
for i, s in enumerate(subjects):
    msg = s + ":"

    for j, m in enumerate(modalities):
        try:
            file = glob.glob(os.path.join(MAIN_DIR, modality_prefix)+f"_{m}/**/*{s}*{m}__Warped*", recursive=True)[0]
            nifti = nib.load(file)
            os.makedirs(os.path.join(figures_dir,s), exist_ok=True)

            if not os.path.exists(f"{os.path.join(figures_dir,s)}/{m}_registration.pdf"):
                plot_registration(T1, nifti, iso_levels[m], os.path.join(figures_dir,s), m, m+": "+s, Nslices=15)
            
            if "segm" in m:
                whole_tumor += np.where(nifti.get_fdata()>=1, 1, 0)
                enhancing_tumor += np.where(nifti.get_fdata()==4, 1, 0)
                nonenhancing_tumor += np.where(nifti.get_fdata()==2, 1, 0)
                core_tumor += np.where(nifti.get_fdata()==1, 1, 0)
            
            """ if m=="T1_bias":
                # Metrics between the MNI brain mask and the registered brain
                registered_mask = np.where(nifti.get_fdata()>.001, 1, 0)
                dice_on_T1[s] = 2*(np.sum(MASK.get_fdata()*registered_mask))/(MASK.get_fdata().sum()+registered_mask.sum()) 
                error_on_T1[s] = np.square(T1_BRAIN - nifti.get_fdata())   # We know that the registered image is already skull-stripped           
                error_on_T1_average[s] = error_on_T1[s].mean() """

        except:
            msg = msg + f" {m}"

    if msg!=s + ":":
        print("\t WARNING!" + msg)
        logs.write(msg+"\n")

    total_s += 1
    print(f"subject {s}, number {total_s} from {len(subjects)} ({100*total_s/len(subjects)}%)\n")
    if args.test and total_s==1:
        break
    
logs.close()

""" plot_lesions(whole_tumor, T1, figures_dir, title="Whole-tumor", Nslices=15)
plot_lesions(enhancing_tumor, T1, figures_dir, title="Enhancing tissue", Nslices=15)
plot_lesions(nonenhancing_tumor, T1, figures_dir, title="Non-enhancing tissue", Nslices=15)
plot_lesions(core_tumor, T1, figures_dir, title="Necrotic tissue", Nslices=15) """

""" values2array = lambda dict: np.array([v for k,v in dict.items()])
fig, ax = plt.subplots(1, 2, figsize=(8,4))

ax[0].violinplot(dice_on_T1.values(), showmeans=False, showmedians=False, showextrema=False)
ax[0].plot(1, np.percentile(values2array(dice_on_T1), 50), 's', color="white", label="Median")
ax[0].plot(1, np.mean(values2array(dice_on_T1)), 'D', color="black", label="Mean")
ax[0].plot(1, np.percentile(values2array(dice_on_T1), 10), 's', color="blue", label="P10")
ax[0].set_title("Dice score", fontweight="bold")
ax[0].spines[["top","right","bottom"]].set_visible(False)
ax[0].set_xticks([])

ax[1].violinplot(error_on_T1_average.values(), showmeans=False, showmedians=False, showextrema=False)
ax[1].plot(1, np.percentile(values2array(error_on_T1_average), 50), 's', color="white", label="Median")
ax[1].plot(1, np.mean(values2array(error_on_T1_average)), 'D', color="black", label="Mean")
ax[1].plot(1, np.percentile(values2array(error_on_T1_average), 90), 's', color="red", label="P90")
ax[1].plot(1, np.percentile(values2array(error_on_T1_average), 10), 's', color="blue", label="P10")
ax[1].set_title("Mean squared error", fontweight="bold")
ax[1].spines[["top","right","bottom"]].set_visible(False)
ax[1].set_xticks([])
ax[1].legend(frameon=False, loc="lower right")

fig.savefig(f"{figures_dir}/metrics.svg", format='svg', dpi=300)
fig.savefig(f"{figures_dir}/metrics.png", format='png', dpi=300)
plt.close() """