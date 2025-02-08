import glob
import nibabel as nib
import nilearn
import pandas as pd
import shutil
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--modality", type=str, choices=["automated_approx_segm", "corrected_segm", "FLAIR", "T1", "T1GD", "T2"], help="Modality to collect and reogranize", default="segmentation")
parser.add_argument("--main_dir", type=str, help="Main data directory", default="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024")
parser.add_argument("--modality_dir", type=str, help="Directory where the data to collect is stored", default="UPENN-GBM_MNI-ICBM-2009b-NLIN-ASYM")
args = parser.parse_args()

modality = args.modality
MAIN_DIR = args.main_dir
DATA_MOD_DIR = os.path.join(args.main_dir, args.modality_dir)# f"{MAIN_DIR}/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM/" if len(sys.argv)==1 else sys.argv[0]
DESTINATION_DIR = f"{MAIN_DIR}/{args.modality_dir}_{modality}/IDH1-"
niftis = glob.glob(f"{DATA_MOD_DIR}/*{modality}__Warped*", recursive=True)
demographics = pd.read_csv(f"{MAIN_DIR}/data/UPENN-GBM_clinical_info_v2.1.csv")
print(f"Destiantion: {DESTINATION_DIR}")
idh1_dict = {"Wildtype":"WT", "Mutated":"Mut", "NOS/NEC":"NOSNEC"}
for nifti in tqdm(niftis, desc="Getting the modalities and sorting based on IDH1 mutation status"):
    subject = nifti.split("/")[-1].split(f"_{modality}__Warped")[0]
    idh1 = str(demographics.loc[demographics["ID"] == subject]["IDH1"].to_numpy()[0])

    os.makedirs(f"{DESTINATION_DIR}{idh1_dict[idh1]}", exist_ok=True)
    shutil.copy2(nifti, f"{DESTINATION_DIR}{idh1_dict[idh1]}/")

    ## CAUTION!
    os.remove(nifti)