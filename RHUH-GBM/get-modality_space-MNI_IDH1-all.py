import glob
import pandas as pd
import shutil
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--modality", type=str, choices=["segmentations", "flair", "t1", "t1ce", "t2", "adc"], help="Modality to collect and reogranize", default="segmentations")
parser.add_argument("--main_dir", type=str, help="Main data directory", default="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_RHUH-GBM_v2-29102025")
parser.add_argument("--modality_dir", type=str, help="Directory where the data to collect is stored", default="RHUH-GBM_data-T0_MNI-ICBM-2009b-NLIN-ASYM")
parser.add_argument("--demographics", type=str, help="Name of the demographics file", default="clinical_data_TCIA_RHUH-GBM")
args = parser.parse_args()

modality = args.modality
MAIN_DIR = args.main_dir
DATA_MOD_DIR = os.path.join(args.main_dir, args.modality_dir)# f"{MAIN_DIR}/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM/" if len(sys.argv)==1 else sys.argv[0]
DESTINATION_DIR = f"{MAIN_DIR}/{args.modality_dir}_{modality}/IDH1-"
niftis = glob.glob(f"{DATA_MOD_DIR}/*/*{modality}__Warped*", recursive=True)
demographics = pd.read_csv(f"{MAIN_DIR}/data/{args.demographics}.csv")
print(f"Data: {DATA_MOD_DIR}")
print(f"Destiantion: {DESTINATION_DIR}")
idh1_dict = {"wt":"WT", "mut":"Mut", "NOS/NEC":"NOSNEC"}
for nifti in tqdm(niftis, desc="Getting the modalities and sorting based on IDH1 mutation status"):
    subject = nifti.split("/")[-1].split(f"_0_{modality}__Warped")[0]
    idh1 = str(demographics.loc[demographics["Patient ID"] == subject]["IDH status = (mutant [mut], wild type [wt], NOS)"].to_numpy()[0])

    os.makedirs(f"{DESTINATION_DIR}{idh1_dict[idh1]}", exist_ok=True)
    shutil.copy2(nifti, f"{DESTINATION_DIR}{idh1_dict[idh1]}/")

    ## CAUTION!
    os.remove(nifti)