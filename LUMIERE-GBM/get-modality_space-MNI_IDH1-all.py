import glob
import pandas as pd
import shutil
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--modality", type=str, choices=["T1", "CT1", "T2", "seg_mask"], help="Modality to collect and reogranize", default="seg_mask")
parser.add_argument("--main_dir", type=str, help="Main data directory", default="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_LUMIERE-GBM_v1-13122022")
parser.add_argument("--modality_dir", type=str, help="Directory where the data to collect is stored", default="LUMIERE-GBM_MNI-ICBM-2009b-NLIN-ASYM")
parser.add_argument("--week", type=str, help="Week to collect (e.g., 000, 000-1, 000-2, ...)", default="000")
parser.add_argument("--demographics", type=str, help="Name of the demographics file", default="LUMIERE-Demographics_Pathology")
args = parser.parse_args()

modality = args.modality
MAIN_DIR = args.main_dir
DATA_MOD_DIR = os.path.join(args.main_dir, args.modality_dir)
if args.week == "000" or args.week == "000-1":
    DESTINATION_DIR = f"{MAIN_DIR}/{args.modality_dir}_data-preop_{modality}/IDH1-"
elif args.week == "000-2":
    DESTINATION_DIR = f"{MAIN_DIR}/{args.modality_dir}_data-postop_{modality}/IDH1-"
else:
    DESTINATION_DIR = f"{MAIN_DIR}/{args.modality_dir}_week-{args.week}_{modality}/IDH1-"
if modality == "seg_mask":
    niftis = glob.glob(f"{DATA_MOD_DIR}/*/week-{args.week}/*{modality}__Warped*")
else:    
    niftis = glob.glob(f"{DATA_MOD_DIR}/*/week-{args.week}/{modality}*__Warped*")

demographics = pd.read_csv(f"{MAIN_DIR}/data/{args.demographics}.csv")
print(f"Data: {DATA_MOD_DIR}")
print(f"Destiantion: {DESTINATION_DIR}")
subject_col = "Patient"
idh_col = "IDH (WT: wild type)"

idh1_dict = {"WT":"WT", "wt": "WT", "R132H mut":"MUT", "IDH1 neg, Sequencing required": "Inconclusive", "na": "UNKNOWN",}
for nifti in tqdm(niftis, desc="Getting the files and sorting based on IDH1 mutation status"):
    subject = nifti.split("/")[-3] 
    idh1 = idh1_dict[str(demographics.loc[demographics[subject_col] == subject][idh_col].to_numpy()[0])]
    
    os.makedirs(f"{DESTINATION_DIR}{idh1}", exist_ok=True)
    shutil.copy2(nifti, f"{DESTINATION_DIR}{idh1}/LUMIERE-{subject.split('-')[-1]}_{modality}___Warped.nii.gz")

    ## CAUTION!
    #os.remove(nifti)