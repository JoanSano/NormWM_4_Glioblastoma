import glob
import nibabel as nib
import nilearn
import pandas as pd
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--modality", type=str, choices=["FLAIR_bias", "segmentation", "T2_bias", "T1c_bias", "T1_bias"], help="Modality to collect and reogranize", default="segmentation")
parser.add_argument("--main_dir", type=str, help="Main data directory", default="/home/sano/Documents/Joan/Data/Glioblastoma_UCSF-PDGM-v3-20230111")
parser.add_argument("--modality_dir", type=str, help="Directory where the data to collect is stored", default="UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM")
args = parser.parse_args()

modality = args.modality
MAIN_DIR = args.main_dir
DATA_MOD_DIR = os.path.join(args.main_dir, args.modality_dir)# f"{MAIN_DIR}/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM/" if len(sys.argv)==1 else sys.argv[0]
DESTINATION_DIR = f"{MAIN_DIR}/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM_{modality}/Grade-"
niftis = glob.glob(f"{DATA_MOD_DIR}/**/*{modality}__Warped*", recursive=True)
demographics = pd.read_csv(f"{MAIN_DIR}/data/UCSF-PDGM-metadata_v3.csv")

grade_dict = {2:"II", 3:"III", 4:"IV"}
for nifti in niftis:
    subject_4digits = nifti.split("/")[-1].split("_")[0].split("-")
    subject = "-".join(subject_4digits[:-1])
    digits = str(int(subject_4digits[-1])).zfill(3)
    subject = subject + "-" + digits
    grade = int(demographics.loc[demographics["ID"] == subject]["WHO CNS Grade"])

    shutil.copy2(nifti, f"{DESTINATION_DIR}{grade_dict[grade]}/")
    