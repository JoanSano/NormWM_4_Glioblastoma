import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import glob

def rewrite_subjectID(subject_ID):
    subject_4digits = subject_ID.split("-")
    subject = "-".join(subject_4digits[:-1])
    digits = str(subject_4digits[-1][1:])
    return subject + "-" + digits

if __name__ == '__main__':
    # Get the subject to process
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="Full ID of the subject (e.g., UCSF-PDGM-XXXX)")
    parser.add_argument("grade", type=str, choices=["II", "III", "IV"], help="Grade of the tumor")
    args = parser.parse_args()
    subject_ID = rewrite_subjectID(args.subject)

    # Loading the metadata that is available in the demographics
    demographics = pd.read_csv(f"../data/UCSF-PDGM-metadata_v3.csv")
    # We preselect only the current working subject
    row = pd.DataFrame(demographics.loc[demographics["ID"]==subject_ID])
    keys = { 
        # Do not alter the order of these entries, they are in correspondance to the TDMaps.sh script
        "tissue-whole": "Whole tumor size (voxels)",
        "tissue-core": "Core size (voxels)",
        "tissue-nonenhancing": "Non-enhancing size (voxels)",
        "tissue-enhancing": "Enhancing size (voxels)",
        "tissue-core+enhancing": "Core+Enhancing size (voxels)",
    }

    # We compute and add the Tract Density (TD) metrics
    for tissue, column in keys.items():
        ts = tissue.split("_")[0]
        mask = nib.load(
            glob.glob(f"../TDMaps_Grade-{args.grade}/{args.subject}/masks/*{ts}*")[0]
        ).get_fdata()
        
        # Save result
        row[column] = mask.sum()

    # Printing the results because:
    #       1. In case we execute this for a single subject we get 
    #          the correct outputs
    #       2. The joint execution from the bash file will correctly 
    #          add the results into the csv with for the whole dataset
    string = ""
    for i, (k,v) in enumerate(zip(row.columns, row.values[0])):
        if k=="Final pathologic diagnosis (WHO 2021)":
            v = (" ".join(v.split(",")))
        if i>0:
            string+=f"{v},"
    print(string[:-1])