import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import glob

if __name__ == '__main__':
    # Get the subject to process
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Main irectory")
    parser.add_argument("subject", type=str, help="Full ID of the subject (e.g., UPENN-GBM-XXXXX_11)")
    parser.add_argument("idh1", type=str, choices=["WT", "MUT", "NOSNEC"], help="IDH1 mutation status")
    parser.add_argument("--demographics", type=str, default="UPENN-GBM_clinical_info_v3.0.csv")
    args = parser.parse_args()
    subject_ID = args.subject
    
    # Loading the metadata that is available in the demographics
    args.dir = args.dir[:-1] if args.dir[-1]=="/" else args.dir # We delete the last "/" if present
    demographics = pd.read_csv(f"{args.dir}/data/{args.demographics}")

    # We preselect only the current working subject
    row = pd.DataFrame(demographics.loc[demographics["ID"]==subject_ID])
    if row.empty:
        raise Warning("______ No entry for the subject was found in the clinical data entered in --demographics ______")
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
            glob.glob(f"{args.dir}/TDMaps_IDH1-{args.idh1}/{args.subject}/masks/*{ts}*")[0]
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
        string+=f"{v},"
    print(string[:-1])