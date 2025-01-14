import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import glob
import os

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
    parser.add_argument("--min_streamlines", type=int, default=10, help="Minimum number of streamlines per voxel to consider")
    parser.add_argument("--tolerance", type=float, default=0.0001, help="Min size of the thresholded TD Map")
    args = parser.parse_args()
    subject_ID = rewrite_subjectID(args.subject)

    # Loading the metadata that is available in the demographics
    demographics = pd.read_csv(f"../data/UCSF-PDGM-metadata_v3.csv")
    # We preselect only the current working subject
    row = pd.DataFrame(demographics.loc[demographics["ID"]==subject_ID])
    keys = { 
        # Do not alter the order of these entries, they are in correspondance to the TDMaps.sh script
        "tissue-whole_TDMap": "Whole TDMap",
        "tissue-whole_TDMap-lesion": "Whole lesion TDMap",
        "tissue-core_TDMap": "Core TDMap",
        "tissue-core_TDMap-lesion": "Core lesion TDMap",
        "tissue-nonenhancing_TDMap": "Non-enhancing TDMap",
        "tissue-nonenhancing_TDMap-lesion": "Non-enhancing lesion TDMap",
        "tissue-enhancing_TDMap": "Enhancing TDMap",
        "tissue-enhancing_TDMap-lesion": "Enhancing lesion TDMap",
        "tissue-core+enhancing_TDMap": "Core+Enhancing TDMap",
        "tissue-core+enhancing_TDMap-lesion": "Core+Enhancing lesion TDMap"
    }

    # We compute and add the Tract Density (TD) metrics
    for tissue, column in keys.items():
        # Load the files in the correct order
        file = f"../TDMaps_Grade-{args.grade}/{args.subject}/maps/{args.subject}_{tissue}.nii.gz"
        if os.path.exists(file):
            # The TD map was created correctly
            td_map = nib.load(file).get_fdata()

            # Average lesion TDI --> We need to threshold a minimum number of streamlines to contribute
            if "TDMap-lesion" in file:
                th_td_map = np.where(td_map>=args.min_streamlines, td_map, np.nan)
                if np.nansum(th_td_map)<args.tolerance:
                    # Lesion TD Map is empty
                    tdi = 0.0
                else:
                    tdi = np.nanmean(th_td_map)
            # Average TDI --> We need to mask and only count contributions within the lesion & tissue mask
            elif "TDMap" in file: 
                ts = tissue.split("_")[0]
                mask = nib.load(f"../TDMaps_Grade-{args.grade}/{args.subject}/masks/{args.subject}_{ts}.nii.gz").get_fdata()
                if mask.sum()<1:
                    # Smaller than a voxel --> There is no tissue!
                    tdi = np.nan
                else:
                    masked_td_map = np.where(mask==1, td_map, np.nan)
                    masked_th_td_map = np.where(masked_td_map>=args.min_streamlines, masked_td_map, np.nan)
                    if np.nansum(masked_th_td_map)<args.tolerance:
                        # TD Map is empty
                        tdi = 0.0
                    else:
                        tdi = np.nanmean(masked_th_td_map)
            else:
                raise Warning("TDMap was not found in the name of the file. A strange file was loaded that did not throw any error!")
        else:
            # The TD map was not created due to zero volume of the label
            tdi = np.nan
            
        # Save result
        row[column] = tdi
   
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