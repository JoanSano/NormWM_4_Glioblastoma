import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=0, help="Streamline density threshold")
parser.add_argument("--correction", type=str, choices=["fwer","fdr"], default="fdr", help="Multiple hypotheses correction method")
args = parser.parse_args()

stream_th = args.threshold
fwer = True if args.correction=="fwer" else False
daysXmonth = 30
percentiles2check = (20,80),(25,75),(30,70),(35,65),(40,60),(45,55),(50,50)
n_resamples = 2500 # Bottstrapping and permutation of correlation values
n_perms = 5000 # Permutation of Cox Prop Hazard models
months = np.array([6,12,18,24,30,36,42,48])

data_all = pd.read_csv(f"../Figures/TDMaps_Grade-IV/demographics-TDMaps_streamTH-{stream_th}.csv")
data = data_all[
    [
        "OS",
        'Whole tumor size (voxels)',
        'Core size (voxels)', 
        'Non-enhancing size (voxels)',
        'Enhancing size (voxels)', 
        'Core+Enhancing size (voxels)'
    ]
]
life = data_all["1-dead 0-alive"].values

"""
Sex
Age at MRI
WHO CNS Grade
Final pathologic diagnosis (WHO 2021)
MGMT status
MGMT index
1p/19q
IDH
1-dead 0-alive
OS
EOR
Biopsy prior to imaging
BraTS21 ID
BraTS21 Segmentation Cohort
BraTS21 MGMT Cohort
Labels
Whole TDMap
Whole lesion TDMap
Core TDMap
Core lesion TDMap
Non-enhancing TDMap
Non-enhancing lesion TDMap
Enhancing TDMap
Enhancing lesion TDMap
Core+Enhancing TDMap
Core+Enhancing lesion TDMap
"""