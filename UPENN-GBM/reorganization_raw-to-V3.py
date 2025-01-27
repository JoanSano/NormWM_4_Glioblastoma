import numpy as np
import pandas as pd
import os
import glob
import shutil

root = "/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024"
destination = os.path.join(root, "UPENN-GBM_data-raw_V3-organized")
raw_loc = os.path.join(root, "UPENN-GBM_data-raw_V2/NIfTI-files")

os.makedirs(destination, exist_ok=True)
raw_folders = ['automated_segm','images_DSC','images_segm','images_DTI/images_DTI','images_structural','images_structural_unstripped'] # Better to handle (for sure) the DTI nested folder

data_availability = pd.read_csv(os.path.join(root, "data/UPENN-GBM_data_availability.csv"), sep=',')
subjects = list(data_availability["ID"].values)

for i, s in enumerate(subjects):
    os.makedirs(f"{destination}/{s}", exist_ok=True)
    for mod in raw_folders:
        if "segm" in mod:     
            imgs = glob.glob(f"{raw_loc}/{mod}/{s}*")
        else:
            imgs = glob.glob(f"{raw_loc}/{mod}/{s}/{s}*")
        for img in imgs:
            shutil.copy(img, f"{destination}/{s}")
    print(f"{s} without problems! ({round((i+1)/len(subjects)*100, 2)}% of subjects completed)")