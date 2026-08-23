import nibabel as nib
import numpy as np
import pandas as pd
import argparse

def get_volume_in_cubic_cm(mask_path, compartment, unit_to_cm, tol=1e-6):

    segmentation = nib.load(mask_path)
    voxel_sizes_preop = segmentation.header["pixdim"][1:4]
    spatial_unit, _ = segmentation.header.get_xyzt_units()
    conversion_to_cm = unit_to_cm.get(spatial_unit.lower())

    if conversion_to_cm is None:
        raise ValueError(f"Unrecognized spatial unit '{spatial_unit}'")

    # Transform every coordinate to cm regardless of isotropy
    voxel_2_cm3 = np.prod(voxel_sizes_preop * conversion_to_cm)
    
    mask = np.where(segmentation.get_fdata()==compartment, 1, 0)
    volume_cm3 = voxel_2_cm3 * mask.sum()
    
    return volume_cm3, voxel_2_cm3

def computeEOR(preop_vol, postop_vol, tol=1e-6): 
    return 100 * (preop_vol - postop_vol) / (preop_vol+tol)
    
unit_to_cm = {
    "meter": 100./1.,
    "m": 100./1.,
    "centimeter": 1./1.,
    "cm": 1./1.,
    "mm": 1./10.,
    "micron": 1./10000.,
    "um": 1./10000.,
    # "unknown" intentionally omitted — will raise ValueError cleanly
}
tol = 1e-10

if __name__ == '__main__':

    # Get the subject to process
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",     type=str, help="Full ID of the subject (e.g., LUMIERE-001)")
    parser.add_argument("--preop_file",  type=str, help="Path to the preoperative file")
    parser.add_argument("--postop_file", type=str, help="Path to the postoperative file")
    parser.add_argument("--label",       type=int, help="Integer label of the comparment")
    args = parser.parse_args()

    if (args.preop_file == "empty") or (args.postop_file == "empty"):
        print(f"LUMIERE-{args.subject.split('-')[-1]},na,na,na")
    else:
        preop_vol, _ = get_volume_in_cubic_cm(args.preop_file, args.label, unit_to_cm)
        postop_vol, _ = get_volume_in_cubic_cm(args.postop_file, args.label, unit_to_cm)
        eor = computeEOR(preop_vol, postop_vol, tol=tol)

        # Printing the results because:
        #       1. In case we execute this for a single subject we get 
        #          the correct outputs
        #       2. The joint execution from the bash file will correctly 
        #          add the results into the csv with for the whole dataset
        print(f"LUMIERE-{args.subject.split('-')[-1]},{preop_vol.round(2)},{postop_vol.round(2)},{eor.round(2)}")
    