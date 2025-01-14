#!/bin/bash

JHU_DIR="../data/JHU-WMAtlas_MNI2009b"
tracts_dir="../data/JHU-WMAtlas_MNI2009b/normative_WM-tracts"
thr_tracts=0
tmp_dir="tmp"
MNI_IMG="/home/joanfr/Documents/Utils/MNI_ICBM_2009b_NLIN_ASYM/T1_0.5mm.nii"

# TMP directory
if [[ -d $tmp_dir ]]; then
    echo "Temporary directory exists"
else
    mkdir $tmp_dir
fi

for label in {1..1}; do
    # Select tract label
    echo "ROI ${label}"

    if [[ -d "${tracts_dir}/ROI-${label}" ]]; then
        echo "ROI-${label} dirextory already exists"
    else
        mkdir "${tracts_dir}/ROI-${label}_thr-${thr_tracts}"
    fi

    # Make mask for tract 
    fslmaths "${JHU_DIR}/JHU-ICBM-labels-0.5mm__Warped.nii.gz" -thr $label -uthr $label "${tmp_dir}/tmp-${label}.nii.gz"
    fslmaths "${tmp_dir}/tmp-${label}.nii.gz" -bin "${tmp_dir}/tmp-${label}.nii.gz"

    # Extract tracts
    tckedit "../data/dTOR_full_tractogram.tck" "${tracts_dir}/ROI-${label}_thr-${thr_tracts}/WM-tracts_ROI-${label}.tck" -include "${tmp_dir}/tmp-${label}.nii.gz" -force

    # Tract density map
    tckmap "${tracts_dir}/ROI-${label}_thr-${thr_tracts}/WM-tracts_ROI-${label}.tck" "${tracts_dir}/ROI-${label}_thr-${thr_tracts}/WM-tracts_ROI-${label}_TDI.nii.gz" -template $MNI_IMG -force

    # Tract density DEC map
    tckmap "${tracts_dir}/ROI-${label}_thr-${thr_tracts}/WM-tracts_ROI-${label}.tck" "${tracts_dir}/ROI-${label}_thr-${thr_tracts}/WM-tracts_ROI-${label}_TDI-DEC.nii.gz" -template $MNI_IMG -dec -force

done

rm -rf $tmp_dir 