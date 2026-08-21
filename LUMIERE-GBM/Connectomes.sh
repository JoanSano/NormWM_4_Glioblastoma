#!/bin/bash

# Default values
IDH=WT
KEEP_TCK=1
N=1

# Usage function
usage() {
  echo "Usage: $0 [-i idh] [-k keep_tck] [-d dec] [-n n]"
  echo "  -i IDH         : Specify the IDH1 statatus to analyze (default: WT; options: WT, MUT, or NOSNEC)"
  echo "  -k KEEP_TCK    : Specify whether to keep the lesion tractograms (default: 1, options: 0, 1)"
  echo "  -n N           : Specify the number of subjects to parallelize (default: 1, options: 1, ..., N)"
  exit 1
}

# Parse command-line options
while getopts ":i:k:d:n:s:h" opt; do
  case ${opt} in
    i)
      IDH=$OPTARG
      ;;
    k)
      KEEP_TCK=$OPTARG
      ;;
    d)
      DEC=$OPTARG
      ;;
    n)
      N=$OPTARG
      ;;
    h)
      usage  # Display help and exit
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Define base paths
MAIN_DIR="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_LUMIERE-GBM_v1-13122022"
MNI_TEMPLATE="/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM/dTOR_full_tractogram"
LESION_DIR="${MAIN_DIR}/LUMIERE-GBM_MNI-ICBM-2009b-NLIN-ASYM_data-preop_seg_mask"
DESTINATION_MAPS="${MAIN_DIR}/TDMaps_IDH1-${IDH}"
HCPEX="/home/joan/Documents/Parcellations/HCPex__Glasser-like+subcortical/MNI_ICBM_2009b_NLIN_ASYM/HCPex_MNI_ICBM_2009b_NLIN_ASYM.nii.gz"
AAL3="/home/joan/Documents/Parcellations/AAL3/MNI_ICBM_2009b_NLIN_ASYM/AAL3v1_1mmxMNI_ICBM_2009b_NLIN_ASYM__Warped.nii.gz"

if [[ ! -d $DESTINATION_MAPS ]]; then
    mkdir $DESTINATION_MAPS
fi

# Redirect standard output
exec 1> $DESTINATION_MAPS"/Logs-Connectomes.txt"
exec 2> $DESTINATION_MAPS"/Errors-Connectomes.txt"

for lesion in $LESION_DIR/IDH1-$IDH/*; do
    ( 
        # Subject ID and directory
        SUBJECT=$(basename "$lesion")
        SUBJECT="${SUBJECT%%_*}"
        SUBJECT_DIR="${DESTINATION_MAPS}/${SUBJECT}"
        
        if [[ ! -d $SUBJECT_DIR ]]; then mkdir $SUBJECT_DIR; fi
        if [[ ! -d $SUBJECT_DIR"/masks" ]]; then mkdir $SUBJECT_DIR"/masks"; fi
        if [[ ! -d $SUBJECT_DIR"/tracts" ]]; then mkdir $SUBJECT_DIR"/tracts"; fi
        if [[ ! -d $SUBJECT_DIR"/maps" ]]; then mkdir $SUBJECT_DIR"/maps"; fi
        if [[ ! -d $SUBJECT_DIR"/maps-dec" ]]; then mkdir $SUBJECT_DIR"/maps-dec"; fi
        if [[ ! -d $SUBJECT_DIR"/connectomes" ]]; then mkdir $SUBJECT_DIR"/connectomes"; fi
        
        # Tissues
        WT="${SUBJECT_DIR}/masks/${SUBJECT}_tissue-whole.nii.gz"
        CR="${SUBJECT_DIR}/masks/${SUBJECT}_tissue-core.nii.gz"
        NE="${SUBJECT_DIR}/masks/${SUBJECT}_tissue-nonenhancing.nii.gz"
        EH="${SUBJECT_DIR}/masks/${SUBJECT}_tissue-enhancing.nii.gz"
        CREH="${SUBJECT_DIR}/masks/${SUBJECT}_tissue-core+enhancing.nii.gz"

        # Masking the tumor tissues
        echo " INFO ${SUBJECT}: Masking the tumor tissues"
        if [[ ! -f $WT ]]; then fslmaths $lesion -bin $WT; fi
        if [[ ! -f $CR ]]; then fslmaths $lesion -thr 1 -uthr 1 -bin $CR; fi
        if [[ ! -f $NE ]]; then fslmaths $lesion -thr 2 -uthr 2 -bin $NE; fi
        if [[ ! -f $EH ]]; then fslmaths $lesion -thr 3 -uthr 3 -bin $EH; fi
        if [[ ! -f $CREH ]]; then fslmaths $CR -add $EH -bin $CREH; fi

        # Compute TDMaps and tracts for each tissue type sequentially
        tissues=("whole" "core" "nonenhancing" "enhancing" "core+enhancing")
        tissue_masks=($WT $CR $NE $EH $CREH)
        for i in {0..4}; do
            TISSUE=${tissues[$i]}
            MASK=${tissue_masks[$i]}
            
            # Compute lesion tractogram for each tissue
            LESION_TCK="${SUBJECT_DIR}/tracts/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion.tck"
            if [[ ! -f $LESION_TCK ]]; then 
              echo " INFO ${SUBJECT}: Extracting streamlines for $TISSUE tissue"
              tckedit "${MNI_TEMPLATE}.tck" \
                  $LESION_TCK \
                  -include $MASK \
                  -nthreads 0 \
                  -force \
                  -quiet
            fi

            # Obtain the correspoding connectome
            for MODE in default ScaleInvnodevol ScaleInvlength ScaleLength; do
                case "$MODE" in
                    ScaleInvnodevol)
                        SCALE_ARG="-scale_invnodevol"
                        ;;
                    ScaleInvlength)
                        SCALE_ARG="-scale_invlength"
                        ;;
                    ScaleLength)
                        SCALE_ARG="-scale_length"
                        ;;
                    *)
                        SCALE_ARG=""
                        ;;
                esac

                # Parcellation HCPEx
                LESION_CONNECTOME="${SUBJECT_DIR}/connectomes/${SUBJECT}_tissue-${TISSUE}_connectome-${MODE}_parcellation-HCPEx.csv"
                if [[ ! -f $LESION_CONNECTOME ]]; then 
                  echo " INFO ${SUBJECT}: Computing connectome for tissue $TISSUE, mode $MODE, and parcellation HCPEx"
                  tck2connectome $LESION_TCK \
                    $HCPEX \
                    $LESION_CONNECTOME \
                    -zero_diagonal \
                    -symmetric \
                    -nthreads 1 \
                    $SCALE_ARG \
                    -force \
                    -quiet
                fi

                # AAL3
                LESION_CONNECTOME="${SUBJECT_DIR}/connectomes/${SUBJECT}_tissue-${TISSUE}_connectome-${MODE}_parcellation-AAL3.csv"
                if [[ ! -f $LESION_CONNECTOME ]]; then 
                  echo " INFO ${SUBJECT}: Computing connectome for tissue $TISSUE, mode $MODE, and parcellation AAL3"
                  tck2connectome $LESION_TCK \
                    $AAL3 \
                    $LESION_CONNECTOME \
                    -zero_diagonal \
                    -symmetric \
                    -nthreads 1 \
                    $SCALE_ARG \
                    -force \
                    -quiet
                fi
            done
            
        done
        
        if [[ $KEEP_TCK -eq 0 ]]; then
            echo " INFO ${SUBJECT}: Deleting lesion tracts!"
            rm -rf $SUBJECT_DIR"/tracts"
        fi
        
    ) &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

wait

echo "Lesion connectome mapping finished. Do not go grab a beer, you need to take care of your health."  mhvghgj