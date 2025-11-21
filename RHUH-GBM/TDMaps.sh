#!/bin/bash

# Default values
IDH=WT
KEEP_TCK=1
DEC=0
N=1
STREAM_D_TH=0

# Usage function
usage() {
  echo "Usage: $0 [-i idh] [-k keep_tck] [-d dec] [-n n]"
  echo "  -i IDH         : Specify the IDH1 statatus to analyze (default: WT; options: WT, MUT, or NOSNEC)"
  echo "  -k KEEP_TCK    : Specify whether to keep the lesion tractograms (default: 1, options: 0, 1)"
  echo "  -d DEC         : Specify whether to compute directionally encoded TD maps (default: 0, options: 0, 1)"
  echo "  -n N           : Specify the number of subjects to parallelize (default: 1, options: 1, ..., N)"
  echo "  -s STREAM_D_TH : Specify the minimum number of streamline density to threshold (default=0, option: 0, 10, ...)"
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
    s)
      STREAM_D_TH=$OPTARG
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
MAIN_DIR="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_RHUH-GBM_v2-29102025"
MNI_TEMPLATE="/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM/dTOR_full_tractogram"
LESION_DIR="${MAIN_DIR}/RHUH-GBM_data-T0_MNI-ICBM-2009b-NLIN-ASYM_segmentations"
DESTINATION_MAPS="${MAIN_DIR}/TDMaps_IDH1-${IDH}"
DEMOGRAPHICS="${DESTINATION_MAPS}/demographics-TDMaps_streamTH-${STREAM_D_TH}.csv"
MORPHOLOGY="${DESTINATION_MAPS}/morphology-tissues.csv"
CLINICAL_DATA_FILE="clinical-data_RHUH-GBM_v2.0_IDH1-${IDH}.csv"

if [[ ! -d $DESTINATION_MAPS ]]; then
    mkdir $DESTINATION_MAPS
fi

# Create the demographics header
TD_LABELS="Whole TDMap,Whole lesion TDMap,Core TDMap,Core lesion TDMap,Non-enhancing TDMap,Non-enhancing lesion TDMap,Enhancing TDMap,Enhancing lesion TDMap,Core+Enhancing TDMap,Core+Enhancing lesion TDMap" 
echo "Patient ID,Days from earliest imaging to surgery,Age,Sex,Preoperative KPS,Previous treatment,Histopathological subtype,WHO grade,IDH status,Operative adjuncts,Preoperative  contrast enhancing tumor volume (cm3),Postoperative contrast enhancing residual tumor (cm3),Preoperative T2/FLAIR abnormality  (cm3) ,Postoperative T2/FLAIR abmnormality (cm3),Extent of resection [EOR]  %,EOR,Adjuvant therapy,Radiotherapy treatment details (technique/dose/number of fractions),Postoperative Neurological Deficit,Postoperative KPS,Progression free survival [PFS] (days),Overall survival [OS] (days),Right Censored,${TD_LABELS}" > $DEMOGRAPHICS

MORP_LABELS="Whole tumor size (voxels),Core size (voxels),Non-enhancing size (voxels),Enhancing size (voxels),Core+Enhancing size (voxels)" 
echo "Patient ID,Days from earliest imaging to surgery,Age,Sex,Preoperative KPS,Previous treatment,Histopathological subtype,WHO grade,IDH status,Operative adjuncts,Preoperative  contrast enhancing tumor volume (cm3),Postoperative contrast enhancing residual tumor (cm3),Preoperative T2/FLAIR abnormality  (cm3) ,Postoperative T2/FLAIR abmnormality (cm3),Extent of resection [EOR]  %,EOR,Adjuvant therapy,Radiotherapy treatment details (technique/dose/number of fractions),Postoperative Neurological Deficit,Postoperative KPS,Progression free survival [PFS] (days),Overall survival [OS] (days),Right Censored,${MORP_LABELS}" > $MORPHOLOGY

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
        
        # Execute Python script
        echo " INFO ${SUBJECT}: Computing lesion morphology metrics"
        python morphology-extraction.py $MAIN_DIR $SUBJECT $IDH --demographics $CLINICAL_DATA_FILE >> $MORPHOLOGY

        # Compute TDMaps and tracts for each tissue type sequentially
        tissues=("whole" "core" "nonenhancing" "enhancing" "core+enhancing")
        tissue_masks=($WT $CR $NE $EH $CREH)

        for i in {0..4}; do
            TISSUE=${tissues[$i]}
            MASK=${tissue_masks[$i]}
            
            # Computing TDMaps for each tissue
            echo " INFO ${SUBJECT}: Computing TD map for $TISSUE tissue"
            out_f="${SUBJECT_DIR}/maps/${SUBJECT}_tissue-${TISSUE}_TDMap.nii.gz"
            if [[ ! -f $out_f ]]; then fslmaths "${MNI_TEMPLATE}_TDMap.nii.gz" -mul $MASK $out_f; fi
            
            # Compute lesion TDMap for each tissue
            out_f="${SUBJECT_DIR}/maps/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion.nii.gz"
            if [[ ! -f $out_f ]]; then 
              echo " INFO ${SUBJECT}: Extracting streamlines for $TISSUE tissue"
              tckedit "${MNI_TEMPLATE}.tck" \
                  "${SUBJECT_DIR}/tracts/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion.tck" \
                  -include $MASK \
                  -nthreads 0 \
                  -force \
                  -quiet

              echo " INFO ${SUBJECT}: Computing lesion TDMap for $TISSUE tissue"
              tckmap "${SUBJECT_DIR}/tracts/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion.tck" \
                  $out_f \
                  -template $lesion \
                  -nthreads 0 \
                  -force \
                  -quiet
            fi
                
            # Compute DEC maps if enabled
            if [[ $DEC -eq 1 ]]; then
                echo " INFO ${SUBJECT}: Computing DEC lesion TDMap for $TISSUE tissue"
                out_f="${SUBJECT_DIR}/maps/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion_DEC.nii.gz"
                if [[ ! -f $out_f ]]; then 
                  tckmap "${SUBJECT_DIR}/tracts/${SUBJECT}_tissue-${TISSUE}_TDMap-lesion.tck" \
                    $out_f \
                    -template $lesion \
                    -nthreads 0 \
                    -dec \
                    -force \
                    -quiet
                fi
            fi
        done        

        # Execute Python script
        echo " INFO ${SUBJECT}: Computing Tract density metrics"
        python TDMaps-extraction.py $MAIN_DIR $SUBJECT $IDH --min_streamlines $STREAM_D_TH --demographics $CLINICAL_DATA_FILE >> $DEMOGRAPHICS
        
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

echo "Tract density mapping finished. Do not go grab a beer, you need to take care of your health."