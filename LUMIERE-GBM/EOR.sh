#!/bin/bash

# Default values
IDH=WT
COMPARTMENT=1
N=1

# Usage function
usage() {
  echo "Usage: $0 [-i idh] [-c compartment] [-n n] [-s s]"
  echo "  -i IDH         : Specify the IDH1 statatus to analyze (default: WT; options: WT, MUT, or NOSNEC)"
  echo "  -c COMPARTMENT : Specify the label of the compartment to use (default: 1 for contrast-enhancing)"
  echo "  -n N           : Specify the number of subjects to parallelize (default: 1, options: 1, ..., N)"
  exit 1
}

# Parse command-line options
while getopts ":i:c:n:h" opt; do
  case ${opt} in
    i)
      IDH=$OPTARG
      ;;
    c)
      COMPARTMENT=$OPTARG
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
DATA_DIR="${MAIN_DIR}/LUMIERE-GBM_data-raw_v2"
DESTINATION_MAPS="${MAIN_DIR}/TDMaps_IDH1-${IDH}"
EOR="${MAIN_DIR}/data/LUMIERE_Extent-of-Resection.csv"
SEGMENTATION_MODEL="DeepBraTumIA-segmentation/native/segmentation"
CONTRAST="CT1"
WEEK_PREOP="week-000-1"
WEEK_POSTOP="week-000-2"

# Redirect standard output
exec 1> $MAIN_DIR"/data/Logs-EOR.txt"
ERROR_LOG=$MAIN_DIR"/data/Errors-EOR.txt"
exec 2> $ERROR_LOG

# Create the header
echo "Patient,Preop volume (cm3),Postop volume (cm3),Extent of Resection (%)" > $EOR

for lesion in $DATA_DIR/*; do
    
    ( 

      # Subject ID and directory
      SUBJECT=$(basename "$lesion")
      SUBJECT="${SUBJECT%%_*}"

      PREOP_SEG="${lesion}/${WEEK_PREOP}/${SEGMENTATION_MODEL}/${CONTRAST,,}_seg_mask.nii.gz"
      POSTOP_SEG="${lesion}/${WEEK_POSTOP}/${SEGMENTATION_MODEL}/${CONTRAST,,}_seg_mask.nii.gz"
      UNIQUE="${lesion}/week-000"

      if [[ -f "$PREOP_SEG" && -f "$POSTOP_SEG" ]]; then

        echo " INFO: ${SUBJECT}: Computing EOR."
        python compute_eor.py \
          --subject "$SUBJECT" \
          --preop_file "$PREOP_SEG" \
          --postop_file "$POSTOP_SEG" \
          --label "$COMPARTMENT" >> $EOR

      elif [[ -d "$UNIQUE" ]]; then

          echo "WARNING: ${SUBJECT}: No post surgical scan available." >> $ERROR_LOG
          python compute_eor.py \
              --subject    "$SUBJECT" \
              --preop_file  "empty"   \
              --postop_file "empty"   \
              --label "$COMPARTMENT" >> $EOR

      else

          echo "WARNING: ${SUBJECT}: Neither pre- nor post-operative scan found." >> $ERROR_LOG
          python compute_eor.py \
              --subject    "$SUBJECT" \
              --preop_file  "empty"   \
              --postop_file "empty"   \
              --label "$COMPARTMENT" >> $EOR

      fi
        
    ) &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

wait

echo " Extent of Resection calculated. I suppose the jokes about beers are no longer funny..."