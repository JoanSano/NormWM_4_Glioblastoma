#!/bin/bash

# Define base paths
BASE_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UCSF-PDGM-v3-20230111"
MNI_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UCSF-PDGM-v3-20230111/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM"
TARGET_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UCSF-PDGM-v3-20230111/UCSF-PDGM-v3_tmp-copy"
FILE="${BASE_DIR}/UCSF-PDGM-v3_MNI-ICBM-2009b-NLIN-ASYM/registered-subjects.txt"
if [[ -d "$TARGET_DIR" ]]; then
    echo "Target directory directory already exists"
else
    mkdir "$TARGET_DIR"
fi
echo "# Registered subjects to MNI" > "${FILE}"
N_SUBJECTS=0

# Iterate over each subject directory
for subject in "$BASE_DIR"/UCSF-PDGM-v3/*; do
    
    # Check if it's a directory
    if [[ -d "$subject" ]]; then
        # Set the subject directory path
        SUBJECT_DIR=$(basename "$subject")
        TARGET_SUBJECT="${TARGET_DIR}/${SUBJECT_DIR}"

        # Check if registration to MNI is complete
        if [[ -d "${MNI_DIR}/${SUBJECT_DIR}" ]]; then
            FILE_COUNT=$(find "${MNI_DIR}/${SUBJECT_DIR}" -type f | wc -l)
            # If the registration is complete there should 9 files (4 contrasts 1 segmentation and 4 transformation files)
            if [[ "${FILE_COUNT}" -eq 9 ]]; then
                echo "${SUBJECT_DIR}" >> "${FILE}"
                N_SUBJECTS=$((N_SUBJECTS + 1))

                # Move the subject to temporary copy folder
                mv $subject $TARGET_DIR
            fi
            #if 
            
        fi
    fi
done
echo "Number of registered subjects to MNI succesfully: ${N_SUBJECTS}" >> "${FILE}"
