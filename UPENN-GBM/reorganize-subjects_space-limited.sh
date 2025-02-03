#!/bin/bash

# Define base paths
BASE_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024"
MNI_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024/UPENN-GBM_MNI-ICBM-2009b-NLIN-ASYM"
TARGET_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024/UPENN-GBM_data-raw_V3-organized_tmp-copy"
FILE="${MNI_DIR}/registered-subjects.txt"
if [[ -d "$TARGET_DIR" ]]; then
    echo "Target directory directory already exists"
else
    mkdir "$TARGET_DIR"
fi
echo "# Registered subjects to MNI" > "${FILE}"
N_SUBJECTS=0

# Iterate over each subject directory
for subject in "$BASE_DIR"/UPENN-GBM_data-raw_V3-organized/*; do
    
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
                echo "${SUBJECT_DIR} with ${FILE_COUNT} registration files" >> "${FILE}"
                N_SUBJECTS=$((N_SUBJECTS + 1))

                # Move the subject to temporary copy folder
                mv $subject $TARGET_DIR
            fi
            # If the registration is complete there should 10 files (4 contrasts 2 segmentation and 4 transformation files)
            if [[ "${FILE_COUNT}" -eq 10 ]]; then
                echo "${SUBJECT_DIR} with ${FILE_COUNT} registration files" >> "${FILE}"
                N_SUBJECTS=$((N_SUBJECTS + 1))

                # Move the subject to temporary copy folder
                mv $subject $TARGET_DIR
            fi 
            # If the registration is complete there should 10 files (4 contrasts 0 segmentation and 4 transformation files)
            if [[ "${FILE_COUNT}" -eq 8 ]]; then
                echo "${SUBJECT_DIR} with ${FILE_COUNT} registration files (SEGMENTATION PROBABLY NOT AVAILABLE)" >> "${FILE}"
                N_SUBJECTS=$((N_SUBJECTS + 1))

                # Move the subject to temporary copy folder
                mv $subject $TARGET_DIR
            fi 
        fi
    fi
done
echo "Number of registered subjects to MNI succesfully: ${N_SUBJECTS}" >> "${FILE}"
echo "Number of registered subjects to MNI succesfully: ${N_SUBJECTS}"