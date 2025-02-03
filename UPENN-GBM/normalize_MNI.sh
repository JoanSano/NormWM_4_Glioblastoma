#!/bin/bash

### WARNING: For the UPENN dataset, be sure to run the following command from the directory where the raw data is stored:
# find . ! -path "*approx*" -type f -name "*_segm.nii.gz" -exec bash -c 'mv "$0" "$(dirname "$0")/$(basename "$0" | sed "s/_segm/_corrected_segm/")"' {} \;
# It will rename the ID_11_segm.nii.gz files to ID_11_corrected_segm.nii.gz files
 
# Define base paths
CONTRAST="T2"
BASE_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024"
MNI_TEMPLATE="/home/sano/Documents/Joan/Data/MNI_ICBM_2009b_NLIN_ASYM/${CONTRAST}_0.5mm_brain"
MNI_DIR="/home/sano/Documents/Joan/Data/Glioblastoma_UPENN-GBM_v2-20221024/UPENN-GBM_MNI-ICBM-2009b-NLIN-ASYM"
if [[ -d "$MNI_DIR" ]]; then
    echo "MNI directory already exists"
else
    mkdir "$MNI_DIR"
fi

N=4

# Iterate over each subject directory
for subject in "$BASE_DIR"/UPENN-GBM_data-raw_V3-organized/*; do
    (
        # Check if it's a directory
        if [[ -d "$subject" ]]; then
            # Set the subject directory path
            SUBJECT_DIR=$(basename "$subject")
            REFERENCE_IMAGE=$(find "$subject" -name "*${CONTRAST}.nii.gz")
            OUTPUT_NAME="$MNI_DIR"/"$SUBJECT_DIR"/$(basename "$REFERENCE_IMAGE")
            # Delete extensions
            OUTPUT_NAME=("${OUTPUT_NAME%.*}")
            OUTPUT_NAME=("${OUTPUT_NAME%.*}")
            
            if [[ -d "$MNI_DIR"/"$SUBJECT_DIR" ]]; then
                echo "MNI subject directory already exists"
            else
                mkdir "$MNI_DIR"/"$SUBJECT_DIR"
            fi

            if [[ -f "${OUTPUT_NAME}__1Warp.nii.gz" ]]; then
                echo "  Transformation already available"
            else
                MASK_NAME=$(find "$subject" -name "*_segm.nii.gz")
                if [[ ! -f "$MASK_NAME" ]]; then
                    MASK_NAME=$(find "$subject" -name "*_automated_approx_segm.nii.gz")
                fi
                INVERSE_MASK="$subject"/inverse_segmentation.nii.gz
                fslmaths "$MASK_NAME" -bin "$INVERSE_MASK"
                ImageMath 3 ${INVERSE_MASK} Neg ${INVERSE_MASK}

                ### Perform bias field correction
                echo "Running N4BiasFieldCorrection on ${SUBJECT_DIR}"
                N4BiasFieldCorrection -d 3 \
                    -i "$REFERENCE_IMAGE" \
                    -o "${OUTPUT_NAME}_biasfieldcorrected.nii.gz"

                ### Run the antsRegistrationSyN.sh command ###
                echo "Running antsRegistrationSyNQuick on ${SUBJECT_DIR}"
                antsRegistrationSyNQuick.sh -d 3 \
                    -f "${MNI_TEMPLATE}.nii.gz" \
                    -m "${OUTPUT_NAME}_biasfieldcorrected.nii.gz" \
                    -o "${OUTPUT_NAME}"__ \
                    -j 1 \
                    -x "${MNI_TEMPLATE}_mask.nii.gz" "$INVERSE_MASK" \
                    -n 4 # Number of threads. BE CAREFUL TO MATCH THE NUMBER OF PROCESSES AND THE NUMBER OF CPU CORES IN THE MACHINE

                rm "$INVERSE_MASK"
                rm "${OUTPUT_NAME}_biasfieldcorrected.nii.gz"
            fi

            ### Apply the registration ###
            images=()
            # Parse the options using getopts
            while getopts ":f:" opt; do
                case ${opt} in
                    f)
                        # If -f is provided, add the images following -f to the images array
                        images+=("${OPTARG}")
                        shift $((OPTIND - 1)) # Skip the processed option and its argument
                        images+=("$@") # Collect remaining arguments as images
                        break
                        ;;
                    \?)
                        echo "Invalid option: -$OPTARG" >&2
                        exit 1
                        ;;
                    :)
                        echo "Option -$OPTARG requires an argument." >&2
                        exit 1
                        ;;
                esac
            done
            
            # Exit and finish if no images were provided with -f
            if [ ${#images[@]} -eq 0 ]; then
                echo "  No images to apply the fitted transformation!"
            else
                for img in "${images[@]}"; do
                    moving_IMAGE=$(find "$subject" -name "*$img.nii.gz")
                    OUTPUT_regis="$MNI_DIR"/"$SUBJECT_DIR"/$(basename "$moving_IMAGE")
                    OUTPUT_regis=("${OUTPUT_regis%.*}")
                    OUTPUT_regis=("${OUTPUT_regis%.*}")

                    if [[ -f $moving_IMAGE ]]; then
                        echo "  Applying the warping to ${img}"
                        
                        if [[ "${moving_IMAGE}" != *segm* ]]; then
                            N4BiasFieldCorrection -d 3 \
                                -i "${moving_IMAGE}" \
                                -o "${OUTPUT_regis}_biasfieldcorrected.nii.gz"

                            antsApplyTransforms -d 3 \
                            -i "${OUTPUT_regis}_biasfieldcorrected.nii.gz" \
                            -r "${MNI_TEMPLATE}".nii.gz \
                            -o "${OUTPUT_regis}__Warped.nii.gz" \
                            -t "${OUTPUT_NAME}"__1Warp.nii.gz \
                            -t "${OUTPUT_NAME}"__0GenericAffine.mat \
                            -n NearestNeighbor

                            rm "${OUTPUT_regis}_biasfieldcorrected.nii.gz"
                        else
                            antsApplyTransforms -d 3 \
                                -i $moving_IMAGE \
                                -r "${MNI_TEMPLATE}".nii.gz \
                                -o "${OUTPUT_regis}__Warped.nii.gz" \
                                -t "${OUTPUT_NAME}"__1Warp.nii.gz \
                                -t "${OUTPUT_NAME}"__0GenericAffine.mat \
                                -n NearestNeighbor
                        fi
                    else
                        echo "  The NIFTI image '''${img}''' does not exist"
                    fi
                done
            fi

        else
            echo "Directory $subject does not exist or is not a directory. Skipping..."
        fi
    
    ) &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

wait

echo "Registration of the dataset has finished! Go grab a drink, you deserve it"