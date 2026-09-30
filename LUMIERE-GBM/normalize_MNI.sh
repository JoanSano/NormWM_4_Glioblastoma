#!/bin/bash

# Define base paths
CONTRAST="CT1"
BASE_DIR="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_LUMIERE-GBM_v1-13122022"
if [ "$CONTRAST" = "CT1" ]; then
    MNI_TEMPLATE="/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM/T1_0.5mm_brain"
else
    MNI_TEMPLATE="/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM/${CONTRAST}_0.5mm_brain"
fi
MNI_TEMPLATE_mask="/home/joan/Documents/MNI_ICBM_2009b_NLIN_ASYM/T1_0.5mm_brain_mask.nii.gz"
MNI_DIR="/home/joan/Desktop/PROJECTS/Glioblastomas/Glioblastoma_LUMIERE-GBM_v1-13122022/LUMIERE-GBM_MNI-ICBM-2009b-NLIN-ASYM"
SEGMENTATION_MODEL="Manual-segmentation/native" #"DeepBraTumIA-segmentation/native/segmentation"
WEEK="week-000*" #-1
if [[ -d "$MNI_DIR" ]]; then
    echo "MNI directory already exists"
else
    mkdir "$MNI_DIR"
fi

# Redirect standard output
exec 1> $MNI_DIR"/Logs-NormalizeMNI.txt"
exec 2> $MNI_DIR"/Errors-NormalizeMNI.txt"

N=14

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

# Iterate over each subject directory
for subject in "$BASE_DIR"/LUMIERE-GBM_data-raw_V1/*; do
    (
        # Check if it's a directory
        if [[ -d "$subject" ]]; then
            for wk in "$subject"/$WEEK; do
                SUBJECT_DIR="$(basename "$subject")"
                if [[ -d "$wk" ]]; then
                    if [[ -d "$MNI_DIR"/"$SUBJECT_DIR" ]]; then
                        echo "  ${subject}: MNI subject directory already exists"
                    else
                        mkdir "$MNI_DIR"/"$SUBJECT_DIR"
                    fi
                    # Set the subject-week directory path
                    SUBJECT_DIR=$SUBJECT_DIR/"$(basename "$wk")"
                    REFERENCE_IMAGE=$(find "$wk" -name "$CONTRAST.nii.gz")
                    OUTPUT_NAME="$MNI_DIR"/"$SUBJECT_DIR"/$(basename "$REFERENCE_IMAGE")
                    # Delete extensions
                    OUTPUT_NAME=("${OUTPUT_NAME%.*}")
                    OUTPUT_NAME=("${OUTPUT_NAME%.*}")

                    if [[ -d "$MNI_DIR"/"$SUBJECT_DIR" ]]; then
                        echo "  ${wk}: MNI week directory already exists"
                    else
                        mkdir "$MNI_DIR"/"$SUBJECT_DIR"
                    fi
                    
                    if [[ -f "${OUTPUT_NAME}__1Warp.nii.gz" ]]; then
                        echo "  ${wk}: Transformation already available"
                    else
                        ### Masking the pathologic tissue
                        MASK_NAME=$(find "$wk/$SEGMENTATION_MODEL" -name "${CONTRAST,,}_seg_mask.nii.gz")
                        INVERSE_MASK="$wk"/inverse_segmentation.nii.gz
                        fslmaths "$MASK_NAME" -bin "$INVERSE_MASK"
                        ImageMath 3 ${INVERSE_MASK} Neg ${INVERSE_MASK}
                        if [[ -f "${INVERSE_MASK}" ]]; then
                            echo "  ${wk}: Inverse mask ready"
                        else
                            echo "  ${wk}: Inverse mask: Problematic!"
                        fi

                        ### Run the antsRegistrationSyN.sh command ###
                        antsRegistrationSyNQuick.sh -d 3 \
                            -f "$MNI_TEMPLATE".nii.gz \
                            -m "$REFERENCE_IMAGE" \
                            -o "${OUTPUT_NAME}"__ \
                            -j 1 \
                            -x "${MNI_TEMPLATE_mask}, ${INVERSE_MASK}" 

                        rm "$INVERSE_MASK" 
                    fi

                    ### Apply the registration ###                 
                    if [ ${#images[@]} -eq 0 ]; then
                        echo "  No images to apply the fitted transformation!"
                    else
                        for img in "${images[@]}"; do
                            if [ "$img" = "mask" ]; then
                                moving_IMAGE=$(find "$wk/$SEGMENTATION_MODEL" -name "${CONTRAST,,}_seg_mask.nii.gz" 2>/dev/null)
                            else
                                moving_IMAGE=$(find "$week" -name "*$img.nii.gz")
                            fi
                            OUTPUT_regis="$MNI_DIR"/"$SUBJECT_DIR"/$(basename "$moving_IMAGE")
                            OUTPUT_regis=("${OUTPUT_regis%.*}")
                            OUTPUT_regis=("${OUTPUT_regis%.*}")
                            
                            if [ -z "$moving_IMAGE" ]; then
                                echo "  ${wk}: No ${img} found likely due to <${SEGMENTATION_MODEL}> outputs"
                                continue
                            fi

                            echo "  ${wk}: Applying the warping to ${img}"

                            antsApplyTransforms -d 3 \
                                -i "${moving_IMAGE}" \
                                -r "${MNI_TEMPLATE}".nii.gz \
                                -o "${OUTPUT_regis}__Warped.nii.gz" \
                                -t "${OUTPUT_NAME}"__1Warp.nii.gz \
                                -t "${OUTPUT_NAME}"__0GenericAffine.mat \
                                -n NearestNeighbor
                        done
                    fi
                else
                    echo "Week $SUBJECT_DIR does not exist or is not a directory. Skipping..."
                fi
            done

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