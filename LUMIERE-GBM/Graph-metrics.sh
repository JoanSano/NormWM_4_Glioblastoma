#!/bin/bash

# Default values
IDH=WT
STREAM_D_TH=0
TISSUE=whole # "core" "nonenhancing" "enhancing" "core+enhancing"
MODE=default # "ScaleInvnodevol" "ScaleInvlength" "ScaleLength"
N=1

# Usage function
usage() {
  echo "Usage: $0 [-i idh] [-s s] [-t t] [-m m] [-n n]"
  echo "  -i IDH         : Specify the IDH1 statatus to analyze (default: WT; options: WT, MUT, Inconclusive, or UNKOWN)"
  echo "  -s STREAM_D_TH : Specify the minimum number of streamline density to threshold (default=0, option: 0, 1, 10, ...)"
  echo "  -t TISSUE      : Specify the tumor tissue (default=whole, options: whole, core, nonenhancing, enhancing, core+enhancing"
  echo "  -m MODE        : Specify the type of connectome to process (default=default, options: default, ScaleInvnodevol, ScaleInvlength,ScaleLength"
  echo "  -n N           : Specify the number of subjects to parallelize (default: 1, options: 1, ..., N)"

  exit 1
}

# Parse command-line options
while getopts ":i:s:t:m:n:h" opt; do
  case ${opt} in
    i)
      IDH=$OPTARG
      ;;
    s)
      STREAM_D_TH=$OPTARG
      ;;
    t)
      TISSUE=$OPTARG
      ;;
    m)
      MODE=$OPTARG
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
DESTINATION="${MAIN_DIR}/TDMaps_IDH1-${IDH}"
CLINICAL_DATA_FILE="${MAIN_DIR}/data/LUMIERE-Demographics_Pathology-v2.csv"
HCPEX="/home/joan/Documents/Parcellations/HCPex__Glasser-like+subcortical/MNI_ICBM_2009b_NLIN_ASYM/HCPex_MNI_ICBM_2009b_NLIN_ASYM.nii.gz"
AAL3="/home/joan/Documents/Parcellations/AAL3/MNI_ICBM_2009b_NLIN_ASYM/AAL3v1_1mmxMNI_ICBM_2009b_NLIN_ASYM__Warped.nii.gz"
DEMOGRAPHICS_HCPEX="${DESTINATION}/demographics-metrics_streamTH-${STREAM_D_TH}_tissue-${TISSUE}_connectome-${MODE}_atlas-HCPEx.csv"
DEMOGRAPHICS_AAL3="${DESTINATION}/demographics-metrics_streamTH-${STREAM_D_TH}_tissue-${TISSUE}_connectome-${MODE}_atlas-AAL3.csv"
LABELS_HCPEX="/home/joan/Documents/Parcellations/HCPex__Glasser-like+subcortical/HCPex-main/HCPex_v1.0/HCPex.nii.txt"
LABELS_AAL3="/home/joan/Documents/Parcellations/AAL3/AAL3-main/AAL3v1.nii.txt"

# Redirect standard output
exec 1> $DESTINATION"/Logs-GraphMetrics.txt"
exec 2> $DESTINATION"/Errors-GraphMetrics.txt"

# Create the demographics header
METRICS_LABELS="size,density,avg_degree,avg_clustering,modularity,global_efficiency,avg_local_efficiency,avg_shortest_path_length,avg_eigenvector_centrality,avg_betweenness_centrality,avg_closeness_centrality,avg_edge_betwenness_centrality,median_edge_betwenness_centrality,min_edge_betwenness_centrality,max_edge_betwenness_centrality,avg_percolation_centrality,median_percolation_centrality,min_percolation_centrality,max_percolation_centrality,avg_participation_coef,median_participation_coef,min_participation_coef,max_participation_coef,deg_assortativity,powerlaw_exponent,LLR_distribution,distribution,s_metric_SF,avg_rich_club_coef,median_rich_club_coef,min_rich_club_coef,max_rich_club_coef" 
echo "Patient,Survival time (weeks),Sex,Age at surgery (years),IDH (WT: wild type),IDH method,MGMT qualitative,MGMT quantitative,${METRICS_LABELS}" > $DEMOGRAPHICS_HCPEX
echo "Patient,Survival time (weeks),Sex,Age at surgery (years),IDH (WT: wild type),IDH method,MGMT qualitative,MGMT quantitative,${METRICS_LABELS}" > $DEMOGRAPHICS_AAL3

for lesion in $DESTINATION/*/; do 
    ( 
        # Subject ID and directory
        SUBJECT=$(basename "$lesion")
        SUBJECT="${SUBJECT%%_*}"
        SUBJECT_DIR="${DESTINATION}/${SUBJECT}"        
        if [[ ! -d $SUBJECT_DIR"/metrics" ]]; then mkdir $SUBJECT_DIR"/metrics"; fi

        # Parcellation HCPEx
        LESION_CONNECTOME="${SUBJECT_DIR}/connectomes/${SUBJECT}_tissue-${TISSUE}_connectome-${MODE}_parcellation-HCPEx.csv"
        LESION_METRICS="${SUBJECT_DIR}/metrics/${SUBJECT}_streamTH-${STREAM_D_TH}_tissue-${TISSUE}_metrics-${MODE}_parcellation-HCPEx"
        if [[ ! -f $LESION_METRICS ]]; then 
          echo " INFO ${SUBJECT}: Computing graph metrics for tissue $TISSUE, mode $MODE, and parcellation HCPEx"
          python Network-metrics.py $DESTINATION \
              $SUBJECT \
              $LESION_CONNECTOME \
              $LESION_METRICS \
              $CLINICAL_DATA_FILE \
              --atlas $LABELS_HCPEX \
              --min_streamlines $STREAM_D_TH >> $DEMOGRAPHICS_HCPEX
        fi

        # AAL3
        LESION_CONNECTOME="${SUBJECT_DIR}/connectomes/${SUBJECT}_tissue-${TISSUE}_connectome-${MODE}_parcellation-AAL3.csv"
        LESION_METRICS="${SUBJECT_DIR}/metrics/${SUBJECT}_streamTH-${STREAM_D_TH}_tissue-${TISSUE}_metrics-${MODE}_parcellation-AAL3"
        if [[ ! -f $LESION_METRICS ]]; then 
          echo " INFO ${SUBJECT}: Computing graph-metrics for tissue $TISSUE, mode $MODE, and parcellation AAL3"
          python Network-metrics.py $DESTINATION \
              $SUBJECT \
              $LESION_CONNECTOME \
              $LESION_METRICS \
              $CLINICAL_DATA_FILE \
              --atlas $LABELS_AAL3 \
              --min_streamlines $STREAM_D_TH >> $DEMOGRAPHICS_AAL3
          fi
        
    ) &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

wait

echo "Computation of graph metrics has finished. Whether or not you grab a beer will have no consequences on the date of your death, do whatever you feel like doing..."