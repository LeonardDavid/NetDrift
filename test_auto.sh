#!/bin/bash

################################################################################
#   CASE ($1)
#   I.      ptest 0.1   + Lxy
#   II.     ptest 0.1   + Lall
#   III.    ptest 10^-4 + Lxy
#   IV.     ptest 10^-4 + Lall
#
################################################################################

# source flags and colour definitions from conf file
source flags.conf 

CASE=$1

# declare -a PERRORS=(0.0)

END1="END1"
END2="END2"
OPERATION="TEST_AUTO_$CASE"
KERNEL_SIZE=3
LOOPS=10
RT_SIZE=64
GPU_ID=0

# PHASE_DIR="phase0"
# PHASE_DIR="phase1"
# PHASE_DIR="phase2"
# PHASE_DIR="phase3"
# PHASE_DIR="phase4"
# PHASE_DIR="mm1"
# PHASE_DIR="mm2"
# PHASE_DIR="mm3"
PHASE_DIR="mm4"

MODEL_DIR="models/$PHASE_DIR"

start_time=$(date +%s.%N)
for MODEL_PATH in "$MODEL_DIR"/*; do

    if [[ "$MODEL_PATH" == *.pt ]]; then
        echo -e "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

        MODEL_NAME=$(basename "$MODEL_PATH") # Get the file name without the path
        EXTENSION="${MODEL_NAME##*.}" # Extract the file extension
        MODEL_NAME_NOEXT="${MODEL_NAME%.*}" # Remove the extension to process the base name

        # Split the base name by underscores
        IFS='_' read -r -a parts <<< "$MODEL_NAME_NOEXT"

        NN_MODEL=${parts[1]^^} # capitalized
        L_TEST=${parts[5]}

        # Remove the "L" and add spaces between digits
        NR_STRING="${L_TEST:1}"
        NR_SPACED_STRING=$(echo "$NR_STRING" | sed 's/./& /g')

        # Convert spaced string to an array of integers
        read -r -a UNPROT_LAYER_IDS <<< "$NR_SPACED_STRING"

        if [[ "$CASE" == "1" ]];
        then
            # Calculate baseline at first using PERROR=0.0

            # declare -a PERRORS=(0.1)
            # declare -a PERRORS=(0.0 0.1 0.1)
            declare -a PERRORS=(0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
            
            LAYER_CONFIG="CUSTOM"

        elif [[ "$CASE" == "2" ]];
        then
            # Calculate baseline at first using PERROR=0.0

            # declare -a PERRORS=(0.1)
            # declare -a PERRORS=(0.0 0.1 0.1)
            declare -a PERRORS=(0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)

            LAYER_CONFIG="ALL"

        elif [[ "$CASE" == "3" ]];
        then
            # Calculate baseline at first using PERROR=0.0

            # declare -a PERRORS=(0.0001)
            declare -a PERRORS=(0.0 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
            
            LAYER_CONFIG="CUSTOM"

        elif [[ "$CASE" == "4" ]];
        then 
            # Calculate baseline at first using PERROR=0.0

            # declare -a PERRORS=(0.0001)
            declare -a PERRORS=(0.0 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)

            LAYER_CONFIG="ALL"

        else
            echo -e "${RED}Invalid CASE $CASE${RESET}"
            exit 1
        fi

        if [[ "$LAYER_CONFIG" == "CUSTOM" ]]; then
            bash run.sh $OPERATION "${PERRORS[@]}" PERRORS $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE "${UNPROT_LAYER_IDS[@]}" $LAYER_CONFIG $GPU_ID $MODEL_PATH
        
        elif [[ "$LAYER_CONFIG" == "ALL" ]]; then
            bash run.sh $OPERATION "${PERRORS[@]}" PERRORS $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $LAYER_CONFIG $GPU_ID $MODEL_PATH
        
        fi
    fi

done
end_time=$(date +%s.%N)

testing_time=$(echo "$end_time - $start_time" | bc)
hours=$(printf "%02d" $(echo "$testing_time/3600" | bc))
minutes=$(printf "%02d" $(echo "($testing_time%3600)/60" | bc))
seconds=$(printf "%06.3f" $(echo "$testing_time%60" | bc))

echo -e "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo -e "\n${CYAN}Total $OPERATION testing times: ${hours}:${minutes}:${seconds}${RESET}"

# plot results
RESULTS_DIR="$MODEL_DIR/$OPERATION"

echo ""
echo -e "${YELLOW}Plotting results for $OPERATION${RESET}"
python scripts-python/plot_auto.py ${RESULTS_DIR} 
echo -e "${GREEN}Done plotting results for $OPERATION${RESET}"