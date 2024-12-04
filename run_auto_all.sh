#!/bin/bash

##########################################################################################
#   LDB
#   
#   Automation script for running RTM misalignment fault simulation for BNNs
# 
#   ### Arguments:
#
#   `PROTECT_LAYERS`:           array of layers to be protected (in the format PROTECT_LAYERS[layer_id+1]={1:protected | 0:unprotected})
#   END1:                       first array termination token
#   `PERRORS`:                  Array of misalignment fault rates to be tested (floats)
#   END2:                       second array termination token
#   `nn_model`:                 FMNIST, CIFAR, RESNET
#   `loops`:                    amount of loops (0, 100]
#   `rt_size`:                  racetrack/nanowire size (typically 64)
#   `layer_config`:             unprotected layers configuration (ALL, CUSTOM, INDIV)
#   `gpu_id`:                   ID of GPU to use for computations (0, 1) 
#
#   ### Optional arguments:
#   `global_bitflip_budget`:    default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
#   `local_bitflip_budget`:     default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.
#
#
##########################################################################################

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
RESET='\033[0m' # Reset to default

# Read the first array - PROTECT_LAYERS (assumes the array ends with "END1")
PROTECT_LAYERS=()
while [[ $1 != "END1" ]]; do
    PROTECT_LAYERS+=("$1")
    shift
done
shift # Skip the "END1" token

# Read the second array - PERRORS (assumes the array ends with "END2")
PERRORS=()
while [[ $1 != "END2" ]]; do
    PERRORS+=("$1")
    shift
done
shift # Skip the "END2" token

# # Print arrays
# echo "PROTECT_LAYERS: ${PROTECT_LAYERS[@]}"
# echo "PERRORS: ${PERRORS[@]}"

# # Remaining arguments
# echo "Other arguments: $@"

# required args
NN_MODEL="$1"   # FMNIST    CIFAR   RESNET
shift
LOOPS=$1
shift
RT_SIZE=$1
shift
LAYER_CONFIG=$1
shift
GPU_ID=$1
shift

# echo "$NN_MODEL"
# echo "$LOOPS"
# echo "$RT_SIZE"
# echo "$LAYER_CONFIG"
# echo "$GPU_ID"

# optional args
GLOBAL_BITFLIP_BUDGET=${1:-0.0}
shift
LOCAL_BITFLIP_BUDGET=${1:-0.0}
shift

# echo "$GLOBAL_BITFLIP_BUDGET"
# echo "$LOCAL_BITFLIP_BUDGET"


## params for rtm testing
TEST_ERROR=1
TEST_RTM=1
## default params
BATCH_SIZE=256
EPOCHS=1
LR=0.001
STEP_SIZE=25

## variables & arrays
declare -a all_results  # array of arrays to store all results

# ## required args
# NN_MODEL="$1"           # FMNIST    CIFAR   RESNET
# LOOPS=$2
# RT_SIZE=$3
# LAYER_CONFIG=$4
# GPU_ID=$5
# ## optional args
# GLOBAL_BITFLIP_BUDGET=${6:-0.0}
# LOCAL_BITFLIP_BUDGET=${7:-0.0}


## create output directory
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
results_dir="RTM_results/$NN_MODEL/$RT_SIZE/$timestamp"
output_dir="$results_dir/outputs"

## TODO -> flags whether to calculate some stuff or not
all_bitflips_dir="$results_dir/all_bitflips"
all_affected_rts_dir="$results_dir/all_affected_rts"
all_misalign_faults_dir="$results_dir/all_misalign_faults"
all_results_dir="$results_dir/all_results"

if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"

    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    else
        echo "Directory $output_dir already exists."
    fi

## TODO -> flags whether to calculate some stuff or not
    if [ ! -d "$all_bitflips_dir" ]; then
        mkdir -p "$all_bitflips_dir"
    else
        echo "Directory $all_bitflips_dir already exists."
    fi

    if [ ! -d "$all_affected_rts_dir" ]; then
        mkdir -p "$all_affected_rts_dir"
    else
        echo "Directory $all_affected_rts_dir already exists."
    fi

    if [ ! -d "$all_misalign_faults_dir" ]; then
        mkdir -p "$all_misalign_faults_dir"
    else
        echo "Directory $all_misalign_faults_dir already exists."
    fi

    if [ ! -d "$all_results_dir" ]; then
        mkdir -p "$all_results_dir"
    else
        echo "Directory $all_results_dir already exists."
    fi

else
    echo "Directory $results_dir already exists."
fi

echo ""

## Check what the argument contains
if [[ $LAYER_CONFIG == *"ALL"* ]]; then
    echo "Number of unprotected layers: ALL"
elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    echo "Number of unprotected layers: CUSTOM"
elif [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
    let "Np1=LAYER_CONFIG+1"
    echo "Number of unprotected layers: only $Np1"
else
    echo "Invalid layer configuration (4th argument)."
    # Break or exit the script
    exit 1
fi

## Initialize parameters based on NN_MODEL
if [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    MODEL_PATH="models/model_fmnist9108.pt"
    # MODEL_PATH="models/model_fmnist5x5_9077.pt"
    # MODEL_PATH="models/model_fmnist7x7_8880.pt"
    TEST_BATCH_SIZE=10000

elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    MODEL_PATH="models/model_cifar8582.pt"
    # MODEL_PATH="model_cifar8660.pt"
    TEST_BATCH_SIZE=10000

elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    MODEL_PATH="models/model_resnet7694.pt"
    TEST_BATCH_SIZE=256

else
    echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: FMNIST, CIFAR, RESNET${RESET}\n"
    exit
fi
# echo -e "${PROTECT_LAYERS[@]}"


## Declare array of misalignment fault values (each entry represents a separate experiment execution, acting as a for-loop. Values can be distinct or equal or both)

# declare -a PERRORS=(0.0)

# declare -a PERRORS=(0.1)
# declare -a PERRORS=(0.1 0.1)
# declare -a PERRORS=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)

# declare -a PERRORS=(0.05)
# declare -a PERRORS=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

# declare -a PERRORS=(0.01)
# declare -a PERRORS=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)

# declare -a PERRORS=(0.001)
# declare -a PERRORS=(0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)

# declare -a PERRORS=(0.0001)
# declare -a PERRORS=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)

# declare -a PERRORS=(0.0000455)
# declare -a PERRORS=(0.00001)
# declare -a PERRORS=(0.000001)

# declare -a PERRORS=(0.1 0.01 0.001 0.0001)
# declare -a PERRORS=(0.0001 0.0000455 0.00001 0.000001)


## TODO -> flags whether to calculate some stuff or not
# out_results_file_bb="$all_results_dir/all_results-$GLOBAL_BITFLIP_BUDGET-$LOCAL_BITFLIP_BUDGET.txt"
out_results_file="$all_results_dir/all_results.txt"
echo "" > "$out_results_file"
> "$out_results_file"

out_bitflips_file="$all_bitflips_dir/all_bitflips.txt"
echo "" > "$out_bitflips_file"
> "$out_bitflips_file"

out_misalign_faults_file="$all_misalign_faults_dir/all_misalign_faults.txt"
echo "" > "$out_misalign_faults_file"
> "$out_misalign_faults_file"

out_affected_rts_file="$all_affected_rts_dir/all_affected_rts.txt"
echo "" > "$out_affected_rts_file"
> "$out_affected_rts_file"

declare -a all_results  # Declare an array of arrays to store all results


## Main loop
for p in "${PERRORS[@]}"
do
    echo -e "\n${GREEN}Running $NN_MODEL for $LOOPS loops with error: $p${RESET}\n"
    
    declare -a list     # stores results

    # in the case of INDIV, 
    if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        for layer in "${!PROTECT_LAYERS[@]}"    
        do
            if [ "${PROTECT_LAYERS[$layer]}" == 0 ]; then
                let "L=layer+1"
                echo -e "\n${YELLOW}UNprotected Layer: $L${RESET}\n"

                # PROTECT_LAYERS[$layer]=0
                # echo "${PROTECT_LAYERS[@]}"
                
                output_dir_L="$output_dir/$L"
                if [ ! -d "$output_dir_L" ]; then
                    mkdir -p "$output_dir_L"
                else
                    echo "Directory $output_dir_L already exists."
                fi
                output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

                python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET | tee "$output_file"
                
## TODO -> flags whether to calculate some stuff or not
                affected_rts_line=$(tail -n 8 "$output_file" | head -n 1)
                echo $affected_rts_line >> "$out_affected_rts_file"

                misalign_faults_line=$(tail -n 6 "$output_file" | head -n 1)
                echo $misalign_faults_line >> "$out_misalign_faults_file"

                bitflips_line=$(tail -n 4 "$output_file" | head -n 1)
                echo $bitflips_line >> "$out_bitflips_file"

                penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
                # Remove square brackets and split values
                values=$(echo "$penultimate_line" | tr -d '[]')

                list+=("$values")
                
                # echo $list
                
                python scripts-python/plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}

                # PROTECT_LAYERS[$layer]=1
            fi
        done
        all_results+=("$list")

    else    # in the case of ALL, CUSTOM

        L=$LAYER_CONFIG
        echo -e "\n${YELLOW}UNprotected Layer: $L${RESET}\n"

        # PROTECT_LAYERS[$layer]=0
        # echo "${PROTECT_LAYERS[@]}"
        
        output_dir_L="$output_dir/$L"
        if [ ! -d "$output_dir_L" ]; then
            mkdir -p "$output_dir_L"
        else
            echo "Directory $output_dir_L already exists."
        fi
        output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

        python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET | tee "$output_file"
        
## TODO -> flags whether to calculate some stuff or not
        affected_rts_line=$(tail -n 8 "$output_file" | head -n 1)
        echo $affected_rts_line >> "$out_affected_rts_file"

        misalign_faults_line=$(tail -n 6 "$output_file" | head -n 1)
        echo $misalign_faults_line >> "$out_misalign_faults_file"

        bitflips_line=$(tail -n 4 "$output_file" | head -n 1)
        echo $bitflips_line >> "$out_bitflips_file"

        penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
        # Remove square brackets and split values
        values=$(echo "$penultimate_line" | tr -d '[]')

        list+=("$values")
        
        # echo $list
        
        python scripts-python/plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}
        
        all_results+=("$list")
    fi

    # csv_file="$output_dir/table_$p.csv"
    # for value in "${list[@]}"
    # do
    #     echo "${value[@]}" >> "$csv_file"
    # done

    unset list
    
done


## TODO -> flags whether to calculate some stuff or not
# Loop through the outer array
for outer_idx in "${!all_results[@]}"; do
  # Print each inner array element with a space separator
  echo "${all_results[$outer_idx]}" >> "$out_results_file"
done


echo ""
echo "============================="
echo -e "${CYAN}Average accuracies${RESET}"
echo "(for each inference iteration across PERRORS misalignment fault rates):"
echo ""
python scripts-python/calculate_avg.py ${out_results_file}
echo "============================="


echo ""
echo -e "${CYAN}Average bitflips${RESET}"
echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
echo ""
python scripts-python/calculate_avg_matrix.py ${out_bitflips_file} 1


echo ""
echo -e "${CYAN}Average misalign_faults${RESET}"
echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
echo ""
python scripts-python/calculate_avg_matrix.py ${out_misalign_faults_file} 1


echo ""
echo -e "${CYAN}Average affected_rts${RESET}"
echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
echo ""
python scripts-python/calculate_avg_matrix.py ${out_affected_rts_file} 1


## Reset layer configuration in case of INDIV
if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
    # echo -e "${PROTECT_LAYERS[@]}"
    PROTECT_LAYERS[$LAYER_CONFIG]=1
    # echo -e "${PROTECT_LAYERS[@]}"
fi

