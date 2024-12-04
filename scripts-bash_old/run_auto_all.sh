#!/bin/bash
##########################################################################################
#   LDB
#   
#   Automation script for running RTM misalignment fault simulation for BNNs
# 
#   ### Arguments:
#   `nn_model`:                 FMNIST, CIFAR, RESNET
#   `loops`:                    amount of loops (0, 100]
#   `block_size`:               aka racetrack/nanowire size (typically 64)
#   `layer_config`:             unprotected layers configuration (ALL, CUSTOM, INDIV)
#   `gpu_id`:                   ID of GPU to use for computations (0, 1) 
#
#   ### Optional arguments:
#   `global_bitflip_budget`:    default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
#   `local_bitflip_budget`:     default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.
#
#
##########################################################################################

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

## required args
NN_MODEL="$1"           # FMNIST    CIFAR   RESNET
LOOPS=$2
BLOCK_SIZE=$3
LAYER_CONFIG=$4
GPU_ID=$5
## optional args
GLOBAL_BITFLIP_BUDGET=${6:-0.0}
LOCAL_BITFLIP_BUDGET=${7:-0.0}


## create output directory
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
results_dir="RTM_results/$NN_MODEL/$BLOCK_SIZE/$timestamp"
output_dir="$results_dir/outputs"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    else
        echo "Directory $output_dir already exists."
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
    declare -a ERRSHIFTS=(0 0 0 0)
    
    if [[ $LAYER_CONFIG == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0)

    elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
        # declare -a PROTECT_LAYERS=(0 1 1 1)
        # declare -a PROTECT_LAYERS=(1 0 1 1)
        declare -a PROTECT_LAYERS=(1 1 0 1)
        # declare -a PROTECT_LAYERS=(1 1 1 0)

        # declare -a PROTECT_LAYERS=(0 0 1 0)
        # declare -a PROTECT_LAYERS=(0 0 0 0)

    elif [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1)
        PROTECT_LAYERS[$LAYER_CONFIG]=0
    fi

elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    MODEL_PATH="models/model_cifar8582.pt"
    # MODEL_PATH="model_cifar8660.pt"
    TEST_BATCH_SIZE=10000
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0)

    if [[ $LAYER_CONFIG == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0 0 0 0 0)

    elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1)
        declare -a PROTECT_LAYERS=(1 0 1 1 1 1 1 1)

        # declare -a PROTECT_LAYERS=(0 1 0 1 1 1 1 0)
        # declare -a PROTECT_LAYERS=(1 0 1 0 1 0 1 0)
        # declare -a PROTECT_LAYERS=(0 1 0 1 0 1 0 1)
        # declare -a PROTECT_LAYERS=(0 1 1 1 1 0 0 0)

    elif [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1)
        PROTECT_LAYERS[$LAYER_CONFIG]=0
    fi

elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    MODEL_PATH="models/model_resnet7694.pt"
    TEST_BATCH_SIZE=256
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

    if [[ $LAYER_CONFIG == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

    elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    
        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
        declare -a PROTECT_LAYERS=(1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)

        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1)
        # declare -a PROTECT_LAYERS=(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0)
        # declare -a PROTECT_LAYERS=(1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
        # declare -a PROTECT_LAYERS=(0 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1)

    elif [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
        PROTECT_LAYERS[$LAYER_CONFIG]=0
    fi

else
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling, capitalization & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit
fi
# echo -e "${PROTECT_LAYERS[@]}"


## Declare array of misalignment fault values (each entry represents a separate experiment execution, acting as a for-loop. Values can be distinct or equal or both)

# declare -a PERRORS=(0.0)

declare -a PERRORS=(0.1)
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


## Main loop
for p in "${PERRORS[@]}"
do
    echo -e "\n\033[0;32mRunning $NN_MODEL for $LOOPS loops with error: $p\033[0m\n"
    
    declare -a list     # stores results

    # in the case of INDIV, 
    if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        for layer in "${!PROTECT_LAYERS[@]}"    
        do
            if [ "${PROTECT_LAYERS[$layer]}" == 0 ]; then
                let "L=layer+1"
                echo -e "\n\033[0;33mUNprotected Layer: $L\033[0m\n"

                # PROTECT_LAYERS[$layer]=0
                # echo "${PROTECT_LAYERS[@]}"
                
                output_dir_L="$output_dir/$L"
                if [ ! -d "$output_dir_L" ]; then
                    mkdir -p "$output_dir_L"
                else
                    echo "Directory $output_dir_L already exists."
                fi
                output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

                python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --block_size=$BLOCK_SIZE --protect_layers ${PROTECT_LAYERS[@]} --err_shifts ${ERRSHIFTS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET | tee "$output_file"
                
                penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
                # Remove square brackets and split values
                values=$(echo "$penultimate_line" | tr -d '[]')

                list+=("$values")
                
                # echo $list
                
                python plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}

                # PROTECT_LAYERS[$layer]=1
            fi
        done
        all_results+=("$list")

    else    # in the case of ALL, CUSTOM

        L=$LAYER_CONFIG
        echo -e "\n\033[0;33mUNprotected Layer: $L\033[0m\n"

        # PROTECT_LAYERS[$layer]=0
        # echo "${PROTECT_LAYERS[@]}"
        
        output_dir_L="$output_dir/$L"
        if [ ! -d "$output_dir_L" ]; then
            mkdir -p "$output_dir_L"
        else
            echo "Directory $output_dir_L already exists."
        fi
        output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

        python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --block_size=$BLOCK_SIZE --protect_layers ${PROTECT_LAYERS[@]} --err_shifts ${ERRSHIFTS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET | tee "$output_file"
        
        penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
        # Remove square brackets and split values
        values=$(echo "$penultimate_line" | tr -d '[]')

        list+=("$values")
        
        # echo $list
        
        python plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}
        
        all_results+=("$list")
    fi

    csv_file="$output_dir/table_$p.csv"

    for value in "${list[@]}"
    do
        echo "${value[@]}" >> "$csv_file"
    done

    unset list
    
done

# for list in "${all_results[@]}"
# do
#     echo $list
# done

# # Specify the output file
# out_results="$output_dir/all_results.txt"
# out_results_python="all_results.txt"

# > "$out_results"
# > "$out_results_python"

# # Loop through the outer array
# for outer_idx in "${!all_results[@]}"; do
#   # Print each inner array element with a space separator
#   echo "${all_results[$outer_idx]}" >> "$out_results"
#   echo "${all_results[$outer_idx]}" >> "$out_results_python"
# done

# echo ""
# echo "Average accuracies:"
# echo ""

# python calculate_avg.py

## Reset layer configuration in case of INDIV
if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
    # echo -e "${PROTECT_LAYERS[@]}"
    PROTECT_LAYERS[$LAYER_CONFIG]=1
    # echo -e "${PROTECT_LAYERS[@]}"
fi

