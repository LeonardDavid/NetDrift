#!/bin/bash
##########################################################################################
#   LDB
#   
#   automation script for running RTM simulation
# 
#   args:
#   $1  NN_MODEL:   FMNIST  CIFAR   RESNET
#   $2  LOOPS:      Number of inference iterations
#   $3  GPU:        GPU to be used (0, 1)
#
##########################################################################################

NN_MODEL="$1"                       # FMNIST    CIFAR   RESNET
## params for rtm testing
TEST_ERROR=1
TEST_RTM=1
LOOPS=$2
BLOCK_SIZE=$3

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

NR_UNPROC=$4
echo ""
# Check what the argument contains
if [[ $NR_UNPROC == *"ALL"* ]]; then
    echo "Number of unprotected layers: ALL"
    # Do something for 'ALL' here
elif [[ $NR_UNPROC == *"CUSTOM"* ]]; then
    echo "Number of unprotected layers: CUSTOM"
    # Do something for 'CUSTOM' here
elif [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
    let "Np1=NR_UNPROC+1"
    echo "Number of unprotected layers: only $Np1"
    # Do something for uinteger here
else
    echo "Invalid NR_UNPROC 4th argument."
    # Break or exit the script
    exit 1
fi

GPU=$5

if [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    TEST_BATCH_SIZE=10000
    if [[ $NR_UNPROC == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0)
    elif [[ $NR_UNPROC == *"CUSTOM"* ]]; then
        # declare -a PROTECT_LAYERS=(0 1 1 1)
        declare -a PROTECT_LAYERS=(1 0 1 1)
        # declare -a PROTECT_LAYERS=(1 1 0 1)
        # declare -a PROTECT_LAYERS=(1 1 1 0)
    elif [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1)
        PROTECT_LAYERS[$NR_UNPROC]=0
    fi
    declare -a ERRSHIFTS=(0 0 0 0)
    # MODEL_PATH="models/model_fmnist9108.pt"
    # MODEL_PATH="models/model_fmnist5x5_9077.pt"
    # MODEL_PATH="models/model_fmnist7x7_8880.pt"
elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    TEST_BATCH_SIZE=10000
    if [[ $NR_UNPROC == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0 0 0 0 0)
    elif [[ $NR_UNPROC == *"CUSTOM"* ]]; then
        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1)
        declare -a PROTECT_LAYERS=(1 0 1 1 1 1 1 1)
        # declare -a PROTECT_LAYERS=(1 1 0 1 1 1 1 1)
        # declare -a PROTECT_LAYERS=(1 1 1 0 1 1 1 1)
        # declare -a PROTECT_LAYERS=(1 1 1 1 0 1 1 1)
        # declare -a PROTECT_LAYERS=(1 1 1 1 1 0 1 1)
        # declare -a PROTECT_LAYERS=(1 1 1 1 1 1 0 1)
        # declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 0)
    elif [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1)
        PROTECT_LAYERS[$NR_UNPROC]=0
    fi
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0)
    MODEL_PATH="models/model_cifar8582.pt"
    # MODEL_PATH="model_cifar8660.pt"
elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    TEST_BATCH_SIZE=256
    if [[ $NR_UNPROC == *"ALL"* ]]; then
        declare -a PROTECT_LAYERS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
    elif [[ $NR_UNPROC == *"CUSTOM"* ]]; then
    
        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
        declare -a PROTECT_LAYERS=(1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)

        # declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1)
        # declare -a PROTECT_LAYERS=(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0)
        # declare -a PROTECT_LAYERS=(1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
        # declare -a PROTECT_LAYERS=(0 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1)
    elif [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
        declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
        PROTECT_LAYERS[$NR_UNPROC]=0
    fi
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
    MODEL_PATH="models/model_resnet7694.pt"
else
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit
fi

## default params
BATCH_SIZE=256
EPOCHS=1
LR=0.001
STEP_SIZE=25

# echo -e "${PROTECT_LAYERS[@]}"


# declare -a PERRORS=(0.0)

# declare -a PERRORS=(0.1 0.01 0.001 0.0001)

# declare -a PERRORS=(0.1)
# declare -a PERRORS=(0.1 0.1)
declare -a PERRORS=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)

# declare -a PERRORS=(0.05)
# declare -a PERRORS=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

# declare -a PERRORS=(0.01)
# declare -a PERRORS=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)

# declare -a PERRORS=(0.001)
# declare -a PERRORS=(0.0001)

# declare -a PERRORS=(0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
# declare -a PERRORS=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)

# declare -a PERRORS=(0.0000455)
# declare -a PERRORS=(0.00001)
# declare -a PERRORS=(0.000001)

# declare -a PERRORS=(0.0001 0.0000455 0.00001 0.000001)

declare -a all_results  # Declare an array of arrays to store all results

for p in "${PERRORS[@]}"
do
    echo -e "\n\033[0;32mRunning $NN_MODEL for $LOOPS loops with error: $p\033[0m\n"
    
    declare -a list

    if [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
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

                python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU --block_size=$BLOCK_SIZE --protect_layers ${PROTECT_LAYERS[@]} --err_shifts ${ERRSHIFTS[@]} | tee "$output_file"
                
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
    else
        L=$NR_UNPROC
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

        python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --test-batch-size=${TEST_BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU --block_size=$BLOCK_SIZE --protect_layers ${PROTECT_LAYERS[@]} --err_shifts ${ERRSHIFTS[@]} | tee "$output_file"
        
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

# Specify the output file
out_results="$output_dir/all_results.txt"
out_results_python="all_results_2.txt"

echo "" > "$out_results_python"

> "$out_results"
> "$out_results_python"

# Loop through the outer array
for outer_idx in "${!all_results[@]}"; do
  # Print each inner array element with a space separator
  echo "${all_results[$outer_idx]}" >> "$out_results"
  echo "${all_results[$outer_idx]}" >> "$out_results_python"
done

echo ""
echo "Average accuracies:"
echo ""

python calculate_avg.py 2


if [[ $NR_UNPROC =~ ^[0-9]+$ ]]; then
    # echo -e "${PROTECT_LAYERS[@]}"
    PROTECT_LAYERS[$NR_UNPROC]=1
    # echo -e "${PROTECT_LAYERS[@]}"
fi

