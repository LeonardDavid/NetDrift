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
#   `PERRORS`:                  Array of misalignment fault rates to be tested/trained (floats)
#   END2:                       second array termination token
#   `OPERATION`:                TRAIN or TEST 
#   `kernel_size`:              size of kernel used for calculations in convolutional layers (if none then use 0)
#   `nn_model`:                 FMNIST, CIFAR, RESNET
#   `loops`:                    amount of loops the experiment should run (0, 100] (not used if OPERATION=TEST)
#   `rt_size`:                  racetrack/nanowire size (typically 64)
#   `layer_config`:             unprotected layers configuration (ALL, CUSTOM, INDIV)
#   `gpu_id`:                   ID of GPU to use for computations (0, 1) 
#
#   ### Additional required arguments if OPERATION = TRAIN
#   `epochs`:                   number of training epochs
#   `batch_size`:               batch size
#   `lr`:                       learning rate
#   `step_size`:                step size
#
#   ### Optional arguments:
#   `global_bitflip_budget`:    default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
#   `local_bitflip_budget`:     default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.
#
##########################################################################################

# source flags and colour definitions from conf file
source flags.conf 

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

# required args
OPERATION=$1    # TRAIN     TEST
shift
KERNEL_SIZE=$1  # 3         5       7
shift
NN_MODEL="$1"   # FMNIST    CIFAR   RESNET
shift
LOOPS=$1        #
shift
RT_SIZE=$1      # 64
shift
LAYER_CONFIG=$1 # INDIV     ALL     CUSTOM
shift
GPU_ID=$1       # 0         1       
shift

## params fixed for RTM
TEST_ERROR=1
TEST_RTM=1

TRAIN_MODEL=0
TEST_AUTO=0

if [[ "$OPERATION" == "TRAIN" ]];
then
    ## params for RTM training
    TRAIN_MODEL=1
    
    EPOCHS=$1       # 10
    shift
    BATCH_SIZE=$1   # 256
    shift
    LR=$1           # 0.001
    shift
    STEP_SIZE=$1    # 25
    shift

elif [[ "$OPERATION" == *"TEST_AUTO"* ]];
then
    TEST_AUTO=1

    GLOBAL_BITFLIP_BUDGET=0.0
    LOCAL_BITFLIP_BUDGET=0.0

    MODEL_PATH=$1
    shift

    MODEL_DIR=$(dirname "$MODEL_PATH")
    MODEL_NAME=$(basename "$MODEL_PATH" .pt)

elif [[ "$OPERATION" == "TEST" ]];
then
    # optional args
    GLOBAL_BITFLIP_BUDGET=${1:-0.0}
    shift
    LOCAL_BITFLIP_BUDGET=${1:-0.0}
    shift

fi

if [ "$TRAIN_MODEL" = 0 ]; then
    ## variables & arrays
    declare -a all_results  # array of arrays to store all results

    ## create output directory
    timestamp=$(date +%Y-%m-%d_%H-%M-%S)
    results_dir="RTM_results/$NN_MODEL/$RT_SIZE/$timestamp"
    output_dir="$results_dir/outputs"

    if [ "$CALC_RESULTS" == "True" ]; then
        all_results_dir="$results_dir/all_results"

        if [ "$TEST_AUTO" = 1 ]; then
            all_results_test_auto_dir="$MODEL_DIR/$OPERATION"
        fi
    fi
    if [ "$CALC_BITFLIPS" == "True" ]; then
        all_bitflips_dir="$results_dir/all_bitflips"
    fi
    if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
        all_misalign_faults_dir="$results_dir/all_misalign_faults"
    fi
    if [ "$CALC_AFFECTED_RTS" == "True" ]; then
        all_affected_rts_dir="$results_dir/all_affected_rts"
    fi


    if [ ! -d "$results_dir" ]; then
        mkdir -p "$results_dir"

        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        else
            echo "Directory $output_dir already exists."
        fi

        echo ""

        if [ "$CALC_RESULTS" == "True" ] && [ ! -d "$all_results_dir" ]; then
            mkdir -p "$all_results_dir"

            if [ "$TEST_AUTO" = 1 ]; then
                if [ ! -d "$all_results_test_auto_dir" ]; then
                    mkdir -p "$all_results_test_auto_dir"
                fi
            fi
        else
            echo -e "${RED}Flag CALC_RESULTS set to False.${RESET}"
        fi

        if [ "$CALC_BITFLIPS" == "True" ] && [ ! -d "$all_bitflips_dir" ]; then
            mkdir -p "$all_bitflips_dir"
        else
            echo -e "${YELLOW}Flag CALC_BITFLIPS set to False.${RESET}"
        fi

        if [ "$CALC_MISALIGN_FAULTS" == "True" ] && [ ! -d "$all_misalign_faults_dir" ]; then
            mkdir -p "$all_misalign_faults_dir"
        else
            echo -e "${YELLOW}Flag CALC_MISALIGN_FAULTS set to False.${RESET}"
        fi

        if [ "$CALC_AFFECTED_RTS" == "True" ] && [ ! -d "$all_affected_rts_dir" ]; then
            mkdir -p "$all_affected_rts_dir"
        else
            echo -e "${YELLOW}Flag CALC_AFFECTED_RTS set to False.${RESET}"
        fi


    else
        echo "Directory $results_dir already exists."
    fi

    echo ""
fi

# Iterate through the array and store indices of 0s
for i in "${!PROTECT_LAYERS[@]}"; do
  if [ "${PROTECT_LAYERS[i]}" -eq 0 ]; then
    UNPROT_LAYER_IDS+=("$((i + 1))")
  fi
done

result="${UNPROT_LAYER_IDS[*]}"
unprot_layers_string="${result[*]// /}"

## Check what the argument contains
if [[ $LAYER_CONFIG == *"ALL"* ]]; then
    echo -e "${CYAN}Number of unprotected layers is ${#UNPROT_LAYER_IDS[@]}/${#PROTECT_LAYERS[@]} ALL (IDs: ${UNPROT_LAYER_IDS[@]})${RESET}"
elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    echo -e "${CYAN}Number of unprotected layers is ${#UNPROT_LAYER_IDS[@]}/${#PROTECT_LAYERS[@]} CUSTOM (IDs: ${UNPROT_LAYER_IDS[@]})${RESET}"
elif [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
    echo -e "${CYAN}Number of unprotected layers is ${#UNPROT_LAYER_IDS[@]}/${#PROTECT_LAYERS[@]} INDIV (IDs: ${UNPROT_LAYER_IDS[@]})${RESET}"
else
    echo -e "${RED}Invalid layer configuration $LAYER_CONFIG.${RESET}"
    # Break or exit the script
    exit 1
fi

## Initialize parameters based on NN_MODEL
if [ "$NN_MODEL" = "MNIST" ]
then
    MODEL="MLP"
    DATASET="MNIST"
    TEST_BATCH_SIZE=10000 # adjust to execute TEST_BATCH_SIZE/batches images at once in each inference iteration

    if [ "$TRAIN_MODEL" = 0 ]; then
        if [ "$KERNEL_SIZE" = 0 ]
        then
            # MODEL_PATH="models/model_mnist9696_bnn.pt"
            # MODEL_PATH="models/model_mnist9418_bnn.pt"
            MODEL_PATH="models/model_mnist8562_bnn.pt"
        else
            echo "Invalid KERNEL_SIZE $KERNEL_SIZE for $NN_MODEL (no kernel size needed, use 0 for MNIST)."
            exit 1
        fi
    else
        MODEL_PATH="" # mnist_rtfi_nomhl_bh_L${unprot_layers_string}
    fi

elif [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    TEST_BATCH_SIZE=10000 # adjust to execute TEST_BATCH_SIZE/batches images at once in each inference iteration

    if [ "$TRAIN_MODEL" = 0 ]
    then
        if [ "$TEST_AUTO" = 0 ]
        then
            if [ "$KERNEL_SIZE" = 3 ]
            then
                # MODEL_PATH="models/model_fmnist9108.pt"
                # MODEL_PATH="models/phase1/model_fmnist_rtfi_nomhl_nobh_L4_p01.pt"
                MODEL_PATH="model_fmnist_binfi_nomhl_nobh_L1_p0.0001.pt"
            elif [ "$KERNEL_SIZE" = 5 ]
            then
                MODEL_PATH="models/model_fmnist5x5_9077.pt"
            elif [ "$KERNEL_SIZE" = 7 ]
            then
                MODEL_PATH="models/model_fmnist7x7_8880.pt"
            else
                echo "Invalid KERNEL_SIZE $KERNEL_SIZE for $NN_MODEL."
                exit 1
            fi
        else
            echo -e "${CYAN}Automatic testing using MODEL_PATH=$MODEL_PATH${RESET}"
        fi
    else
        MODEL_PATH="fmnist_binfi_nomhl_nobh_L${unprot_layers_string}"
    fi

elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    TEST_BATCH_SIZE=10000 # adjust to execute TEST_BATCH_SIZE/batches images at once in each inference iteration

    if [ "$TRAIN_MODEL" = 0 ]; then
        if [ "$KERNEL_SIZE" = 3 ]
        then
            MODEL_PATH="models/model_cifar8582.pt"
            # MODEL_PATH="model_cifar8660.pt"
        else
            echo "Invalid KERNEL_SIZE $KERNEL_SIZE for $NN_MODEL."
            exit 1
        fi
    else
        MODEL_PATH="cifar_rtfi_nomhl_nobh_L${unprot_layers_string}"
    fi

elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    TEST_BATCH_SIZE=4096 # adjust to execute TEST_BATCH_SIZE/batches images at once in each inference iteration

    if [ "$TRAIN_MODEL" = 0 ]; then
        if [ "$KERNEL_SIZE" = 3 ]
        then
            MODEL_PATH="models/model_resnet7694.pt"
        else
            echo "Invalid KERNEL_SIZE $KERNEL_SIZE for $NN_MODEL."
            exit 1
        fi
    else
        MODEL_PATH="resnet_rtfi_nomhl_nobh_L${unprot_layers_string}"
    fi
else
    echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: MNIST, FMNIST, CIFAR, RESNET${RESET}\n"
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

if [ "$TRAIN_MODEL" = 0 ]; then
    if [ "$CALC_RESULTS" == "True" ]; then
        # out_results_file_bb="$all_results_dir/all_results-$GLOBAL_BITFLIP_BUDGET-$LOCAL_BITFLIP_BUDGET.txt"
        out_results_file="$all_results_dir/all_results.txt"
        echo "" > "$out_results_file"
        > "$out_results_file"

        if [ "$TEST_AUTO" = 1 ]; then
            out_results_file_test_auto="$all_results_test_auto_dir/$MODEL_NAME.txt"
            echo "" > "$out_results_file_test_auto"
            > "$out_results_file_test_auto"
        fi
    fi
    if [ "$CALC_BITFLIPS" == "True" ]; then
        out_bitflips_file="$all_bitflips_dir/all_bitflips.txt"
        echo "" > "$out_bitflips_file"
        > "$out_bitflips_file"
    fi
    if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
        out_misalign_faults_file="$all_misalign_faults_dir/all_misalign_faults.txt"
        echo "" > "$out_misalign_faults_file"
        > "$out_misalign_faults_file"
    fi
    if [ "$CALC_AFFECTED_RTS" == "True" ]; then
        out_affected_rts_file="$all_affected_rts_dir/all_affected_rts.txt"
        echo "" > "$out_affected_rts_file"
        > "$out_affected_rts_file"
    fi
fi

declare -a all_results  # Declare an array of arrays to store all results
inference_time=0

## Main loop
start_time=$(date +%s.%N)
for i in "${!PERRORS[@]}"
do
    p=${PERRORS[$i]}

    if [ "$TRAIN_MODEL" = 0 ]; then
        echo -e "\n${GREEN}Run $((i + 1))/${#PERRORS[@]} with PERROR=$p on $NN_MODEL for $LOOPS inference iteration(s).${RESET}"
    else
        MODEL_PATH="${MODEL_PATH}_p${p}"
        echo -e "\n${GREEN}Training run $((i + 1))/${#PERRORS[@]} with PERROR=$p on $NN_MODEL for $EPOCHS epochs with lr=$LR, batch_size=$BATCH_SIZE, step_size=$STEP_SIZE.${RESET}"
    fi
    
    declare -a list     # stores results

    # in the case of INDIV, 
    if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
        for layer in "${!PROTECT_LAYERS[@]}"    
        do
            if [ "${PROTECT_LAYERS[$layer]}" == 0 ]; then
                let "L=layer+1"
                echo -e "${YELLOW}UNprotected Layer: only $L -> INDIV${RESET}\n"
                
                if [ "$TRAIN_MODEL" = 0 ]; then

                    # PROTECT_LAYERS[$layer]=0
                    # echo "${PROTECT_LAYERS[@]}"
                
                    output_dir_L="$output_dir/$L"
                    if [ ! -d "$output_dir_L" ]; then
                        mkdir -p "$output_dir_L"
                    else
                        echo "Directory $output_dir_L already exists."
                    fi
                    output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

                    python run.py --model=${MODEL} --dataset=${DATASET} --test-batch-size=${TEST_BATCH_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --kernel_size=${KERNEL_SIZE} --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET --calc_results=${CALC_RESULTS} --calc_bitflips=${CALC_BITFLIPS} --calc_misalign_faults=${CALC_MISALIGN_FAULTS} --calc_affected_rts=${CALC_AFFECTED_RTS} | tee "$output_file"

                    if [ "$CALC_RESULTS" == "True" ]; then
                        results_line=$(tail -n 2 "$output_file" | head -n 1)
                        # Remove square brackets and split values
                        values=$(echo "$results_line" | tr -d '[]')
                    fi
                    if [ "$CALC_BITFLIPS" == "True" ]; then
                        bitflips_line=$(tail -n 4 "$output_file" | head -n 1)
                        echo $bitflips_line >> "$out_bitflips_file"
                    fi
                    if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
                        misalign_faults_line=$(tail -n 6 "$output_file" | head -n 1)
                        echo $misalign_faults_line >> "$out_misalign_faults_file"
                    fi
                    if [ "$CALC_AFFECTED_RTS" == "True" ]; then
                        affected_rts_line=$(tail -n 8 "$output_file" | head -n 1)
                        echo $affected_rts_line >> "$out_affected_rts_file"
                    fi

                    times_line=$(tail -n 11 "$output_file" | head -n 1)
                    times=$(echo "$times_line" | tr -d '[]' | tr ',' ' ')

                    for time in $times; do
                        inference_time=$(echo "$inference_time + $time" | bc)
                    done
                    
                    list+=("$values")
                    # echo $list
                    
                    python scripts-python/plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}

                    # PROTECT_LAYERS[$layer]=1
                else
                    python3 run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --train-model=${TRAIN_MODEL} --save-model=${MODEL_PATH} --kernel_size=${KERNEL_SIZE} --rt_size=${RT_SIZE} --test_rtm=${TEST_RTM} --perror=${p} --gpu-num=$GPU_ID --protect_layers ${PROTECT_LAYERS[@]}
                fi
            fi
        done
        all_results+=("$list")

    else    # in the case of ALL, CUSTOM

        L=$LAYER_CONFIG
        echo -e "${YELLOW}UNprotected Layer(s): $L (${UNPROT_LAYER_IDS[@]})${RESET}\n"

            if [ "$TRAIN_MODEL" = 0 ]; then

                # PROTECT_LAYERS[$layer]=0
                # echo "${PROTECT_LAYERS[@]}"
                
                output_dir_L="$output_dir/$L"
                if [ ! -d "$output_dir_L" ]; then
                    mkdir -p "$output_dir_L"
                else
                    echo "Directory $output_dir_L already exists."
                fi
                output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

                python run.py --model=${MODEL} --dataset=${DATASET} --test-batch-size=${TEST_BATCH_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --kernel_size=${KERNEL_SIZE} --test_rtm=${TEST_RTM} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET  --calc_results=${CALC_RESULTS} --calc_bitflips=${CALC_BITFLIPS} --calc_misalign_faults=${CALC_MISALIGN_FAULTS} --calc_affected_rts=${CALC_AFFECTED_RTS} | tee "$output_file"
                

                if [ "$CALC_RESULTS" == "True" ]; then
                    results_line=$(tail -n 2 "$output_file" | head -n 1)
                    # Remove square brackets and split values
                    values=$(echo "$results_line" | tr -d '[]')
                fi
                if [ "$CALC_BITFLIPS" == "True" ]; then
                    bitflips_line=$(tail -n 4 "$output_file" | head -n 1)
                    echo $bitflips_line >> "$out_bitflips_file"
                fi
                if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
                    misalign_faults_line=$(tail -n 6 "$output_file" | head -n 1)
                    echo $misalign_faults_line >> "$out_misalign_faults_file"
                fi
                if [ "$CALC_AFFECTED_RTS" == "True" ]; then
                    affected_rts_line=$(tail -n 8 "$output_file" | head -n 1)
                    echo $affected_rts_line >> "$out_affected_rts_file"
                fi

                times_line=$(tail -n 11 "$output_file" | head -n 1)
                times=$(echo "$times_line" | tr -d '[]' | tr ',' ' ')

                for time in $times; do
                    inference_time=$(echo "$inference_time + $time" | bc)
                done

                list+=("$values")
                # echo $list
                
                python scripts-python/plot.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}
                
                all_results+=("$list")
            else
                python3 run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --train-model=${TRAIN_MODEL} --save-model=${MODEL_PATH} --kernel_size=${KERNEL_SIZE} --rt_size=${RT_SIZE} --test_rtm=${TEST_RTM} --perror=${p} --protect_layers ${PROTECT_LAYERS[@]} --gpu-num=$GPU_ID
            fi
    fi

    # csv_file="$output_dir/table_$p.csv"
    # for value in "${list[@]}"
    # do
    #     echo "${value[@]}" >> "$csv_file"
    # done

    unset list
    
done
end_time=$(date +%s.%N)

if [ "$TRAIN_MODEL" = 0 ]; then

    if [ "$CALC_RESULTS" == "True" ]; then
        # Loop through the outer array
        for outer_idx in "${!all_results[@]}"; do
            # Print each inner array element with a space separator
            echo "${all_results[$outer_idx]}" >> "$out_results_file"
            if [ "$TEST_AUTO" = 1 ]; then
                echo "${all_results[$outer_idx]}" >> "$out_results_file_test_auto"
            fi
        done

        echo ""
        echo "============================="
        echo -e "${CYAN}Average accuracies${RESET}"
        echo "(for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg.py ${out_results_file}
        echo "============================="
    fi
    if [ "$CALC_BITFLIPS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average bitflips${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_bitflips_file} 1
    fi
    if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average misalign_faults${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_misalign_faults_file} 1
    fi
    if [ "$CALC_AFFECTED_RTS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average affected_rts${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_affected_rts_file} 1
    fi
fi


if [ "$TRAIN_MODEL" = 0 ]; then
    hours=$(printf "%02d" $(echo "$inference_time/3600" | bc))
    minutes=$(printf "%02d" $(echo "($inference_time%3600)/60" | bc))
    seconds=$(printf "%06.3f" $(echo "$inference_time%60" | bc))

    echo -e "\n${CYAN}Total model inference times: ${hours}:${minutes}:${seconds}${RESET}"
else
    training_time=$(echo "$end_time - $start_time" | bc)
    hours=$(printf "%02d" $(echo "$training_time/3600" | bc))
    minutes=$(printf "%02d" $(echo "($training_time%3600)/60" | bc))
    seconds=$(printf "%06.3f" $(echo "$training_time%60" | bc))

    echo -e "\n${CYAN}Total model training times: ${hours}:${minutes}:${seconds}${RESET}"
fi


## Reset layer configuration in case of INDIV
if [[ $LAYER_CONFIG =~ ^[0-9]+$ ]]; then
    # echo -e "${PROTECT_LAYERS[@]}"
    PROTECT_LAYERS[$LAYER_CONFIG]=1
    # echo -e "${PROTECT_LAYERS[@]}"
fi

