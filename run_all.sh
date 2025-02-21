#!/bin/bash

##########################################################################################
#   LDB
#   
#   Automation script for running RTM misalignment fault simulation for BNNs
# 
#   Arguments:
#     --protect-layers, -pl   Array of layer protection flags (space-separated, 1:protected | 0:unprotected)
#     --perrors, -p           Array of misalignment fault rates (space-separated)
#     --operation, -o         TRAIN or TEST or TEST_AUTO
#     --loops, -l             Number of experiment loops (0-100] (if --operation=TRAIN -> use --epochs instead)
#     --model, -m             Neural network model (MNIST|FMNIST|CIFAR|RESNET)
#     --kernel-size, -ks      Kernel size for conv layers (0 if none)
#     --kernel-mapping, -km   Mapping configuration of weights in kernels (ROW|COL|CLW|ACW)
#     --rt-size, -rs          Racetrack size (typically 64)
#     --rt-mapping, -rm       Mapping configuration of data onto racetracks (ROW|COL|MIX)
#     --layer-config, -lc     Layer configuration of unprotected layers (ALL|CUSTOM|INDIV)
#     --gpu, -g               GPU ID to run computations on (0|1)
#   
#   Training specific options:
#     --epochs, -e            Number of training epochs
#     --batch-size, -bs       Batch size
#     --learning-rate, -lr    Learning rate
#     --step-size, -ss        Step size
#   
#   Optional:
#     --model-path, -mp       Path to model file to be used for TEST and TEST_AUTO
#     --global-budget, -gb    Global bitflip budget (0.0-1.0]
#     --local-budget, -lb     Local bitflip budget (0.0-1.0]
#     --help, -h              Show this help message
#
##########################################################################################

# source flags and colour definitions from conf file
source flags.conf 

# Default values
TRAIN_MODEL=0
TEST_MODEL=0
TEST_AUTO=0
GLOBAL_BITFLIP_BUDGET=0.0
LOCAL_BITFLIP_BUDGET=0.0

## params fixed for RTM
TEST_ERROR=1
TEST_RTM=1

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --protect-layers, -pl   Array of layer protection flags (space-separated, 1:protected | 0:unprotected)"
    echo "  --perrors, -p           Array of misalignment fault rates (space-separated)"
    echo "  --operation, -o         TRAIN or TEST or TEST_AUTO"
    echo "  --loops, -l             Number of experiment loops (0-100] (if --operation=TRAIN -> use --epochs instead)"
    echo "  --model, -m             Neural network model (MNIST|FMNIST|CIFAR|RESNET)"
    echo "  --kernel-size, -ks      Kernel size for conv layers (0 if none)"
    echo "  --kernel-mapping, -km   Mapping configuration of weights in kernels (ROW|COL|CLW|ACW)"
    echo "  --rt-size, -rs          Racetrack size (typically 64)"
    echo "  --rt-mapping, -rm       Mapping configuration of data onto racetracks (ROW|COL|MIX)"
    echo "  --layer-config, -lc     Layer configuration of unprotected layers (ALL|CUSTOM|INDIV)"
    echo "  --gpu, -g               GPU ID to run computations on (0|1)"
    echo ""
    echo "Training specific options:"
    echo "  --epochs, -e            Number of training epochs"
    echo "  --batch-size, -bs       Batch size"
    echo "  --learning-rate, -lr    Learning rate"
    echo "  --step-size, -ss        Step size"
    echo ""
    echo "Optional:"
    echo "  --model-path, -mp       Path to model file to be used for TEST and TEST_AUTO"
    echo "  --global-budget, -gb    Global bitflip budget (0.0-1.0]"
    echo "  --local-budget, -lb     Local bitflip budget (0.0-1.0]"
    echo "  --help, -h              Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --protect-layers|-pl)
            PROTECT_LAYERS=()
            shift
            # Read all arguments until the next flag
            while [[ $# -gt 0 && ! $1 =~ ^(-{1,2}) ]]; do
                PROTECT_LAYERS+=("$1")
                shift
            done
            ;;
        --perrors|-p)
            PERRORS=()
            shift
            # Read all arguments until the next flag
            while [[ $# -gt 0 && ! $1 =~ ^(-{1,2}) ]]; do
                PERRORS+=("$1")
                shift
            done
            ;;
        --operation|-o)
            OPERATION="$2"
            shift 2
            ;;
        --loops|-l)
            LOOPS="$2"
            shift 2
            ;;
        --model|-m)
            NN_MODEL="$2"
            shift 2
            ;;
        --kernel-size|-ks)
            KERNEL_SIZE="$2"
            shift 2
            ;;
        --kernel-mapping|-km)
            KERNEL_MAPPING="$2"
            shift 2
            ;;
        --rt-size|-rs)
            RT_SIZE="$2"
            shift 2
            ;;
        --rt-mapping|-rm)
            GLOBAL_RT_MAPPING="$2"
            shift 2
            ;;
        --layer-config|-lc)
            LAYER_CONFIG="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU_ID="$2"
            shift 2
            ;;
        --model-path|-mp)
            MODEL_PATH="$2"
            shift 2
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size|-bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate|-lr)
            LR="$2"
            shift 2
            ;;
        --step-size|-ss)
            STEP_SIZE="$2"
            shift 2
            ;;
        --global-budget|-gb)
            GLOBAL_BITFLIP_BUDGET="$2"
            shift 2
            ;;
        --local-budget|-lb)
            LOCAL_BITFLIP_BUDGET="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            exit 1
            ;;
    esac
done

## Check if all required arguments are provided correctly
if [[ ! $KERNEL_MAPPING =~ ^(ROW|COL|CLW|ACW)$ ]]; then
    echo -e "\n${RED}Kernel mapping ($KERNEL_MAPPING) must be ROW, COL, CLW, or ACW${RESET}\n"
    exit 1
fi

if [[ ! $GLOBAL_RT_MAPPING =~ ^(ROW|COL|MIX)$ ]]; then
    echo -e "\n${RED}Global RT mapping ($GLOBAL_RT_MAPPING) must be ROW, COL, or MIX${RESET}\n"
    exit 1
fi


if [[ "$OPERATION" == "TRAIN" ]];
then
    TRAIN_MODEL=1

elif [[ "$OPERATION" == "TEST" ]];
then
    TEST_MODEL=1

elif [[ "$OPERATION" == *"TEST_AUTO"* ]];
then
    TEST_AUTO=1

    MODEL_DIR=$(dirname "$MODEL_PATH")
    MODEL_NAME=$(basename "$MODEL_PATH" .pt)
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
            MODEL_PATH="models/model_mnist9696_bnn.pt"
        else
            echo "Invalid KERNEL_SIZE $KERNEL_SIZE for $NN_MODEL (no kernel size needed, use 0 for MNIST)."
            exit 1
        fi
    else
        MODEL_PATH="mnist_rtfi_nomhl_nobh_L${unprot_layers_string}"
    fi

elif [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    TEST_BATCH_SIZE=10000 # adjust to execute TEST_BATCH_SIZE/batches images at once in each inference iteration
    # !TODO different TEST_BATCH_SIZEs + in combination with 10000/TEST_BATCH_SIZE = N index_offset iterations that do not execute inference (consistency -> test before and after)
    # TEST_BATCH_SIZE=200 # 500/250...

    if [ "$TRAIN_MODEL" = 0 ]
    then
        if [ "$TEST_AUTO" = 0 ]
        then
            if [ "$KERNEL_SIZE" = 3 ]
            then
                MODEL_PATH="models/model_fmnist9108.pt"
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
        if [ "$EXEC_EVEN2ODD_DEC" = "True" ]; then
            MODEL_PATH="fmnist_rtfi_edgeodddec${STEP_SIZE}_n${EXEC_EVERY_NRUN}_L${unprot_layers_string}"
        elif [ "$EXEC_EVEN2ODD_INC" = "True" ]; then
            MODEL_PATH="fmnist_rtfi_edgeoddinc${STEP_SIZE}_n${EXEC_EVERY_NRUN}_L${unprot_layers_string}"
        else
            MODEL_PATH="fmnist_rtfi_nomhl_nobh_L${unprot_layers_string}"
        fi
        echo -e "${CYAN}MODEL_PATH=$MODEL_PATH${RESET}"
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

                    python run.py --model=${MODEL} --dataset=${DATASET} --test-batch-size=${TEST_BATCH_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --kernel_size=${KERNEL_SIZE} --kernel_mapping=${KERNEL_MAPPING} --test_rtm=${TEST_RTM} --global_rt_mapping=${GLOBAL_RT_MAPPING} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET --calc_results=${CALC_RESULTS} --calc_bitflips=${CALC_BITFLIPS} --calc_misalign_faults=${CALC_MISALIGN_FAULTS} --calc_affected_rts=${CALC_AFFECTED_RTS} | tee "$output_file"

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
                    python3 run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --train-model=${TRAIN_MODEL} --save-model=${MODEL_PATH} --kernel_size=${KERNEL_SIZE} --kernel_mapping=${KERNEL_MAPPING} --rt_size=${RT_SIZE} --test_rtm=${TEST_RTM} --global_rt_mapping=${GLOBAL_RT_MAPPING} --perror=${p} --gpu-num=$GPU_ID --protect_layers ${PROTECT_LAYERS[@]}
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

                python run.py --model=${MODEL} --dataset=${DATASET} --test-batch-size=${TEST_BATCH_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --kernel_size=${KERNEL_SIZE} --kernel_mapping=${KERNEL_MAPPING} --test_rtm=${TEST_RTM} --global_rt_mapping=${GLOBAL_RT_MAPPING} --gpu-num=$GPU_ID --rt_size=$RT_SIZE --protect_layers ${PROTECT_LAYERS[@]} --global_bitflip_budget=$GLOBAL_BITFLIP_BUDGET --local_bitflip_budget=$LOCAL_BITFLIP_BUDGET  --calc_results=${CALC_RESULTS} --calc_bitflips=${CALC_BITFLIPS} --calc_misalign_faults=${CALC_MISALIGN_FAULTS} --calc_affected_rts=${CALC_AFFECTED_RTS} | tee "$output_file"
                

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
                python3 run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --train-model=${TRAIN_MODEL} --save-model=${MODEL_PATH} --kernel_size=${KERNEL_SIZE} --kernel_mapping=${KERNEL_MAPPING} --rt_size=${RT_SIZE} --test_rtm=${TEST_RTM} --global_rt_mapping=${GLOBAL_RT_MAPPING} --perror=${p} --protect_layers ${PROTECT_LAYERS[@]} --gpu-num=$GPU_ID
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

    if [ "$CALC_AFFECTED_RTS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average affected_rts${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_affected_rts_file} 1
    fi
    if [ "$CALC_MISALIGN_FAULTS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average misalign_faults${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_misalign_faults_file} 1
    fi
    if [ "$CALC_BITFLIPS" == "True" ]; then
        echo ""
        echo -e "${CYAN}Average bitflips${RESET}"
        echo "(per layer for each inference iteration across PERRORS misalignment fault rates):"
        echo ""
        python scripts-python/calculate_avg_matrix.py ${out_bitflips_file} 1
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

