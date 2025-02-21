#!/bin/bash

##########################################################################################
#   LDB
#   
#   Helper script for automating array generation and ease of use of main script for running RTM misalignment fault simulation for BNNs
#
#   Arguments:
#     --operation, -o         TRAIN or TEST or TEST_AUTO
#     --loops, -l             Number of experiment loops (0-100] (if --operation=TRAIN -> use --epochs instead)
#     --model, -m             Neural network model (MNIST|FMNIST|CIFAR|RESNET)
#     --perrors, -p           Array of misalignment fault rates (space-separated)
#     --kernel-size, -ks      Kernel size for conv layers (0 if none)
#     --kernel-mapping, -km   Mapping configuration of weights in kernels (ROW|COL|CLW|ACW)
#     --rt-size, -rs          Racetrack size (typically 64)
#     --rt-mapping, -rm       Mapping configuration of data onto racetracks (ROW|COL|MIX)
#     --layer-config, -lc     Layer configuration of unprotected layers (ALL|CUSTOM|INDIV)
#     --layers, -ls           Layer IDs array to be left unprotected (space-separated, if --layer-config=CUSTOM)
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

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --operation, -o         TRAIN or TEST or TEST_AUTO"
    echo "  --loops, -l             Number of experiment loops (0-100] (if --operation=TRAIN -> use --epochs instead)"
    echo "  --model, -m             Neural network model (MNIST|FMNIST|CIFAR|RESNET)"
    echo "  --perrors, -p           Array of misalignment fault rates (space-separated)"
    echo "  --kernel-size, -ks      Kernel size for conv layers (0 if none)"
    echo "  --kernel-mapping, -km   Mapping configuration of weights in kernels (ROW|COL|CLW|ACW)"
    echo "  --rt-size, -rs          Racetrack size (typically 64)"
    echo "  --rt-mapping, -rm       Mapping configuration of data onto racetracks (ROW|COL|MIX)"
    echo "  --layer-config, -lc     Layer configuration of unprotected layers (ALL|CUSTOM|INDIV)"
    echo "  --layers, -ls           Layer IDs array to be left unprotected (space-separated, if --layer-config=CUSTOM)"
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

start_time=$(date +%s.%N)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --perrors|-p)
            PERRORS=()
            shift
            # Read all arguments until the next flag
            while [[ $# -gt 0 && ! $1 =~ ^(-{1,2}) ]]; do
                PERRORS+=("$1")
                shift
            done
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
        --layers|-ls)
            LAYERS=()
            shift
            # Read all arguments until the next flag
            while [[ $# -gt 0 && ! $1 =~ ^(-{1,2}) ]]; do
                LAYERS+=("$1")
                shift
            done
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
if [[ "$OPERATION" == "TRAIN" || "$OPERATION" == "TEST" || "$OPERATION" == *"TEST_AUTO"* ]]; 
then
    echo -e "\n${CYAN}Executing $OPERATION Operation${RESET}\n"
    if [ "$OPERATION" == "TRAIN" ]; then
        TRAIN_MODEL=1
    elif [ "$OPERATION" == "TEST" ]; then
        TEST_MODEL=1
    elif [ "$OPERATION" == *"TEST_AUTO"* ]; then
        TEST_AUTO=1
    fi
else
    echo -e "\n${RED}$OPERATION not supported, check spelling, capitalization & available operations: TRAIN or TEST or TEST_AUTO${RESET}\n"
    exit 1
fi


## Specify the number of total layers in NN model
if [ "$NN_MODEL" = "MNIST" ]
then
    layers_total=3
elif [ "$NN_MODEL" = "FMNIST" ]
then
    layers_total=4
elif [ "$NN_MODEL" = "CIFAR" ]
then
    layers_total=8
elif [ "$NN_MODEL" = "RESNET" ]
then 
    layers_total=21
else
    echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: MNIST, FMNIST, CIFAR, RESNET${RESET}\n"
    exit 1
fi


## Specify CUSTOM and DEFAULT CUSTOM layer config
if [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then

    ## Edit CUSTOM layer configuration based on the specified layers
    if [ ${#LAYERS[@]} -gt 0 ]; then
        ## Initialize with 1s for every layer (all protected)
        PROTECT_LAYERS=($(for i in $(seq 1 $layers_total); do echo 1; done))

        ## exit if length of array $LAYERS > layers_total
        if [ ${#LAYERS[@]} -le $layers_total ]; then
            ## Set specified layers to be unprotected (the first layer is numbered with 1)
            for l in "${LAYERS[@]}"
            do
                ## exit if array $LAYERS specifies layer numbered outside of bounds 0 < l <= layers_total
                if [ $l -gt 0 ] && [ $l -le $layers_total ]; then
                    PROTECT_LAYERS[$l-1]=0
                else
                    echo -e "\n${RED}Specified layer $l is outside of the bounds for $NN_MODEL = (0, $layers_total]. Note that layer numbering starts at 1 instead of 0.${RESET}\n"
                    exit 1
                fi
            done
        else
            echo -e "\n${RED}Specified number of layers is larger than the total number of layers in $NN_MODEL = $layers_total${RESET}\n"
            exit 1
        fi
    else

        ## No specified layers => use DEFAULT CUSTOM configs
        ## Only 1 at a time
        if [ "$NN_MODEL" = "MNIST" ]
        then
        
            declare -a PROTECT_LAYERS=(0 0 1)

        elif [ "$NN_MODEL" = "FMNIST" ]
        then
        
            declare -a PROTECT_LAYERS=(1 1 1 1)
            # declare -a PROTECT_LAYERS=(0 0 1 0)
            # declare -a PROTECT_LAYERS=(0 0 1 1)

        elif [ "$NN_MODEL" = "CIFAR" ]
        then

            declare -a PROTECT_LAYERS=(0 1 0 1 1 1 1 0)
            # declare -a PROTECT_LAYERS=(1 0 1 0 1 0 1 0)
            # declare -a PROTECT_LAYERS=(0 1 0 1 0 1 0 1)
            # declare -a PROTECT_LAYERS=(0 1 1 1 1 0 0 0)

        elif [ "$NN_MODEL" = "RESNET" ]
        then 

            declare -a PROTECT_LAYERS=(0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1)
            # declare -a PROTECT_LAYERS=(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0)
            # declare -a PROTECT_LAYERS=(1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
            # declare -a PROTECT_LAYERS=(0 1 0 1 1 1 1 1 0 0 1 0 0 0 0 1 1 0 0 0 1)

        else
            echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: MNIST, FMNIST, CIFAR, RESNET${RESET}\n"
            exit 1
        fi
    fi  
fi

## Call main script
if [[ $LAYER_CONFIG == *"ALL"* ]]; then
    # echo "Number of unprotected layers: ALL"
    PROTECT_LAYERS=($(for i in $(seq 1 $layers_total); do echo 0; done))

    if [ "$TRAIN_MODEL" = 1 ]; then
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --epochs "$EPOCHS" \
                       --batch-size "$BATCH_SIZE" \
                       --learning-rate "$LR" \
                       --step-size "$STEP_SIZE" \
                       --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                       --local-budget "$LOCAL_BITFLIP_BUDGET"
        
    elif [ "$TEST_AUTO" = 1 ]; then
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --model-path "$MODEL_PATH"
    else
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                       --local-budget "$LOCAL_BITFLIP_BUDGET"
        
        total=$((${#PERRORS[@]}*$LOOPS))
        echo -e "\n${PURPLE}Total individual experiments: ${#PERRORS[@]}x${LOOPS} = ${total}${RESET}"
    fi

elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    # echo "Number of unprotected layers: CUSTOM"

    if [ "$TRAIN_MODEL" = 1 ]; then
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --epochs "$EPOCHS" \
                       --batch-size "$BATCH_SIZE" \
                       --learning-rate "$LR" \
                       --step-size "$STEP_SIZE" \
                       --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                       --local-budget "$LOCAL_BITFLIP_BUDGET"
    
    elif [ "$TEST_AUTO" = 1 ]; then
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --model-path "$MODEL_PATH"
    else
        bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                       --perrors "${PERRORS[@]}" \
                       --operation "$OPERATION" \
                       --loops "$LOOPS" \
                       --model "$NN_MODEL" \
                       --kernel-size "$KERNEL_SIZE" \
                       --kernel-mapping "$KERNEL_MAPPING" \
                       --rt-size "$RT_SIZE" \
                       --rt-mapping "$GLOBAL_RT_MAPPING" \
                       --layer-config "$LAYER_CONFIG" \
                       --gpu "$GPU_ID" \
                       --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                       --local-budget "$LOCAL_BITFLIP_BUDGET"
        
        total=$((${#PERRORS[@]}*$LOOPS))
        echo -e "\n${PURPLE}Total individual experiments: ${#PERRORS[@]}x${LOOPS} = ${total}${RESET}"        
    fi 

elif [[ $LAYER_CONFIG == *"INDIV"* ]]; then
    # echo "Number of unprotected layers: INDIVIDUAL"
    ## Initialize with 1s for every layer (all protected)
    PROTECT_LAYERS=($(for i in $(seq 1 $layers_total); do echo 1; done))
    
    ## Loop through all layers individually
    for ((layer_id=0; layer_id<layers_total; layer_id++))
    do
        ## Set unprotected layer
        PROTECT_LAYERS[$layer_id]=0

        ## Run the bash file
        if [ "$TRAIN_MODEL" = 1 ]; then
            bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                           --perrors "${PERRORS[@]}" \
                           --operation "$OPERATION" \
                           --loops "$LOOPS" \
                           --model "$NN_MODEL" \
                           --kernel-size "$KERNEL_SIZE" \
                           --kernel-mapping "$KERNEL_MAPPING" \
                           --rt-size "$RT_SIZE" \
                           --rt-mapping "$GLOBAL_RT_MAPPING" \
                           --layer-config "$layer_id" \
                           --gpu "$GPU_ID" \
                           --epochs "$EPOCHS" \
                           --batch-size "$BATCH_SIZE" \
                           --learning-rate "$LR" \
                           --step-size "$STEP_SIZE" \
                           --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                           --local-budget "$LOCAL_BITFLIP_BUDGET"
        elif [ "$TEST_AUTO" = 1 ]; then
            bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                           --perrors "${PERRORS[@]}" \
                           --operation "$OPERATION" \
                           --loops "$LOOPS" \
                           --model "$NN_MODEL" \
                           --kernel-size "$KERNEL_SIZE" \
                           --kernel-mapping "$KERNEL_MAPPING" \
                           --rt-size "$RT_SIZE" \
                           --rt-mapping "$GLOBAL_RT_MAPPING" \
                           --layer-config "$layer_id" \
                           --gpu "$GPU_ID" \
                           --model-path "$MODEL_PATH"
        else
            bash run_all.sh --protect-layers "${PROTECT_LAYERS[@]}" \
                           --perrors "${PERRORS[@]}" \
                           --operation "$OPERATION" \
                           --loops "$LOOPS" \
                           --model "$NN_MODEL" \
                           --kernel-size "$KERNEL_SIZE" \
                           --kernel-mapping "$KERNEL_MAPPING" \
                           --rt-size "$RT_SIZE" \
                           --rt-mapping "$GLOBAL_RT_MAPPING" \
                           --layer-config "$layer_id" \
                           --gpu "$GPU_ID" \
                           --global-budget "$GLOBAL_BITFLIP_BUDGET" \
                           --local-budget "$LOCAL_BITFLIP_BUDGET"
        fi 

        ## Reset for next iteration
        PROTECT_LAYERS[$layer_id]=1

    done

    if [ "$TRAIN_MODEL" = 0 ]; then
        echo -e "${YELLOW}Warning! Displayed total model inference times in this case (INDIV) are ${RED}FOR EACH${YELLOW} tested layer at a time (scroll up to see previous inference times) ${BLUE}TODO -> sum up into grand total${RESET}"

        total=$((${#PERRORS[@]}*${LOOPS}*${layers_total}))
        echo -e "\n${PURPLE}Total individual experiments: ${#PERRORS[@]}x${LOOPS}x${layers_total} = ${total}${RESET}"
    fi

else
    echo "Invalid layer configuration $LAYER_CONFIG."
    exit 1
fi
end_time=$(date +%s.%N)

elapsed_time=$(echo "$end_time - $start_time" | bc)
hours=$(printf "%02d" $(echo "$elapsed_time/3600" | bc))
minutes=$(printf "%02d" $(echo "($elapsed_time%3600)/60" | bc))
seconds=$(printf "%06.3f" $(echo "$elapsed_time%60" | bc))

echo -e "${PURPLE}Total script execution time: ${hours}:${minutes}:${seconds}${RESET}"
