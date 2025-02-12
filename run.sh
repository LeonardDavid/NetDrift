#!/bin/bash

##########################################################################################
#   LDB
#   
#   Helper script for automating array generation and ease of use of main script for running RTM misalignment fault simulation for BNNs
# 
#   ### Arguments:
#   `OPERATION`:                TRAIN or TEST 
#   {arr_perrors}:              Array of misalignment fault rates to be tested/trained (floats)
#   `PERRORS`:                  REQUIRED: array termination token
#   `kernel_size`:              size of kernel used for calculations in convolutional layers (if none then use 0)
#   `nn_model`:                 FMNIST, CIFAR, RESNET
#   `loops`:                    amount of loops the experiment should run (0, 100] (not used if OPERATION=TEST)
#   `rt_size`:                  racetrack/nanowire size (typically 64)
#   `global_rt_mapping`:        mapping configuration of data onto racetracks: ROW, COL or MIX
#   {arr_layer_ids}:            [OPTIONAL] specify optional layer_ids in an array (starting at 1 upto total_layers) ONLY BEFORE `layer_config` WITH TERMINATION `CUSTOM`!
#   `layer_config`:             unprotected layers configuration (ALL, CUSTOM, INDIV). Note that if no optional {arr_layer_ids} is specified, CUSTOM will execute DEFAULT CUSTOM defined manually in run.sh
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

start_time=$(date +%s.%N)

# source flags and colour definitions from conf file
source flags.conf 

## Fixed ending tokens for array arguments when calling main script
END1="END1"
END2="END2"

## Specify the operation: TRAIN or TEST
OPERATION=$1

TRAIN_MODEL=0
TEST_AUTO=0

if [[ "$OPERATION" == "TRAIN" || "$OPERATION" == "TEST" || "$OPERATION" == *"TEST_AUTO"* ]]; 
then
    echo -e "\n${CYAN}Executing $OPERATION Operation${RESET}\n"
    if [ "$OPERATION" == "TRAIN" ]; then
        TRAIN_MODEL=1
    elif [ "$OPERATION" == *"TEST_AUTO"* ]; then
        TEST_AUTO=1
    fi
else
    echo -e "\n${RED}$OPERATION not supported, check spelling, capitalization & available operations: TRAIN or TEST or TEST_AUTO${RESET}\n"
    exit 1
fi

shift
PERRORS_TOKEN=$1

## Specify PERRORS array of misalignment faults to be tested. Values can be distinct or equal or both
## ARRAY HAS TO END WITH "PERRORS" ENDING TOKEN! 
if [[ ! $PERRORS_TOKEN = ^[0-9]+$ ]]; then 
    ## TODO SOLVE INFINITE LOOP!!!
    PERRORS=()
    while [[ $1 != "PERRORS" ]]; do
        PERRORS+=("$1")
        shift
    done
    PERRORS_TOKEN=$1
else
    echo -e "\n${RED}Misalignment fault values not defined before PERRORS ending token OR missing PERRORS ending token before NN_MODEL argument!${RESET}\n"
    exit 1
fi

## Required args
shift # skip token necessary
KERNEL_SIZE=$1
shift
NN_MODEL="$1"
shift
LOOPS=$1
shift
RT_SIZE=$1
shift
GLOBAL_RT_MAPPING=$1 #      ROW     COL     MIX

if [[ ! $GLOBAL_RT_MAPPING =~ ^(ROW|COL|MIX)$ ]]; then
    echo -e "\n${RED}Global RT mapping ($GLOBAL_RT_MAPPING) must be ROW, COL, or MIX${RESET}\n"
    exit 1
fi

shift
LAYER_CONFIG=$1

## LAYER_CONFIG can be specified as: ALL, INDIV, CUSTOM, {LAYERS} CUSTOM
## if {LAYERS} is specified, the specified layers will be left unprotected (e.g. 1 3 CUSTOM protects all layers except the first and third layer)
if [[ ! $LAYER_CONFIG =~ ^(ALL|CUSTOM|INDIV)$ ]]; then 
    # Read the LAYERS array (assumes the array ends with "CUSTOM"!) 
    # -> if no LAYERS are specified, use default CUSTOM configs
    LAYERS=()
    while [[ $1 != "CUSTOM" ]]; do
        LAYERS+=("$1")
        shift
    done
    LAYER_CONFIG=$1
fi

## Remaining required args
shift # skip token necessary
GPU_ID=$1
shift


if [ "$TRAIN_MODEL" = 1 ]; then
    ## params for RTM training

    EPOCHS=$1       # 10
    shift
    BATCH_SIZE=$1   # 256
    shift
    LR=$1           # 0.001
    shift
    STEP_SIZE=$1    # 25
    shift

elif [ "$TEST_AUTO" = 1 ]; then

    GLOBAL_BITFLIP_BUDGET=0.0
    LOCAL_BITFLIP_BUDGET=0.0

    MODEL_PATH=$1
    shift

else
    # optional args
    GLOBAL_BITFLIP_BUDGET=${1:-0.0}
    shift
    LOCAL_BITFLIP_BUDGET=${1:-0.0}
    shift
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
                    echo -e "\n${RED}Specified layer $l is outside of the bounds for $NN_MODEL = (0, $layers_total]. Note that layer numberring starts at 1 instead of 0.${RESET}\n"
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


## Call main script
if [[ $LAYER_CONFIG == *"ALL"* ]]; then
    # echo "Number of unprotected layers: ALL"
    PROTECT_LAYERS=($(for i in $(seq 1 $layers_total); do echo 0; done))

    if [ "$TRAIN_MODEL" = 1 ]; then
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $EPOCHS $BATCH_SIZE $LR $STEP_SIZE $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
        
    elif  [ "$TEST_AUTO" = 1 ]; then
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $MODEL_PATH

    else
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
        
        total=$((${#PERRORS[@]}*$LOOPS))
        echo -e "\n${PURPLE}Total individual experiments: ${#PERRORS[@]}x${LOOPS} = ${total}${RESET}"
    fi 

elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    # echo "Number of unprotected layers: CUSTOM"

    if [ "$TRAIN_MODEL" = 1 ]; then
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $EPOCHS $BATCH_SIZE $LR $STEP_SIZE $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
    
    elif  [ "$TEST_AUTO" = 1 ]; then
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $MODEL_PATH
    else
        bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
        
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
        #
        ##  required params:
        ###  $1: array of PROTECT_LAYERS (integers)
        ###  END1: first array termination token
        ###  $3: array of PERRORS (floats)
        ###  END2: second array termination token
        ###
        ###  $5: NN model           FMNIST, CIFAR, RESNET
        ###  $6: loops              amount of loops (0, 100]
        ###  $7: rt_size            racetrack/nanowire size (typically 64)
        ###  $8: layer_id           unprotected layer
        ###  $9: GPU_id             ID of GPU to use for computations (0, 1)
        #
        ##  optional params:
        ###  $10: global_bitflip_budget: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
        ###  $11: local_bitflip_budget: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.
     
        if [ "$TRAIN_MODEL" = 1 ]; then
            bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $layer_id $GPU_ID $EPOCHS $BATCH_SIZE $LR $STEP_SIZE $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
        elif [ "$TEST_AUTO" = 1 ]; then
            bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $layer_id $GPU_ID $MODEL_PATH
        else
            bash run_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $OPERATION $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $GLOBAL_RT_MAPPING $layer_id $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
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
