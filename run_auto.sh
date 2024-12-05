#!/bin/bash

##########################################################################################
#   LDB
#   
#   Helper script for automating array generation and eas of use of main script for running RTM misalignment fault simulation for BNNs
# 
#   ### Arguments:
#   {arr_perrors}:                  Array of misalignment fault rates to be tested (floats)
#   `PERRORS`:                  REQUIRED: array termination token
#   `nn_model`:                 FMNIST, CIFAR, RESNET
#   `loops`:                    amount of loops (0, 100]
#   `rt_size`:               racetrack/nanowire size (typically 64)
#   `layer_config`:             unprotected layers configuration (ALL, CUSTOM, INDIV). Note that if no optional {arr_layer_ids} is specified, CUSTOM will execute DEFAULT CUSTOM defined manually in run_auto.sh
#   `gpu_id`:                   ID of GPU to use for computations (0, 1) 
#
#   ### Optional arguments:
#   {arr_layer_ids}:            specify optional layer_ids in an array (starting at 1 upto total_layers) ONLY BEFORE `layer_config` WITH TERMINATION `CUSTOM`!
#   `global_bitflip_budget`:    default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
#   `local_bitflip_budget`:     default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.
#
#
##########################################################################################

# source flags and colour definitions from conf file
source flags.conf 

## Fixed ending tokens for array arguments when calling main script
END1="END1"
END2="END2"

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

## Optional args
GLOBAL_BITFLIP_BUDGET=${1:-0.0}
shift
LOCAL_BITFLIP_BUDGET=${1:-0.0}
shift


# echo "PERRORS: ${PERRORS[@]}"
# echo "LAYERS: ${LAYERS[@]}"

# echo "$KERNEL_SIZE"
# echo "$NN_MODEL"
# echo "$LOOPS"
# echo "$RT_SIZE"
# echo "$LAYER_CONFIG"
# echo "$GPU_ID"
# echo "$GLOBAL_BITFLIP_BUDGET"
# echo "$LOCAL_BITFLIP_BUDGET"


## Specify the number of total layers in NN model
if [ "$NN_MODEL" = "FMNIST" ]
then
    layers_total=4
elif [ "$NN_MODEL" = "CIFAR" ]
then
    layers_total=8
elif [ "$NN_MODEL" = "RESNET" ]
then 
    layers_total=21
else
    echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: FMNIST, CIFAR, RESNET${RESET}\n"
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
        if [ "$NN_MODEL" = "FMNIST" ]
        then
            declare -a PROTECT_LAYERS=(0 0 1 0)
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
            echo -e "\n${RED}$NN_MODEL not supported, check spelling, capitalization & available models: FMNIST, CIFAR, RESNET${RESET}\n"
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

    bash run_auto_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET

elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    # echo "Number of unprotected layers: CUSTOM"

    bash run_auto_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET

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
        
        bash run_auto_all.sh "${PROTECT_LAYERS[@]}" $END1 "${PERRORS[@]}" $END2 $KERNEL_SIZE $NN_MODEL $LOOPS $RT_SIZE $layer_id $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET

        ## Reset for next iteration
        PROTECT_LAYERS[$layer_id]=1

    done
else
    echo "Invalid layer configuration $LAYER_CONFIG."
    exit 1
fi
