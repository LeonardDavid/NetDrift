#!/bin/bash

# required args
NN_MODEL="$1"
LOOPS=$2
BLOCK_SIZE=$3
LAYER_CONFIG=$4
GPU_ID=$5

# optional args
GLOBAL_BITFLIP_BUDGET=${6:-0.0}
LOCAL_BITFLIP_BUDGET=${7:-0.0}


# Specify the number of total layers in NN model
if [ "$NN_MODEL" = "FMNIST" ]
then
    layers=4
elif [ "$NN_MODEL" = "CIFAR" ]
then
    layers=8
elif [ "$NN_MODEL" = "RESNET" ]
then 
    layers=21
else
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling, capitalization & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit 1
fi


# Check what the argument contains
if [[ $LAYER_CONFIG == *"ALL"* ]]; then
    # echo "Number of unprotected layers: ALL"
    bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
elif [[ $LAYER_CONFIG == *"CUSTOM"* ]]; then
    # echo "Number of unprotected layers: CUSTOM"
    bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $LAYER_CONFIG $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
elif [[ $LAYER_CONFIG == *"INDIV"* ]]; then
    # echo "Number of unprotected layers: INDIVIDUAL"
    # Loop through all layers individually
    for ((layer=0; layer<layers; layer++))
    do
        # Run the bash file
        #
        #   required params:
        ##  $1: NN model name
        ##  $2: loops
        ##  $3: block_size
        ##  $4: unprotected layer
        ##  $5: GPU used
        #
        #   optional params:
        ##  $6: global bitflip budget (default 0.0)
        ##  $7: local bitflip budget (default 0.0)
        
        bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $layer $GPU_ID $GLOBAL_BITFLIP_BUDGET $LOCAL_BITFLIP_BUDGET
    done
else
    echo "Invalid layer configuration (4th argument)."
    # Break or exit the script
    exit 1
fi
