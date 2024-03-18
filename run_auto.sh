#!/bin/bash


NN_MODEL="$1"
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
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit 1
fi

LOOPS=$2
BLOCK_SIZE=$3
COMMAND=$4
GPU=$5

# Check what the argument contains
if [[ $COMMAND == *"ALL"* ]]; then
    # echo "Number of unprotected layers: ALL"
    bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $COMMAND $GPU
elif [[ $COMMAND == *"CUSTOM"* ]]; then
    # echo "Number of unprotected layers: CUSTOM"
    bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $COMMAND $GPU
elif [[ $COMMAND == *"INDIV"* ]]; then
    # echo "Number of unprotected layers: INDIVIDUAL"
    # Loop through all layers individually
    for ((layer=0; layer<layers; layer++))
    do
        # Run the bash file
        #   params:
        ##  $1: NN model name
        ##  $2: loops
        ##  $3: block_size
        ##  $4: unprotected layer
        ##  $5: GPU used
        bash run_auto_all.sh $NN_MODEL $LOOPS $BLOCK_SIZE $layer $GPU
    done
else
    echo "Invalid COMMAND 4th argument."
    # Break or exit the script
    exit 1
fi

