# NetDrift

Presentation [Paper on Springer Nature](https://doi.org/10.1007/978-3-031-78377-7_16)

Models and other large files can be found [here](https://huggingface.co/leonarddavid/NetDrift-models/tree/main)

The NetDrift framework enables profiling BNNs on unreliable RTMs, facilitating the investigation into the impact of misalignment faults on model accuracy. 

It enables controlled error injection in selected BNN layers with varying fault rates and simulates the impact of accumulated misalignments in weight tensors of several BNN models (FashionMNIST, CIFAR10, ResNet18) stored in RTM. 

The framework allows for tuning reliability for performance and vice versa, providing an estimate of the number of inference iterations required for a BNN model to drop below a certain lower threshold, with no protection, limited protection, and full protection, along with the associated impact on performance.

## Prerequisites (Linux Ubuntu or WSL on Windows):

### Anaconda3 instalation (from https://docs.anaconda.com/free/anaconda/install/linux/):

```
$ curl -O https://repo.anaconda.com/archive/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
$ bash ~/Downloads/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
$ source <PATH_TO_CONDA>/bin/activate
$ conda init
$ source ~/.bashrc
```

### Create conda environment from pre-made environment file:
```
$ conda env create -f environment.yml
```

### OR manually:
- create conda environment:
```
$ conda create -n netdrift python=3.10
$ conda activate netdrift
```

- Install nvidia-cuda-toolkit 
```
$ sudo apt-get update 
$ sudo apt install nvidia-cuda-toolkit
```
or:
```
$ conda install nvidia::cuda-toolkit
```

- Install pytorch
```
$ pip install torch torchvision torchaudio
```

*Note: this installation assumes that the latest CUDA drivers are installed on the GPU.
If the GPU is older and does not support the latest drivers, then make suret hat the major version of CUDA is to the same as the version of CUDA in which Pytorch was compiled in*

- others:
```
$ pip install matplotlib
$ pip install scipy
```

## Run the Simulator

### before first run (or after any changes in any CUDA kernels)
- if there are errors on windows, try changing from CRLF to LF
```
$ bash ./code/cuda/install_kernels.sh
```

### Set Flags in `flags.conf`:

Calculation Flags:
- `CALC_RESULTS: True/False` (**Leave True**. _Calculate Results_ for each inference iteration across PERRORS misalignment fault rates)
- `CALC_BITFLIPS: True/False` (_Calculate Bitflips_ per layer for each inference iteration across PERRORS misalignment fault rates)
- `CALC_MISALIGN_FAULTS: True/False` (_Calculate Misalignment Faults_ per layer for each inference iteration across PERRORS misalignment fault rates)
- `CALC_AFFECTED_RTS: True/False` (_Calculate Affected Racetracks_ per layer for each inference iteration across PERRORS misalignment fault rates)

NN Model Parameters:
- `kernel_size: 3/5/7` (size of convolutional kernel in convolutional layers, depending on the available trained models)

Execution Flags (**At most 1 EXEC flag in `flags.conf` can the value `True`**):
- `EXEC_RATIO_BLOCKS_IND_OFF: True/False`
- `EXEC_ENDLEN: True/False`
- `EXEC_ENDLEN_IND_OFF: True/False`
- `EXEC_ODD2EVEN_DEC: True/False`
- `EXEC_ODD2EVEN_INC: True/False`
- `EXEC_EVEN2ODD_DEC: True/False`
- `EXEC_EVEN2ODD_INC: True/False`
- `EXEC_BIN_REVERT_MID: True/False`
- `EXEC_BIN_REVERT_EDGES: True/False`

Read Flags & Parameters (**At most 1 READ flag in `flags.conf` can the value `True`**):
- `READ_ECC: True/False` (whether to read qweights for each unprotected layer from FOLDER_ECC)
- `FOLDER_ECC = string` (select subfolder to read)
- `READ_ECC_IND_OFF: True/False` (whether to read qweights for each unprotected layer from FOLDER_ECC_IND_OFF)
- `FOLDER_ECC_IND_OFF = string` (select subfolder to read)
- `READ_ENDLEN: True/False` (whether to read qweights for each unprotected layer from FOLDER_ENDLEN)
- `FOLDER_ENDLEN: string` (select subfolder to read)
- `TYPE_ENDLEN: string` (select which endlen type termination to read)

Print Flags & Parameter:
- `PRNT_LAYER_NAME: True/False` (print unprotected layer types and numbers)
- `PRNT_INPUT_FILE_INFO: True/False` (print qweights input file info for each unprotected layer if any READ flag set to True)
- `PRNT_QWEIGHTS_BEFORE: True/False` (print to file entire weight tensor of each unprotected layer _before_ any modifications)
- `PRNT_QWEIGHTS_AFTER: True/False` (print to file entire weight tensor of each unprotected layer _after_ modifications)
- `PRNT_QWEIGHTS_AFTER_NRUN = 1` (select after how many runs to print if PRNT_QWEIGHTS_AFTER flag is set to True)
- `PRNT_IND_OFF_BEFORE: True/False` (print to file index offsets of each unprotected layer _before_ any modifications)
- `PRNT_IND_OFF_AFTER: True/False` (print to file index offsets of each unprotected layer _after_ modifications)
- `PRNT_IND_OFF_AFTER_NRUN = 1` (select after how many runs to print if PRNT_IND_OFF_AFTER flag is set to True)


### Launch 

For Training
```
$ bash ./run.sh TRAIN {arr_perrors} PERRORS {kernel_size} {nn_model} {loops} {rt_size} {OPT: arr_layer_ids} {layer_config} {gpu_id} {epochs} {batch_size} {lr} {step_size} {OPT: global_bitflip_budget} {OPT: local_bitflip_budget}
```

For Testing
```
$ bash ./run.sh TEST {arr_perrors} PERRORS {kernel_size} {nn_model} {loops} {rt_size} {OPT: arr_layer_ids} {layer_config} {gpu_id} {OPT: global_bitflip_budget} {OPT: local_bitflip_budget}
```

### Arguments:
- `OPERATION`: TRAIN or TEST 
- `{arr_perrors}`: Array of misalignment fault rates to be tested/trained (floats)
- **`PERRORS`: REQUIRED: array termination token**
- `kernel_size`: size of kernel used for calculations in convolutional layers (if none then use 0)
- `nn_model`: FMNIST, CIFAR, RESNET
- `loops`: amount of loops the experiment should run (0, 100] (not used if OPERATION=TEST)
- `rt_size`: racetrack/nanowire size (typically 64)
- `{arr_layer_ids}`: [OPTIONAL] specify optional layer_ids in an array (starting at 1 upto total_layers) ONLY BEFORE `layer_config` WITH TERMINATION `CUSTOM`!
- `layer_config`: unprotected layers configuration (ALL, CUSTOM, INDIV). Note that if no optional {arr_layer_ids} is specified, CUSTOM will execute DEFAULT CUSTOM defined manually in run.sh
- `gpu_id`: ID of GPU to use for computations (0, 1) 

### Additional required arguments if OPERATION = TRAIN
- `epochs`: number of training epochs
- `batch_size`: batch size
- `lr`: learning rate
- `step_size`: step size

### Optional arguments:
- `global_bitflip_budget`: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
- `local_bitflip_budget`: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.

### Example with OPERATION=TEST

```
bash ./run.sh TEST 0.01 PERRORS 0 MNIST 2 64 CUSTOM 0
```
Executes at misalignment fault rates of 1%: MNIST with kernel_size 0 (not needed but required argument) for 2 iterations with racetrack of size 64 using **`DEFAULT CUSTOM` layer configuration (defined in `run.sh`)** on GPU 0.

```
bash ./run.sh TEST 0.1 PERRORS 3 FMNIST 1 64 CUSTOM 0
```
Executes at misalignment fault rates of 10%: FMNIST with kernel_size 3 for 1 iteration with racetrack of size 64 using **`DEFAULT CUSTOM` layer configuration (defined in `run.sh`)** on GPU 0.

```
bash ./run.sh TEST 0.1 0.01 PERRORS 3 CIFAR 10 64 1 5 CUSTOM 0 0.15 0.3
```
Executes at misalignment fault rates of 10% and 1% (separate runs): CIFAR with kernel_size 3 for 10 iterations with racetrack of size 64 with the **first and fifth layers unprotected** on GPU 0 with a global bitflip budget of 15% and a local bitflip budget of 30%.

```
bash ./run.sh TEST 0.05 PERRORS 3 RESNET 10 64 INDIV 0
```
Executes at misalignment fault rates of 5%: RESNET with kernel_size 3 for 10 iterations with racetrack of size 64 with **each layer at a time unprotected in individual runs** on GPU 0.

### Example with OPERATION=TRAIN
```
bash ./run.sh TRAIN 0.0001 PERRORS 3 FMNIST 1 64 1 2 CUSTOM 0 10 256 0.001 25
```
Training at misalignment fault rates of 10^(-4): FMNIST with kernel_size 3 with racetrack of size 64 in which the **first and second layers are unprotected*** on GPU 0. Training parameters are: epochs=10, batch_size=256, learning_rate=0.001, step_size=25.



### Troubleshooting

- If errors during git cloning arise:
    - increase the postBuffer size using `git config --global http.postBuffer 524288000`

- In case CUDA Memory errors arise:
    - flush the VRAM using `torch.cuda.empty_cache()` -> find line in Inference loop in `run.py`

- Set `TEST_BATCH_SIZE` in `run_all.sh` for every NN_MODEL to adjust the amount of images pe batch executed at once in each inference iteration (changes stats and graphs, see terminal output)
   
## Contact
Maintaner [leonard.bereholschi@tu-dortmund.de](mailto:leonard.bereholschi@tu-dortmund.de)

## Acknowledgements

Special thanks goes to [Mikail Yayla](https://github.com/myay) for providing the original SPICE-Torch framework as a base.
