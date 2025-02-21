# NetDrift

Presentation [Paper on Springer Nature](https://doi.org/10.1007/978-3-031-78377-7_16)

Models and other large files can be found [here](https://huggingface.co/leonarddavid/NetDrift-models/tree/main)

The NetDrift framework enables profiling BNNs on unreliable RTMs, facilitating the investigation into the impact of misalignment faults on model accuracy. 

It enables controlled error injection in selected BNN layers with varying fault rates and simulates the impact of accumulated misalignments in weight tensors of several BNN models (FashionMNIST, CIFAR10, ResNet18) stored in RTM. 

The framework allows for tuning reliability for performance and vice versa, providing an estimate of the number of inference iterations required for a BNN model to drop below a certain lower threshold, with no protection, limited protection, and full protection, along with the associated impact on performance.

## Prerequisites (Linux Ubuntu or WSL on Windows):

### Anaconda3 instalation (from https://docs.anaconda.com/free/anaconda/install/linux/):

```
curl -O https://repo.anaconda.com/archive/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
bash ~/Downloads/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
source <PATH_TO_CONDA>/bin/activate
conda init
source ~/.bashrc
```

### Create conda environment from pre-made environment file:
```
conda env create -f environment.yml
```

### OR manually:
- create conda environment:
```
conda create -n netdrift python=3.10
conda activate netdrift
```

- Install nvidia-cuda-toolkit 
```
sudo apt-get update 
sudo apt install nvidia-cuda-toolkit
```
or:
```
conda install nvidia::cuda-toolkit
```

- Install pytorch
```
pip install torch torchvision torchaudio
```

*Note: this installation assumes that the latest CUDA drivers are installed on the GPU.
If the GPU is older and does not support the latest drivers, then make suret hat the major version of CUDA is to the same as the version of CUDA in which Pytorch was compiled in*

- others:
```
pip install matplotlib
pip install scipy
```

## Run the Simulator

### before first run (or after any changes in any CUDA kernels)
- if there are errors on windows, try changing from CRLF to LF
```
bash ./code/cuda/install_kernels.sh
```

### Set Flags in `flags.conf`:

Calculation Flags:
- `CALC_RESULTS: True/False` (**Leave True**. _Calculate Results_ for each inference iteration across PERRORS misalignment fault rates)
- `CALC_BITFLIPS: True/False` (_Calculate Bitflips_ per layer for each inference iteration across PERRORS misalignment fault rates)
- `CALC_MISALIGN_FAULTS: True/False` (_Calculate Misalignment Faults_ per layer for each inference iteration across PERRORS misalignment fault rates)
- `CALC_AFFECTED_RTS: True/False` (_Calculate Affected Racetracks_ per layer for each inference iteration across PERRORS misalignment fault rates)

Execution Flags:
- `EXEC_ENDLEN: True/False` (execute Blockhypothesis implementation using the Endlen optimization algorithm)
- `EXEC_ODD2EVEN_DEC: True/False` (execute odd2even decreasing method)
- `EXEC_ODD2EVEN_INC: True/False` (execute odd2even increasing method)
- `EXEC_EVEN2ODD_DEC: True/False` (execute even2odd decreasing method)
- `EXEC_EVEN2ODD_INC: True/False` (execute even2odd decreasing method)
- `EXEC_BIN_REVERT_MID: True/False` (execute binomial revert from middle method)
- `EXEC_BIN_REVERT_EDGES: True/False` (execute binomial revert from edges method)
- `EXEC_EVERY_NRUN=uint` (select after how many runs to execute any EXEC_ flag that is set to True)

Read Flags & Parameters (**At most 1 READ flag in `flags.conf` can the value `True`**):
- `READ_ENDLEN: True/False` (whether to read qweights for each unprotected layer from FOLDER_ENDLEN)
- `FOLDER_ENDLEN: string` (select subfolder to read)
- `READ_TEST: True/False` (whether to read test qweights for each unprotected layer from FOLDER_TEST)
- `FOLDER_TEST: string` (select test subfolder to read)

Print Flags & Parameter:
- `PRNT_LAYER_NAME: True/False` (print unprotected layer types and numbers)
- `PRNT_INPUT_FILE_INFO: True/False` (print qweights input file info for each unprotected layer if any READ flag set to True)
- `PRNT_QWEIGHTS_BEFORE: True/False` (print to file entire weight tensor of each unprotected layer _before_ any modifications)
- `PRNT_QWEIGHTS_AFTER: True/False` (print to file entire weight tensor of each unprotected layer _after_ modifications)
- `PRNT_QWEIGHTS_AFTER_NRUN=uint` (select after how many runs to print if PRNT_QWEIGHTS_AFTER flag is set to True)
- `PRNT_IND_OFF_BEFORE: True/False` (print to file index offsets of each unprotected layer _before_ any modifications)
- `PRNT_IND_OFF_AFTER: True/False` (print to file index offsets of each unprotected layer _after_ modifications)
- `PRNT_IND_OFF_AFTER_NRUN=uint` (select after how many runs to print if PRNT_IND_OFF_AFTER flag is set to True)


### Launch arguments:
- `--operation, -o`         TRAIN or TEST or TEST_AUTO
- `--loops, -l`             Number of experiment loops (0, 100] (if --operation=TRAIN -> use --epochs instead)
- `--model, -m`             Neural network model (MNIST|FMNIST|CIFAR|RESNET)
- `--perrors, -p`           Array of misalignment fault rates (space-separated)
- `--kernel-size, -ks`      Kernel size for conv layers (0 if none)
- `--kernel-mapping, -km`   Mapping configuration of weights in kernels (ROW|COL|CLW|ACW)
- `--rt-size, -rs`          Racetrack size (typically 64)
- `--rt-mapping, -rm`       Mapping configuration of data onto racetracks (ROW|COL|MIX)
- `--layer-config, -lc`     Layer configuration of unprotected layers (ALL|CUSTOM|INDIV)
- `--layers, -ls`           Layer IDs array to be left unprotected (space-separated, used if `--layer-config CUSTOM`)
- `--gpu-id, -g`            GPU ID to run computations on (0|1)
   
Training specific options:
- `--epochs, -e`            Number of training epochs
- `--batch-size, -bs`       Batch size
- `--learning-rate, -lr`    Learning rate
- `--step-size, -ss`        Step size

Optional:
- `--model-path, -mp`        Path to model file to be used for TEST and TEST_AUTO
- `--global-budget, -gb`     Default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work
- `--local-budget, -lb`      Default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work
- `--help, -h`               Show this help message

### Example for TEST operation

```
bash ./run.sh -o TEST -l 2 --model MNIST --perrors 0.01 -ks 0 -km ROW -rs 64 -rm ROW --layer-config CUSTOM --gpu 0
```
Testing for 2 inference iteration(s) the MNIST dataset at misalignment fault rates of 1%, using kernel of size 0 mapped ROW (not needed but required arguments), racetrack of size 64 (tensor mapped row-wise onto racetrack) and protecting layers are **`DEFAULT CUSTOM` layer configuration (defined in `run.sh`)** on GPU 0.

```
bash ./run.sh -o TEST -l 1 --model FMNIST --perrors 0.1 -ks 3 -km CLW -rs 64 -rm COL --layer-config CUSTOM --layers 1 3 --gpu 0
```
Testing for 1 inference iteration(s) the FMNIST dataset at misalignment fault rates of 10%, using kernel of size 3x3 (data mapped clockwise in kernel), racetrack of size 64 (tensor mapped column-wise onto racetrack) and testing for layers 1 and 3 **un**protected on GPU 0.

```
bash ./run.sh -o TEST -l 10 --model CIFAR --perrors 0.1 0.01 -ks 3 -km COL -rs 64 -rm COL --layer-config ALL --gpu 0 -gb 0.15 -lb 0.3
```
Testing for 10 inference iteration(s) the CIFAR dataset at misalignment fault rates of 10% and 1% (separate runs), using kernel of size 3x3 (data mapped column-wise in kernel), racetrack of size 64 (tensor mapped column-wise onto racetrack) and with **all layers unprotected** on GPU 0 with a global bitflip budget of 15% and a local bitflip budget of 30%.

```
bash ./run.sh -o TEST -l 10 --model RESNET --perrors 0.05 -ks 3 -km ACW -rs 64 -rm COL --layer-config INDIV --gpu 0
```
Testing for 10 inference iteration(s) the RESNET dataset at misalignment fault rates of 5%, using kernel of size 3x3 (data mapped anticlockwise in kernel), racetrack of size 64 (tensor mapped column-wise onto racetrack) and **each layer at a time unprotected in individual runs** on GPU 0.

### Example with OPERATION=TRAIN
```
bash ./run.sh -o TRAIN --model FMNIST --perrors 0.0001 -ks 3 -km ROW -rs 64 -rm ROW --layer-config CUSTOM --layers 1 2 --gpu 0 --epochs 10 -bs 256 -lr 0.001 -ss 25
```
Training the FMNIST dataset at misalignment fault rates of 10^(-4), using kernel of size 3x3 (data mapped clockwise in kernel), racetrack of size 64 (tensor mapped row-wise onto racetrack) and the **first and second layers are unprotected*** on GPU 0. Training parameters are: epochs=10, batch_size=256, learning_rate=0.001, step_size=25.


### Troubleshooting
- if there are errors on Windows when running `bash ./code/cuda/install_kernels.sh`
    - try changing from CRLF to LF (or vice-versa)

- If errors during git cloning arise:
    - increase the postBuffer size using `git config --global http.postBuffer 524288000`

- In case CUDA Memory errors arise:
    - flush the VRAM using `torch.cuda.empty_cache()` -> find line in Inference loop in `run.py`

- Set `TEST_BATCH_SIZE` in `run_all.sh` for every NN_MODEL to adjust the amount of images pe batch executed at once in each inference iteration (changes stats and graphs, see terminal output)
   

## Contact
Maintaner [leonard.bereholschi@tu-dortmund.de](mailto:leonard.bereholschi@tu-dortmund.de)

## Acknowledgements

Special thanks goes to [Mikail Yayla](https://github.com/myay) for providing the original SPICE-Torch framework as a base.
