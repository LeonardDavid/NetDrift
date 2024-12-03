# NetDrift

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

### Launch
```
$ bash ./run_auto.sh {arr_perrors} PERRORS {nn_model} {loops} {rt_size} {OPT: arr_layer_ids} {layer_config} {gpu_id} {OPT: global_bitflip_budget} {OPT: local_bitflip_budget}
```

### Arguments:
- `{arr_perrors}`: Array of misalignment fault rates to be tested (floats)
- **`PERRORS`**: REQUIRED: array termination token
- `nn_model`: FMNIST, CIFAR, RESNET
- `loops`: amount of loops (0, 100]
- `rt_size`: racetrack/nanowire size (typically 64)
- `layer_config`: unprotected layers configuration (ALL, CUSTOM, INDIV). Note that if no optional `{arr_layer_ids}` is specified, `CUSTOM` will execute `**DEFAULT** CUSTOM` defined manually in `run_auto.sh` (see optional arguments below).
- `gpu_id`: ID of GPU to use for computations (0, 1) 

### Optional arguments:
- `{arr_layer_ids}`: specify optional layer_ids in an array (starting at 1 upto total_layers) **ONLY BEFORE `layer_config` WITH TERMINATION `CUSTOM`**!
- `global_bitflip_budget`: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (global) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in the whole weight tensor of each layer). Note that both budgets have to be set to values > 0.0 to work.
- `local_bitflip_budget`: default 0.0 (off) -> set to any float value between (0.0, 1.0] to activate (local) bitflip budget (equivalent to allowing (0%, 100%] of total bits flipped in each racetrack). Note that both budgets have to be set to values > 0.0 to work.

### Example in a table TODO
```
$ bash ./run_auto.sh 0.01 0.001 PERRORS FMNIST 10 64 CUSTOM 0
```
Executes at misalignment fault rates of 10% and 1% (separate runs): FMNIST for 10 iterations with racetrack of size 64 using **`DEFAULT CUSTOM` layer configuration (defined in `run_auto.sh`)** on GPU 0.

```
$ bash ./run_auto.sh 0.1 PERRORS CIFAR 10 64 1 5 CUSTOM 0 0.15 0.3
```
Executes at misalignment fault rates of 10%: CIFAR for 10 iterations with racetrack of size 64 with the **first and fifth layers unprotected** on GPU 0 with a global bitflip budget of 15% and a local bitflip budget of 30%.

```
$ bash ./run_auto.sh 0.05 PERRORS RESNET 10 64 INDIV 0
```
Executes at misalignment fault rates of 5%: RESNET for 10 iterations with racetrack of size 64 with **each layer at a time unprotected in individual runs** on GPU 0

# TODO
First, the custom NN needs to be implemented in a separate class in the file located at \texttt{code/python/Model.py}.\ak{Does this mean, taking the new model? We don't need to implement it from scratch.. The model is implemented by someone.}
The adjacent file \texttt{QuantizedNN.py} contains the \texttt{QuantizedConv2D()} and \texttt{QuantizedLinear()} classes which are written specifically for Quantized and Binarized Neural Networks. \ak{We haven't specified these files and modules before..}
These can be used as such in the custom NN, or used as a reference to create a specific convolutional Layer.
It should be noted that the following parameters are required for RTM simulation:\ak{where are these parameters used? In which module?}
\begin{itemize}
    % \item \texttt{error_model}: name of the error model used
    \item \texttt{test\_rtm}: flag whether to use RTM simulation (\texttt{True} by default);
    \item \texttt{block\_size}: size of the nanowire (usually $n\le64$, see Section~\ref{subsec:mapping_weights});
    \item \texttt{index\_offset}: array which stores the offsets of each block, depending on how many times a fault occurs in that block (initialised with zeros);
    \item \texttt{protectLayers}: array that specifies which layers are protected against faults (1) and which are not (0). The size of this array is equal to the number of tested layers;
    \item \texttt{err\_shifts}: array in which the amount of faults in each tested layer is summed up;
\end{itemize}

The simulation of the RTM faults described in Section~\ref{subsec:error_injection} begins after the line of code that reads \texttt{if(self.error\_model is not None)}.\ak{Maybe this is too low level.}
The simulator stores an offset value for each block of the mapped weight tensor in the \texttt{index\_offset} array. 
This value is increased or decreased when a misalignment fault occurs that over-shifts or under-shifts the block, respectively.

A different mapping can be achieved by switching the order of the first two \texttt{for}-loops, which changes the implementation from row-wise to column-wise.
Alternatively, a custom mapping can be designed as needed.

Finally, we apply the error model to the weight tensor with the following code:
\begin{verbatim}
weight = apply_error_model(weight, self.index_offset,
                           self.block_size, self.error_model)
\end{verbatim}

The error injection model is usable out of the box and should not require any modifications. \ak{Great! Was waiting to see this ;)}
% However, if needed, it can be extended in \texttt{code/cuda/binarizationPM1FI}, or used as reference.
However, if needed, it can be extended (or used as reference) under \texttt{code/cuda/binarizationPM1FI}.

All the necessary initializations, i.e., variables, NN models, and functions to start the inference iteration are located in \texttt{run.py}, while helpful scripts can be written to automate multiple iterations, such as the ones used for running our experiments (\texttt{run\_auto.sh} and \texttt{run\_auto\_all.sh}).\ak{This seems not specific to custom NNs but to BNNs as well. Maybe some details from here can be moved to the high level desciption, e.g., in the openning para of Sec 3.}
