import os
import torch
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

import math
import random
import numpy as np

# # # numba CUDA debugging -> for better error messages
# # # https://stackoverflow.com/a/68859699
# # needs to appear before `from numba import cuda`
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# # set to "1" for more debugging, but slower performance
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import metrics.binomial_revert.binomial_revert as bin_revert
import metrics.blockhyp_endlen.blockhyp_endlen as blockhyp


@cuda.jit
def calc_index_offset_kernel(rng_states, index_offset, misalign_faults, rt_size, ap_reads, rt_error):
    """
    A CUDA kernel function that simulates racetrack memory operations with potential misalignment faults.
    The function uses xoroshiro128p, a fast pseudo-random number generator (PRNG) specifically designed 
    for parallel computations. Xoroshiro128p stands for XOR/rotate/shift/rotate with 128-bit state 
    and p-rotations, providing a period of 2^128-1.
    Args:
        rng_states (cuda.random.xoroshiro128p):
            Random number generator states for CUDA threads
        index_offset (ndarray): 
            2D array tracking the current position offset for each racetrack
        misalign_faults (ndarray):
            2D array counting the number of misalignment faults per racetrack
        rt_size (int):
            Size of the racetrack
        ap_reads (int):
            Number of access port reads to perform
        rt_error (float):
            Probability of a misalignment fault occurring (between 0 and 1)
    Notes:
        - The function uses thread-safe random number generation for parallel execution
        - Misalignment faults can occur with probability rt_error
        - When a fault occurs, the offset is adjusted by ±1 within the bounds of rt_size/2 (i.e., buffer on each side)
        - Each thread processes one racetrack at position (i,j)
    """
    
    # Get the thread's (i.e. racetrack's) unique position in the grid
    i, j = cuda.grid(2)

    # Check if the thread/racetrack is within the bounds of the index_offset array
    if i < index_offset.shape[0] and j < index_offset.shape[1]:
        # Read every value from the racetrack (i,j)
        
        # Depending on the rt_mapping, the AP reads a different number of bits per shift
        # Start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
        for k in range(1, ap_reads):

            # Use thread-safe random number generation
            rand = xoroshiro128p_uniform_float32(rng_states, i*index_offset.shape[1] + j)
            if rand < rt_error:
            
                if misalign_faults.shape[0] > 1:    # i.e. if CALC_MISALIGN_FAULTS flag is True
                    misalign_faults[i, j] += 1
            
                # offset cannot misalign over the buffer on each side (rt_size/2)
                if (abs(index_offset[i, j]) < rt_size/2): 
                    
                    # 50/50 random possibility of right or left misalign_fault
                    rand2 = xoroshiro128p_uniform_float32(rng_states, rand)
                    if rand2 > 0.5:
                        # right misalign_fault
                        index_offset[i, j] += 1
                    else:
                        # left misalign_fault
                        index_offset[i, j] -= 1


@cuda.jit
def simulate_racetrack_kernel(rng_states, qweight_in, qweight_out, index_offset, rt_size):
    """
    Simulates a racetrack memory device using CUDA parallel processing.
    This kernel function processes each position in the racetrack memory in parallel,
    shifting weights according to the given offset and handling boundary conditions.
    Args:
        rng_states (cuda.random.xoroshiro128p):
            Random number generator states for CUDA threads
        qweight_in (ndarray):
            2D Input array containing the initial (unshifted) weights in the racetrack memory
        qweight_out (ndarray):
            2D Output array where the shifted weights to be read out will be stored
        index_offset (ndarray):
            2D array containing the position offsets for each racetrack
        rt_size (int):
            Size of each racetrack/number of bits per racetrack.
    Notes:
        - The function operates on a 2D grid where each thread processes one position
        - Out-of-bounds shifts are handled by generating random values (-1 or 1)
        - The racetrack shift direction is determined by the sign of index_offset:
            * Negative offset shift to the left
            * Positive offset shift to the right
    """
    
    # Get the thread's unique position in the grid
    i, j = cuda.grid(2)

    # Check if the thread is within the bounds of the index_offset array
    if i < index_offset.shape[0] and j < index_offset.shape[1]:
        
        # Read every value from the racetrack (i,j)
        for k in range(rt_size):

            # Calculate the output and input indices for the racetrack, where the racetrack is shifted by the index_offset value 
            # the i index of qweight is the same as the i index of racetrack (index_offset), since the internal representation is shaped in the same way, regardless of rt_mapping
            q_out_index = j*rt_size + k                         # the index of the weight read out at the current position is stable
            q_in_index = q_out_index - index_offset[i, j]       # (minus intentional) if index_offset is negative, the racetrack is shifted to the left, and vice versa
            
            if q_out_index < qweight_out.shape[1]:  # in case the racetrack is not fully filled, continue (since the output index might be out of bounds)

                # Copy the weight from the shifted position to be the actual weight to be read out
                if j * rt_size <= q_in_index <= (j+1) * rt_size - 1 and q_in_index < qweight_in.shape[1]:
                    # only if the shifted index is within the bounds of the racetrack and the input weight array (in case racetrack is not fully filled)
                    qweight_out[i, q_out_index] = qweight_in[i, q_in_index]
                else:
                    # if out of bounds, read a random value either -1 or 1
                    # note that original values are still stored on the racetrack (i.e. in the buffer)
                    rand = xoroshiro128p_uniform_float32(rng_states, q_out_index)
                    qweight_out[i, q_out_index] = 1 if rand > 0.5 else -1


def racetrack_sim(quantized_weight, index_offset, rt_size, rt_error, ap_reads, flags, nr_run, layerNR):
    """
    Simulates misalignment faults in racetrack memory using CUDA numba parallelization.
    This function simulates the behavior of racetrack memory, including position errors
    and specific memory access patterns. It uses CUDA kernels for parallel computation
    of index offsets and memory operations.
    Args:
        quantized_weight (torch.Tensor): 
            The 2D quantized weight tensor to be stored in racetrack memory
        index_offset (numpy.ndarray): 
            2D Array storing position offsets for each racetrack
        rt_size (int): 
            Size of each racetrack/number of bits per racetrack
        rt_error (float): 
            Probability of misalignment fault occurrence
        ap_reads (int):
            Number of access port reads to perform
        flags (dict): 
            Configuration flags for various simulation options (found in flags.conf)
        nr_run (int): 
            Current run number used if modifications are specified to be applied every N runs (EXEC_EVERY_NRUN)
        layerNR (int): 
            Current layer number being processed (used for analysis and debugging)
    Returns:
        tuple: Contains:
            - quantized_weight (torch.Tensor): Modified weight tensor as read out from racetrack memory
            - index_offset (numpy.ndarray): Updated position offsets for each racetrack
            - misalign_faults_sum (int): Total number of misalignment faults which occured during read operations
    Notes:
        - quantized_weight expects a 2D tensor, if a different shaped tensor is passed, it must be reshaped to 2D before calling the function (e.g. for conv layers)
        - ROW mapping: requires rt_size shifts to read a single word (1 bit per shift)
        - COL mapping: reads an entire word per shift (bit b_i from each racetrack)
        - Supports various modification flags like bin_revert, odd2even, blockhyp
        - Handles misalignment fault tracking when CALC_MISALIGN_FAULTS flag is enabled
    """

    ## CUDA setup
    # Get the maximum number of threads per block for the current GPU, usually 1024
    device = cuda.get_current_device()                          # disable line for debugging
    # max_threads_per_block = device.MAX_THREADS_PER_BLOCK      # max_threads_per_block = 1024
    # Initialize CUDA context
    cuda.select_device(device.id)                               # cuda.select_device(0) for debugging

    # Allocate memory on Device and copy data from Host to Device
    index_offset_gpu = cuda.to_device(index_offset)
    
    # Define an array to store the number of misalignment faults per racetrack
    # This array is only used if the CALC_MISALIGN_FAULTS flag is set to True
    if flags.get("CALC_MISALIGN_FAULTS") == "True":
        misalign_faults = np.zeros_like(index_offset)
    else:
        # If the flag is set to False, create an empty array to avoid unnecessary 
        # calculations and large data movement while keeping compatibility with the rest of the code
        misalign_faults = np.zeros((1,1))
    
    misalign_faults_gpu = cuda.to_device(misalign_faults)
    
    # Calculate the number of blocks and threads per block
    num_rows, num_cols = index_offset.shape
    threads_per_block = (min(num_rows, 32), min(num_cols, 32))      # 32x32 threads per block, max_threads_per_block = 1024
    blocks_per_grid = (                                             # keep grid shape <=> index_offset shape as best as possible
        math.ceil(num_rows / threads_per_block[0]),                 # max blocks in x-direction: 2^31 - 1 => for a rt_size of 64, this will be exceeded when tensor is larger than 64*(2^31 - 1)
        math.ceil(num_cols / threads_per_block[1])                  # max blocks in y-direction: 65535
    )

    # Initialize random number generator states
    rng_states = create_xoroshiro128p_states(threads_per_block[0] * threads_per_block[1] * blocks_per_grid[0] * blocks_per_grid[1], seed=random.randint(1, 1000)) # constant seed leads to same index offsets and misalign_faults at every inference iteration (i.e. if some racetrack has 0 offset, it will never misalign)


    ## Launch kernel to calculate index_offset for each racetrack (1 RT per 1 thread)
    calc_index_offset_kernel[blocks_per_grid, threads_per_block](rng_states, index_offset_gpu, misalign_faults_gpu, rt_size, ap_reads, rt_error)

    # Synchronize results from all threads first before copying any data from Device to Host
    cuda.synchronize()

    # Copy data back to CPU memory, and convert to PyTorch tensor
    index_offset = index_offset_gpu.copy_to_host()
    misalign_faults = misalign_faults_gpu.copy_to_host()


    ## Flag mods

    ## Modify index_offset according to flags set in flags.conf (e.g. bin_revert, odd2even, even2odd)
    # Adjust index_offset and misalign_faults accordingly
    index_offset, misalign_faults = execute_flags_index_offset(flags, index_offset, misalign_faults, nr_run, layerNR)

    ## Modify quantized_qweight according to flags set in flags.conf (e.g. blockhyp algorithm)
    # Allocate memory on Device and copy (modified) data from Host to Device
    quantized_weight_gpu = cuda.to_device(quantized_weight.detach().cpu().numpy())
    index_offset_gpu = cuda.to_device(index_offset)

    qweight_out = np.zeros(quantized_weight.shape)  # for simulate_racetrack_kernel
    qweight_out_gpu = cuda.to_device(qweight_out)

    # Launch kernel to apply blockhyp_endlen algorithm
    if flags.get("EXEC_ENDLEN") == "True":
        blockhyp.blockhyp_endlen_algorithm_parallel_kernel[blocks_per_grid, threads_per_block](quantized_weight_gpu, rt_size)
        
        # Synchronize results from all threads first before launching the next kernel
        cuda.synchronize()
        # Not necessary to copy quantized_weight_gpu back to CPU memory, as the same (but modified) data will be used in simulate_racetrack_kernel


    ## Launch kernel to simulate racetrack memory operations (read shifted position for each qweight in every racetrack)
    simulate_racetrack_kernel[blocks_per_grid, threads_per_block](rng_states, quantized_weight_gpu, qweight_out_gpu, index_offset_gpu, rt_size)
    
    # Synchronize results from all threads first before copying any data from Device to Host
    cuda.synchronize()

    # Copy data back to CPU memory, and convert to PyTorch tensor (?)
    quantized_weight.copy_(torch.from_numpy(qweight_out_gpu.copy_to_host()))


    return quantized_weight, index_offset, np.sum(misalign_faults)


def execute_flags_index_offset(flags, index_offset, misalign_faults, nr_run, layerNR):
    """
    This function applies various methods to modify index_offset values and tracks
    misalignment faults that occur during these modifications. The modifications include binomial
    reversion (from middle or edges) and odd/even number adjustments.
    Args:
        flags (dict): 
            Dictionary of configuration flags controlling which modifications to apply
        index_offset (numpy.ndarray): 
            2D array of index offset values to be modified for each racetrack
        misalign_faults (numpy.ndarray): 
            2D array tracking number of misalignment faults for each racetracks
        nr_run (int): 
            Current run number used if modifications are specified to be applied every N runs (EXEC_EVERY_NRUN)
        layerNR (int): 
            Current layer number being processed (used for analysis and debugging)
    Returns:
        tuple: (index_offset, misalign_faults)
            - index_offset (numpy.ndarray): Modified index offset values after modifications
            - misalign_faults (numpy.ndarray): Updated misalignment fault counts
    Key Flags:
        CALC_MISALIGN_FAULTS: Track misalignment faults during modifications
        PRNT_IND_OFF_BEFORE: Print index offsets before modifications
        EXEC_EVERY_NRUN: Apply modifications every N runs
        EXEC_BIN_REVERT_MID: Apply binomial reversion from middle
        EXEC_BIN_REVERT_EDGES: Apply binomial reversion from edges
        EXEC_ODD2EVEN_DEC/INC: Convert odd offsets to even (decrease/increase)
        EXEC_EVEN2ODD_DEC/INC: Convert even offsets to odd (decrease/increase)
        PRNT_IND_OFF_AFTER: Print index offsets after modifications
    """

    # Needed for misalign_fault adjustments
    if flags.get("CALC_MISALIGN_FAULTS") == "True":
        index_offset_before = index_offset.copy()


    ### PRNT BEFORE ###
    if flags.get("PRNT_IND_OFF_BEFORE") == "True" and nr_run==1:
        with open("ind_off_"+str(layerNR)+"_run_0.txt", "w") as f:
            for i in range(0, index_offset.shape[0]):
                for j in range(0, index_offset.shape[1]):
                    f.write(str(index_offset[i][j]) + " ")
                f.write("\n")


    ### BINOMIAL REVERT ###

    # Reset some index offset values above a certain threshold
    # Theoretical approach, because this would mean that ECC is applied only to some racetracks, but in practice it is either full ECC or no ECC
    # Some possible thresholds: cut 80% of the amount of values starting from the middle (0, 1, -1, 2, -2 etc) and leave 20% on the edges
    # or possibility 2: cut 80% of the total sizes starting from the edges (40% on the right, 40% on the left)
    # Significant overhead especially for counting (and creating histogram)

    if flags.get("EXEC_BIN_REVERT_MID") == "True":

        # 80/20 from middle (total elements)
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = bin_revert.revert_elements_2d_mid(index_offset)
            
            # adjust misalign_faults accordingly in racetracks where index_offset was modified (add absolute amount of shift before modifying racetrack)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[index_offset_before != index_offset] += np.abs(index_offset_before[index_offset_before != index_offset])
    
    if flags.get("EXEC_BIN_REVERT_EDGES") == "True":

        # 80/20 from edges (total bins)
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = bin_revert.revert_elements_2d_edges(index_offset)

            # adjust misalign_faults accordingly in racetracks where index_offset was modified (add absolute amount of shift before modifying racetrack)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[index_offset_before != index_offset] += np.abs(index_offset_before[index_offset_before != index_offset])


    ### ODD2EVEN ###

    # Perform additional shift in case of an odd number of misalignment faults (to help repair alternating structures):
    # This shifting also induces its own misalignment faults with the same probability
    # -> we leave it out for now
    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
    # This would add additional overhead in practice

    if flags.get("EXEC_ODD2EVEN_DEC") == "True":
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = np.where((index_offset % 2 != 0), index_offset - np.sign(index_offset), index_offset)
            
            # adjust misalign_faults accordingly in racetracks (add 1 where index_offset was modified where index_offset was modified and was non-zero before)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[(index_offset_before != index_offset) & (index_offset_before != 0)] += 1

    if flags.get("EXEC_ODD2EVEN_INC") == "True":
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = np.where((index_offset % 2 != 0), index_offset + np.sign(index_offset), index_offset)

            # adjust misalign_faults accordingly in racetracks (add 1 where index_offset was modified where index_offset was modified and was non-zero before)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[(index_offset_before != index_offset) & (index_offset_before != 0)] += 1
        

    ### EVEN2ODD ###

    # Perform additional shift in case of an even number of misalignment faults:
    # This shifting also induces its own misalignment faults with the same probability
    # -> we leave it out for now
    # -> in future, we could create a best-case and worst-case, in which latter would be that shift error happens also during this "correction"
    # This would add additional overhead in practice

    if flags.get("EXEC_EVEN2ODD_DEC") == "True":
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = np.where((index_offset != 0) & (index_offset % 2 == 0), index_offset - np.sign(index_offset), index_offset)

            # adjust misalign_faults accordingly in racetracks (add 1 where index_offset was modified where index_offset was modified)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[index_offset_before != index_offset] += 1

    if flags.get("EXEC_EVEN2ODD_INC") == "True":
        if nr_run % int(flags.get("EXEC_EVERY_NRUN")) == 0:
            index_offset = np.where((index_offset != 0) & (index_offset % 2 == 0), index_offset + np.sign(index_offset), index_offset)
            
            # adjust misalign_faults accordingly in racetracks (add 1 where index_offset was modified where index_offset was modified)
            if flags.get("CALC_MISALIGN_FAULTS") == "True":
                misalign_faults[index_offset_before != index_offset] += 1


    ### PRNT AFTER ###

    if flags.get("PRNT_IND_OFF_AFTER") == "True" and flags.get("PRNT_IND_OFF_AFTER_NRUN") == str(nr_run-1):
        with open("ind_off_"+str(layerNR)+"_run_"+str(nr_run-1)+".txt", "w") as f:
            for i in range(0, index_offset.shape[0]):      # 
                for j in range(0, index_offset.shape[1]):  #
                    f.write(str(index_offset[i][j]) + " ")
                f.write("\n")


    return index_offset, misalign_faults

