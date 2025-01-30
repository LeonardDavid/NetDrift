import math
import os

# # # numba CUDA debugging -> for better error messages
# # # https://stackoverflow.com/a/68859699
# # needs to appear before `from numba import cuda`
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# # set to "1" for more debugging, but slower performance
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import numpy as np
import torch
import numba
from numba import cuda


# Implements the blockhyp algorithm for a 1D tensor (e.g. a weight tensor)
# If the tensor is 2D or 3D, it should be reshaped to 1D before calling this function
# The tensor is modified in place
# 
# data: the tensor to be modified
# rt_size: the size of the racetracks
#
def blockhyp_endlen_algorithm(data, rt_size):
    
    # Specify the number of racetracks based on the size of the tensor and the racetrack size
    rt = max(math.ceil(data.shape[0]/rt_size), 1)
    print(f"rt_shape: {rt}x{rt_size}")

    # For each racetrack...
    for i in range(rt):
        
        # Initializations
        rt_tuples = []                          # store the tuples containing the information of the sliding windows for each racetrack
        count = 0                               # count the sign changes
        k = 0                                   # index of the tuples
        bitgroup = [0, 0, 0]                    # in a sliding window of size 3, store the lengths of bitgroups
        i_mid = [0, 0, 0]                       # store the starting indices of the middle bitgroup
        current_sign = data[i * rt_size]        # sign of the current bitgroup, initialize with sign of the first bitgroup

        # ...of size rt_size bits
        for j in range(rt_size):

            # Iterate through every weight
            index = i * rt_size + j

            # This will first hit after the first bitgroup (because of the initialization), and then every time the sign changes
            if data[index] != current_sign:
                i_mid[(count+1)%3] = index      # store the starting index of the middle bitgroup
                current_sign = -current_sign    # flip the sign to match the current bitgroup
                count += 1                      # increment the count of sign changes

                if count > 2:
                    # Create tuple of format (tuple_index, bitgroup_length_of_window, required_flips, starting_index_of_middle_bitgroup)
                    # Retrieve middle bitgroup by clever indexing using the sliding window property
                    rt_tuples.append((k, sum(bitgroup), bitgroup[(k+1)%3], i_mid[(k+1)%3]))
                    k += 1                      # increment the index of the tuples
                    bitgroup[count%3] = 0       # reset the length of the current bitgroup
            
            # To save memory, after sliding the window, store the new bitgroup length in the unused part of the array
            bitgroup[count%3] += 1 # <=> abs(data[index]) <=> increment the length of the current bitgroup 

        # Handle the last tuple
        rt_tuples.append((k, sum(bitgroup), bitgroup[(k+1)%3], i_mid[(k+1)%3]))

        # Sort the list of tuples (tuple_index, endlen, flips, index_mid) by endlen descending, then by flips ascending
        rt_tuples.sort(key=lambda x: (-x[1], x[2]))

        # Greedily select the bitgroups with the highest endlen, and the lowest number of required flips, as long as there are still available bitgroups left
        while len(rt_tuples) > 0: # while rt_tuples:

            tuple_index = rt_tuples[0][0]                         # first tuple  always contains the maximum (it has been sorted previously)
            start_index_mid = rt_tuples[0][3]                     # starting index of the middle bitgroup to be flipped
            final_index_mid = start_index_mid + rt_tuples[0][2]   # start index + required bitflips
            
            # Flip the middle bitgroup
            for idx in range(start_index_mid, final_index_mid):
                # print(f"{idx}: flipped {data[idx]} -> {-data[idx]}")
                data[idx] = -data[idx]

            # Remove selected tuple after storing it, along with its neighbouring tuples
            rt_tuples = [t for t in rt_tuples if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]


@cuda.jit
def blockhyp_endlen_algorithm_parallel_kernel(data, rt_size):

    # Get the racetrack index from the corresponding grid containing the block containing the thread
    rt_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Variable definitions
    count = 0                               # count the sign changes                         
    k = 0                                   # index of the tuples (max rt_size)            
    tuple_counts = 0                        # count number of valid tuples (max rt_size)
    current_sign = data[rt_id * rt_size]    # sign of the current bitgroup, initialize with sign of the first bitgroup

    # Array definitions
    # TODO have to use a fixed size of 64... tried to use rt_size but it didn't work, tried const.array_like but it didn't work (https://stackoverflow.com/a/63311668)
    rt_tuples = cuda.local.array((64, 4), dtype=numba.int64)    # store the tuples containing the information of the sliding windows for each racetrack
    bitgroup = cuda.local.array(3, dtype=numba.int32)           # in a sliding window of size 3, store the lengths of bitgroups  
    i_mid = cuda.local.array(3, dtype=numba.int32)              # store the starting indices of the middle bitgroup

    # Initialize arrays
    for x in range(rt_size):
        for y in range(4):
            rt_tuples[x, y] = 0
    
    for idx in range(3):
        bitgroup[idx] = 0
        i_mid[idx] = 0


    # Process racetrack of size rt_size bits
    for j in range(rt_size):

        # Iterate through every weight
        index = rt_id * rt_size + j
        
        # This will first hit after the first bitgroup (because of the initialization), and then every time the sign changes
        if index < data.size and data[index] != current_sign:
            i_mid[(count+1)%3] = index      # store the starting index of the middle bitgroup
            current_sign = -current_sign    # flip the sign to match the current bitgroup
            count += 1                      # increment the count of sign changes

            if count > 2:
                # Create tuple of format (tuple_index, bitgroup_length_of_window, required_flips, starting_index_of_middle_bitgroup)
                # Retrieve middle bitgroup by clever indexing using the sliding window property    
                
                if k < rt_size:  # Prevent buffer overflow
                    rt_tuples[k, 0] = k                                         # tuple_index
                    rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]   # bitgroup_length_of_window (aka endlen)
                    rt_tuples[k, 2] = bitgroup[(k+1)%3]                         # required_flips
                    rt_tuples[k, 3] = i_mid[(k+1)%3]                            # starting_index_of_middle_bitgroup
                    k += 1              # increment the index of the tuples

                bitgroup[count%3] = 0   # reset the length of the current bitgroup
        
        # To save memory, after sliding the window, store the new bitgroup length in the unused part of the array    
        if index < data.size:
            bitgroup[count%3] += 1      # <=> abs(data[index]) <=> increment the length of the current bitgroup

    # Handle last tuple by storing the information about it gathered previously at the end
    if k < rt_size:  # Prevent buffer overflow
        rt_tuples[k, 0] = k
        rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
        rt_tuples[k, 2] = bitgroup[(k+1)%3]
        rt_tuples[k, 3] = i_mid[(k+1)%3]
        tuple_counts = k + 1            # keep track of the number of valid tuples (max rt_size)

    # Sort the list of tuples (tuple_index, endlen, flips, index_mid) by endlen descending, then by flips ascending
    # Naive bubble sort in place, using a simple approach due to limited Python features in Numba
    for _ in range(tuple_counts):
        for j in range(tuple_counts - 1):
            endlen_j   = rt_tuples[j,   1]
            flips_j    = rt_tuples[j,   2]
            endlen_j1  = rt_tuples[j+1, 1]
            flips_j1   = rt_tuples[j+1, 2]
            # Compare (endlen, flips)
            if (endlen_j < endlen_j1) or (endlen_j == endlen_j1 and flips_j > flips_j1):
                # Swap
                for k in range(4):
                    temp = rt_tuples[j, k]
                    rt_tuples[j, k] = rt_tuples[j+1, k]
                    rt_tuples[j+1, k] = temp

    # Greedily select the bitgroups with the highest endlen, and the lowest number of required flips, as long as there are still available bitgroups left
    processed = 0   # keep track of the number of processed tuples

    while True:

        # Break the loop if all tuples have been processed
        if processed >= tuple_counts:
            break
        idx_tuple = processed   # index of the tuple to be processed

        tuple_index     = rt_tuples[idx_tuple, 0]   # first tuple  always contains the maximum (it has been sorted previously)
        endlen          = rt_tuples[idx_tuple, 1]   # bitgroup_length_of_window (aka endlen)
        flips           = rt_tuples[idx_tuple, 2]   # required_flips
        start_index_mid = rt_tuples[idx_tuple, 3]   # starting index of the middle bitgroup to be flipped
        
        # Skip if tuple has been removed, and increment processed counter to break the loop if all tuples have been processed
        if endlen == 0:
            processed += 1
            continue

        # Calculate the final index of the middle bitgroup to be flipped
        final_index_mid = start_index_mid + flips   # start index + required bitflips

        # Flip bits in the middle bitgroup, changing data array in place 
        for idx in range(start_index_mid, final_index_mid):
            if idx < data.size:
                data[idx] = -data[idx]

        # Remove selected tuple after storing it, along with its neighbouring tuples (by setting endlen=0)
        for j in range(tuple_counts):
            tj_index = rt_tuples[j, 0]
            if tj_index in [tuple_index - 1, tuple_index, tuple_index + 1]:
            # if tj_index >= tuple_index - 1 and tj_index <= tuple_index + 1:
                rt_tuples[j, 1] = 0  # set endlen to 0 to mark as removed

        processed += 1 # increment after finishing processing selected tuple


def blockhyp_endlen_algorithm_parallel(data, rt_size):

    # Calculate number of racetracks
    rt = max(math.ceil(data.shape[0]/rt_size), 1)
    # print(f"rt_shape: {rt}x{rt_size}")
    
    # Initialize CUDA context
    cuda.select_device(0)  

    # Allocate memory on Device and copy data from Host to Device, also convert to numpy array
    data_gpu = cuda.to_device(data.detach().cpu().numpy())
    
    # Get the maximum number of threads per block for the current GPU, usually 1024
    device = cuda.get_current_device()                      # disable when debugging
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK    # max_threads_per_block = 1024

    # Define block and grid structures
    threads_per_block_x = min(rt, max_threads_per_block)                            # although 3D arrangement is used, 1024x1x1 is preferred since it uses MAX_THREADS_PER_BLOCK
    blocks_per_grid_x = (rt + (threads_per_block_x - 1)) // threads_per_block_x     # Maximum grid size is (2^31-1)x65535x65535, so a tensor of size larger than (2^31-1)*1024 would be needed to exceed the grid size in x-dimension, therefore y- and z-dimensions are 1

    # Launch kernel
    blockhyp_endlen_algorithm_parallel_kernel[blocks_per_grid_x, threads_per_block_x](data_gpu, rt_size)

    # Synchronize results from all threads first before copying any data from Device to Host
    cuda.synchronize()

    # Copy data back to CPU memory, and convert to PyTorch tensor
    data.copy_(torch.from_numpy(data_gpu.copy_to_host()))

