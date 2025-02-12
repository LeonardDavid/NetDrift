import numpy as np
import torch
import numba
from numba import cuda

# # # numba CUDA debugging -> for better error messages
# # # https://stackoverflow.com/a/68859699
# # needs to appear before `from numba import cuda`
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# # set to "1" for more debugging, but slower performance
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import math


@cuda.jit
def blockhyp_endlen_algorithm_parallel_kernel(qweight, rt_size):
    """
    CUDA kernel implementation of the Blockhypothesis using the Endlen optimization algorithm.
    This kernel processes 2D quantized weight tensors organized in (parallel) racetracks by
    selectively flipping bits in bitgroups based on their "end length" metric. The algorithm
    uses a sliding window approach to identify and process bitgroups.
    Args:
        qweight (cuda.device_array): 
            2D array of quantized weights (+1/-1) organized in racetracks of rt_size
        rt_size (int): 
            Size of each racetrack/number of bits per racetrack
    Implementation Details:
        - Uses a sliding window of size 3 to identify bitgroups
        - For each window, calculates tuple containing:
            * tuple_index: Index of the tuple
            * endlen: Total length of the window (sum of all bitgroup lengths)
            * required_flips: Number of bits that need to be flipped in order to achieve endlen
            * starting_index: Start position of middle bitgroup
        - Sorts tuples by endlen (descending) and required flips (ascending)
        - Greedily processes tuples to maximize endlen and minimize required flips
        - Modifies qweight array in-place by flipping selected bit groups
    Note:
        Currently uses fixed-size local arrays (64 elements) due to CUDA/Numba limitations
        In the future, the rt_size parameter should be used to define the array size
    """

    # Get the racetrack index from the corresponding grid containing the block containing the thread
    rt_i, rt_j = cuda.grid(2)

    if rt_i < qweight.shape[0] and rt_j < qweight.shape[1]:
        
        # row index of the weight tensor (qweight) is the same as the rt_i racetrack index
        w_i = rt_i
    
        # Variable definitions
        count = 0                                       # count the sign changes                         
        k = 0                                           # index of the tuples (max rt_size)            
        tuple_counts = 0                                # count number of valid tuples (max rt_size)
        current_sign = qweight[w_i, rt_j * rt_size]     # sign of the current bitgroup, initialize with sign of the first bitgroup

        # Array definitions
        # TODO have to use a fixed size of 64... tried to use rt_size but it didn't work, tried const.array_like but it didn't work (https://stackoverflow.com/a/63311668)
        rt_tuples = cuda.local.array((64, 4), dtype=numba.int64)    # store the tuples containing the information of the sliding windows for each racetrack
        bitgroup = cuda.local.array(3, dtype=numba.int32)           # in a sliding window of size 3, store the lengths of bitgroups  
        j_mid = cuda.local.array(3, dtype=numba.int32)              # store the starting indices of the middle bitgroup

        # Initialize arrays
        for x in range(rt_size):
            for y in range(4):
                rt_tuples[x, y] = 0
        
        for idx in range(3):
            bitgroup[idx] = 0
            j_mid[idx] = 0


        # Process racetrack of size rt_size bits
        for bit_index in range(rt_size):

            # Iterate through every weight
            w_j = rt_j * rt_size + bit_index
            
            # This will first hit after the first bitgroup (because of the initialization), and then every time the sign changes
            if w_j < qweight.shape[1] and qweight[w_i, w_j] != current_sign:
                j_mid[(count+1)%3] = w_j        # store the starting index of the middle bitgroup
                current_sign = -current_sign    # flip the sign to match the current bitgroup
                count += 1                      # increment the count of sign changes

                if count > 2:
                    # Create tuple of format (tuple_index, bitgroup_length_of_window, required_flips, starting_index_of_middle_bitgroup)
                    # Retrieve middle bitgroup by clever indexing using the sliding window property    
                    
                    if k < rt_size:  # Prevent buffer overflow
                        rt_tuples[k, 0] = k                                         # tuple_index
                        rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]   # bitgroup_length_of_window (aka endlen)
                        rt_tuples[k, 2] = bitgroup[(k+1)%3]                         # required_flips
                        rt_tuples[k, 3] = j_mid[(k+1)%3]                            # starting_index_of_middle_bitgroup
                        k += 1              # increment the index of the tuples

                    bitgroup[count%3] = 0   # reset the length of the current bitgroup
            
            # To save memory, after sliding the window, store the new bitgroup length in the unused part of the array    
            if w_j < qweight.shape[1]:
                bitgroup[count%3] += 1      # <=> abs(data[index]) <=> increment the length of the current bitgroup

        # Handle last tuple by storing the information about it gathered previously at the end
        if k < rt_size:  # Prevent buffer overflow
            rt_tuples[k, 0] = k
            rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
            rt_tuples[k, 2] = bitgroup[(k+1)%3]
            rt_tuples[k, 3] = j_mid[(k+1)%3]
            tuple_counts = k + 1            # keep track of the number of valid tuples (max rt_size)

        # Sort the list of tuples (tuple_index, endlen, flips, index_mid) by endlen descending, then by flips ascending
        # Naive bubble sort in place, using a simple approach due to limited Python features in Numba
        for _ in range(tuple_counts):
            for id in range(tuple_counts - 1):
                endlen_j   = rt_tuples[id,   1]
                flips_j    = rt_tuples[id,   2]
                endlen_j1  = rt_tuples[id+1, 1]
                flips_j1   = rt_tuples[id+1, 2]
                # Compare (endlen, flips)
                if (endlen_j < endlen_j1) or (endlen_j == endlen_j1 and flips_j > flips_j1):
                    # Swap
                    for k in range(4):
                        temp = rt_tuples[id, k]
                        rt_tuples[id, k] = rt_tuples[id+1, k]
                        rt_tuples[id+1, k] = temp

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
            for w_jx in range(start_index_mid, final_index_mid):
                if w_jx < qweight.shape[1]:
                    qweight[w_i, w_jx] = -qweight[w_i, w_jx]

            # Remove selected tuple after storing it, along with its neighbouring tuples (by setting endlen=0)
            for id in range(tuple_counts):
                tj_index = rt_tuples[id, 0]
                if tj_index in [tuple_index - 1, tuple_index, tuple_index + 1]:
                    rt_tuples[id, 1] = 0  # set endlen to 0 to mark as removed

            processed += 1 # increment after finishing processing selected tuple


@cuda.jit
def blockhyp_endlen_algorithm_parallel_kernel_old(data, rt_size):
    """
    CUDA kernel implementation of the Blockhypothesis using the Endlen optimization algorithm.
    This kernel processes 1D quantized weight tensors organized in (parallel) racetracks by
    selectively flipping bits in bitgroups based on their "end length" metric. The algorithm
    uses a sliding window approach to identify and process bitgroups.
    Args:
        qweight (cuda.device_array): 
            1D array of quantized weights (+1/-1) organized in racetracks of rt_size
        rt_size (int): 
            Size of each racetrack/number of bits per racetrack
    Implementation Details:
        - Uses a sliding window of size 3 to identify bitgroups
        - For each window, calculates tuple containing:
            * tuple_index: Index of the tuple
            * endlen: Total length of the window (sum of all bitgroup lengths)
            * required_flips: Number of bits that need to be flipped in order to achieve endlen
            * starting_index: Start position of middle bitgroup
        - Sorts tuples by endlen (descending) and required flips (ascending)
        - Greedily processes tuples to maximize endlen and minimize required flips
        - Modifies qweight array in-place by flipping selected bit groups
    Note:
        Currently uses fixed-size local arrays (64 elements) due to CUDA/Numba limitations
        In the future, the rt_size parameter should be used to define the array size
    """

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


def blockhyp_endlen_algorithm_parallel_old(data, rt_size):
    """
    Initializes CUDA context to run the Endlen algorithm on GPU (old version).
    This function implements the Blockhypothesis using the Endlen algorithm 
    in parallel on GPU using CUDA kernels. It processes data stored in racetracks.
    Args:
        data (torch.Tensor): 
            Input tensor to be encoded. Must be a 1D tensor
        rt_size (int): 
            Size of each racetrack/number of bits per racetrack
    Returns:
        None. The input tensor is modified in-place.
    Notes:
        - Input data should contain only -1s and 1s
        - If the input data tensor is 2D or 3D, it should be reshaped to 1D before calling this function
        - The function requires numba.cuda to be properly configured
        - This is the old version of the algorithm
    """

    # Calculate number of racetracks
    rt = max(math.ceil(data.shape[0]/rt_size), 1)
    print(f"rt_shape: {rt}x{rt_size}")
    
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
    blockhyp_endlen_algorithm_parallel_kernel_old[blocks_per_grid_x, threads_per_block_x](data_gpu, rt_size)

    # Synchronize results from all threads first before copying any data from Device to Host
    cuda.synchronize()

    # Copy data back to CPU memory, and convert to PyTorch tensor
    data.copy_(torch.from_numpy(data_gpu.copy_to_host()))


def blockhyp_endlen_algorithm_cpu_old(data, rt_size):
    """
    Applies block hypothesis-based end length algorithm on CPU to optimize racetrack memory.
    This algorithm processes data in racetracks by identifying and flipping bit groups to maximize
    the length of identical bits at the ends of sliding windows, while minimizing the number of
    required bit flips.
    Args:
        data (numpy.ndarray): 
            1D Input array containing the binary data (-1s and 1s) to be optimized
        rt_size (int): 
            Size of each racetrack/number of bits per racetrack
    Returns:
        None. The input data array is modified in-place
    Notes:
        - Input data should contain only -1s and 1s
        - If the input data tensor is 2D or 3D, it should be reshaped to 1D before calling this function
        - Modifies data in-place by flipping selected bitgroups
        - Original version of the algorithm - unparallelized (marked as 'old')
    Algorithm steps:
        - Divides input data into racetracks of specified size
        - For each racetrack:
            * Uses sliding window of size 3 to identify bitgroups
            * Records tuple information (index, total length, required flips, middle group start)
            * Sorts tuples by endlen (descending) and required flips (ascending)
            * Greedily selects and flips optimal bitgroups
    """
    
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


### Local testing ###
def load_data_from_file(filename):
  
  with open(filename, 'r') as f:
    data_str = f.read()

  # Use eval to convert the string representation to a nested list
  data_list = eval(data_str)

  # Convert the nested list to a NumPy array for efficiency
  data = np.array(data_list)

  return data.tolist()


if __name__ == '__main__':

    # writing to file 

    rt_size = 64

    for layer in range(1,5):

        print("==========================================")
        print(f"Layer {layer}")
        print("")

        _1flip_flag = True

        in_file1 = 'q_in/qweights_orig_'+str(layer)+'.txt'
        out_file_flip1 = "q_out/qweights_orig_"+str(layer)+"_1flip.txt"

        if _1flip_flag:
            print(in_file1)
            data = load_data_from_file(in_file1)

            # print(f"data before: {data}")

            data_initial_shape = np.shape(data)
            print(f"data initial shape: {data_initial_shape}")
         
            data = np.reshape(data, -1)
            print(f"data reshaped: {np.shape(data)}")
                                

            blockhyp_endlen_algorithm_cpu_old(data=data, rt_size=rt_size)


            data=np.reshape(data, data_initial_shape)
            print(f"data reshaped to initial shape: {np.shape(data)}")

            
            # print(f"data after: {data}")

            with open(out_file_flip1, 'w') as f:
                f.write("[")
                for sublist in data.tolist()[:-1]:
                    f.write(str(sublist) + ',\n')
                
                f.write(str(data.tolist()[-1]) + "]")


