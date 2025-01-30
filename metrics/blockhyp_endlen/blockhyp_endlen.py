import math 
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import cuda
import numpy as np

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


        # print(f"rt_tuples[{i}]: {rt_tuples}")

        # Greedily select the bitgroups with the highest endlen, and the lowest number of required flips, as long as there are still available bitgroups left
        while len(rt_tuples) > 0: # TODO while rt_tuples:

            tuple_index = rt_tuples[0][0]                         # first tuple  always contains the maximum (it has been sorted previously)
            start_index_mid = rt_tuples[0][3]                     # starting index of the middle bitgroup to be flipped
            final_index_mid = start_index_mid + rt_tuples[0][2]   # start index + required bitflips

            # print(f"array_tuples[{i}][0]: {rt_tuples[0]}")
            
            # Flip the middle bitgroup
            for idx in range(start_index_mid, final_index_mid):
                # print(f"{idx}: flipped {data[idx]} -> {-data[idx]}")
                data[idx] = -data[idx]

            # Remove selected tuple after storing it, along with its neighbouring tuples
            rt_tuples = [t for t in rt_tuples if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]
            # print(f"array_tuples[{i}]: {rt_tuples}")
            # print("")
        # print("")

    # print(f"data shape: {data.shape}")
    # print(data)

# TODO optimizations
# CUDA claude optimized no diff text
@cuda.jit
def process_racetrack_cuda_claude(data, rt_size, rt_tuples, tuple_counts):
    # Get the racetrack index from thread/block ID
    rt_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # print(rt_id)

    # Initializations
    # TODO initialize rt_tuples here?
    count = 0                               
    k = 0                                   
    bitgroup = cuda.local.array(3, dtype=numba.int32)
    i_mid = cuda.local.array(3, dtype=numba.int32)
    current_sign = data[rt_id * rt_size]

    # Initialize arrays
    for idx in range(3):
        bitgroup[idx] = 0
        i_mid[idx] = 0

    # Process racetrack
    for j in range(rt_size):
        index = rt_id * rt_size + j
        
        if index < data.size and data[index] != current_sign:
            i_mid[(count+1)%3] = index
            current_sign = -current_sign
            count += 1

            if count > 2:
                if k < rt_size:  # Prevent buffer overflow
                    rt_tuples[rt_id, k, 0] = k
                    rt_tuples[rt_id, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
                    rt_tuples[rt_id, k, 2] = bitgroup[(k+1)%3]
                    rt_tuples[rt_id, k, 3] = i_mid[(k+1)%3]
                    k += 1
                bitgroup[count%3] = 0
        
        if index < data.size:
            bitgroup[count%3] += 1

    # Handle last tuple
    if k < rt_size:  # Prevent buffer overflow
        rt_tuples[rt_id, k, 0] = k
        rt_tuples[rt_id, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
        rt_tuples[rt_id, k, 2] = bitgroup[(k+1)%3]
        rt_tuples[rt_id, k, 3] = i_mid[(k+1)%3]
        tuple_counts[rt_id] = k + 1     # only needed if returning rt_tuples

    # TODO cuda sort rt_tuples here?

    # TODO modify data in place here? -> if yes, copy data back from gpu to cpu


# TODO optimizations
def blockhyp_endlen_algorithm_cuda_claude(data, rt_size):

    print("")

    # Calculate number of racetracks
    rt = max(math.ceil(data.shape[0]/rt_size), 1)
    print(f"rt_shape: {rt}x{rt_size}")
    
    # Allocate memory for tuples and counts
    rt_tuples = np.zeros((rt, rt_size, 4), dtype=np.int32)  # maximum number of tuples is rt_size, if data contains alternating 1s and -1s
    tuple_counts = np.zeros(rt, dtype=np.int32)
    
    # Copy data to GPU
    data_gpu = cuda.to_device(data.detach().cpu().numpy())
    rt_tuples_gpu = cuda.to_device(rt_tuples)
    tuple_counts_gpu = cuda.to_device(tuple_counts)

    device = cuda.get_current_device()                      
    # Get the maximum number of threads per block for the current GPU, usually 1024
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    
    # Define block and grid structures
    threads_per_block_x = min(rt, max_threads_per_block)                            # although 3D arrangement is used, 1024x1x1 is preferred since it uses MAX_THREADS_PER_BLOCK
    blocks_per_grid_x = (rt + (threads_per_block_x - 1)) // threads_per_block_x     # Maximum grid size is (2^31-1)x65535x65535, so a tensor of size larger than (2^31-1)*1024 would be needed to exceed the grid size in x-dimension, therefore y- and z-dimensions are 1
    
    print("")
    print(f"blocks_per_grid: {blocks_per_grid_x}, threads_per_block: {threads_per_block_x}")
    print("")

    # Launch kernel
    process_racetrack_cuda_claude[blocks_per_grid_x, threads_per_block_x](data_gpu, rt_size, rt_tuples_gpu, tuple_counts_gpu)

    print("kernel done")
    print("")

    # Copy results back
    rt_tuples = rt_tuples_gpu.copy_to_host()
    tuple_counts = tuple_counts_gpu.copy_to_host()
    # data = data_gpu.copy_to_host() # TODO unless data is modified in place in the kernel, this is not needed

    # print(f"rt_tuples: {rt_tuples}")
    # print("")
    # print(f"tuple_counts: {tuple_counts}")
    # print("")
    
    # Process results for each racetrack
    for i in range(rt):

        # Get valid tuples for this racetrack
        valid_tuples = rt_tuples[i, :tuple_counts[i]]

        # print(f"valid_tuples{i}: {valid_tuples}")
        # print("")
        
        # Sort tuples by endlen descending, then flips ascending
        tuples_list = [tuple(x) for x in valid_tuples]
        tuples_list.sort(key=lambda x: (-x[1], x[2]))

        # print(f"tuples_list[{i}]: {tuples_list}")
        
        # Process tuples
        while tuples_list:
            tuple_index = tuples_list[0][0]
            start_index_mid = tuples_list[0][3]
            final_index_mid = start_index_mid + tuples_list[0][2]
            
            # Flip bits
            for idx in range(start_index_mid, final_index_mid):
                # print(f"{idx}: flipped {data[idx]} -> {-data[idx]}")
                data[idx] = -data[idx]
                
            # Remove processed tuples
            tuples_list = [t for t in tuples_list if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]
            # print(f"array_tuples[{i}]: {tuples_list}")
            # print("")

        # print("")

    # print(f"data shape: {data.shape}")
    # print(data)
    print("flipping done")



# # CUDA claude optimized no diff text
# @cuda.jit
# def process_racetrack_cuda_claude(data, rt_size, rt_tuples, tuple_counts):
#     # Get the racetrack index from thread/block ID
#     rt_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

#     # print(rt_id)

#     # Initializations
#     count = 0                               
#     k = 0                                   
#     bitgroup = cuda.local.array(3, dtype=numba.int32)
#     i_mid = cuda.local.array(3, dtype=numba.int32)
#     current_sign = data[rt_id * rt_size]

#     # Initialize arrays
#     for idx in range(3):
#         bitgroup[idx] = 0
#         i_mid[idx] = 0

#     # Process racetrack
#     for j in range(rt_size):
#         index = rt_id * rt_size + j
        
#         if index < data.size and data[index] != current_sign:
#             i_mid[(count+1)%3] = index
#             current_sign = -current_sign
#             count += 1

#             if count > 2:
#                 if k < rt_size:  # Prevent buffer overflow
#                     rt_tuples[rt_id, k, 0] = k
#                     rt_tuples[rt_id, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#                     rt_tuples[rt_id, k, 2] = bitgroup[(k+1)%3]
#                     rt_tuples[rt_id, k, 3] = i_mid[(k+1)%3]
#                     k += 1
#                 bitgroup[count%3] = 0
        
#         if index < data.size:
#             bitgroup[count%3] += 1

#     # Handle last tuple
#     if k < rt_size:  # Prevent buffer overflow
#         rt_tuples[rt_id, k, 0] = k
#         rt_tuples[rt_id, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#         rt_tuples[rt_id, k, 2] = bitgroup[(k+1)%3]
#         rt_tuples[rt_id, k, 3] = i_mid[(k+1)%3]
#         tuple_counts[rt_id] = k + 1


# def blockhyp_endlen_algorithm_cuda_claude(data, rt_size):

#     # print("")

#     # Calculate number of racetracks
#     rt = max(math.ceil(data.shape[0]/rt_size), 1)
#     # print(f"rt_shape: {rt}x{rt_size}")
    
#     # Allocate memory for tuples and counts
#     rt_tuples = np.zeros((rt, rt_size, 4), dtype=np.int32)
#     tuple_counts = np.zeros(rt, dtype=np.int32)
    
#     # Copy data to GPU
#     data_gpu = cuda.to_device(data.detach().cpu().numpy())
#     rt_tuples_gpu = cuda.to_device(rt_tuples)
#     tuple_counts_gpu = cuda.to_device(tuple_counts)


#     # Get the maximum number of threads per block for the current GPU
#     device = cuda.get_current_device()
#     max_threads_per_block = device.MAX_THREADS_PER_BLOCK    # usually 1024
#     # max_threads_per_block = 64
#     # max_grid_size_x = device.MAX_GRID_DIM_X
#     # max_grid_size_y = device.MAX_GRID_DIM_Y
#     # max_grid_size_z = device.MAX_GRID_DIM_Z
#     # print(f"Maximum threads per block: {max_threads_per_block}")
#     # print(f"Maximum grid size: ({max_grid_size_x}, {max_grid_size_y}, {max_grid_size_z})")
    
#     threads_per_block = min(rt, max_threads_per_block)                      # although 3D arrangement is used, 1024x1x1 is preferred since it uses MAX_THREADS_PER_BLOCK
#     blocks_per_grid = (rt + (threads_per_block - 1)) // threads_per_block   # Maximum grid size is (2^31-1)x65535x65535, so a tensor of size larger than (2^31-1)*1024 would be needed to exceed the grid size in x-dimension, therefore y- and z-dimensions are 1
    
#     # print("")
#     # print(f"blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}")
#     # print("")
    

#     # # Calculate the 3D grid arrangement
#     # blocks_per_grid_x = min(blocks_per_grid, max_grid_size_x)
#     # blocks_per_grid_y = min((blocks_per_grid + blocks_per_grid_x - 1) // blocks_per_grid_x, max_grid_size_y)
#     # blocks_per_grid_z = min((blocks_per_grid + blocks_per_grid_x * blocks_per_grid_y - 1) // (blocks_per_grid_x * blocks_per_grid_y), max_grid_size_z)

#     # print(f"3D grid arrangement: ({blocks_per_grid_x}, {blocks_per_grid_y}, {blocks_per_grid_z})")


#     # # Calculate the 3D thread arrangement within a block
#     # threads_per_block_x = min(threads_per_block, max_threads_per_block)
#     # threads_per_block_y = min((threads_per_block + threads_per_block_x - 1) // threads_per_block_x, max_threads_per_block // threads_per_block_x)
#     # threads_per_block_z = min((threads_per_block + threads_per_block_x * threads_per_block_y - 1) // (threads_per_block_x * threads_per_block_y), max_threads_per_block // (threads_per_block_x * threads_per_block_y))

#     # print(f"3D thread arrangement: ({threads_per_block_x}, {threads_per_block_y}, {threads_per_block_z})")
#     # print("")

#     # Launch kernel
#     # process_racetrack_cuda_claude[rt, 1](data_gpu, rt_size, rt_tuples_gpu, tuple_counts_gpu)
#     process_racetrack_cuda_claude[blocks_per_grid, threads_per_block](data_gpu, rt_size, rt_tuples_gpu, tuple_counts_gpu)

#     print("kernel done")
#     # print("")

#     # Copy results back
#     rt_tuples = rt_tuples_gpu.copy_to_host()
#     tuple_counts = tuple_counts_gpu.copy_to_host()
#     # data = data_gpu.copy_to_host() # TODO REMOVE??

#     # print(f"rt_tuples: {rt_tuples}")
#     # print("")
#     # print(f"tuple_counts: {tuple_counts}")
#     # print("")
    
#     # Process results for each racetrack
#     for i in range(rt):

        

#         # Get valid tuples for this racetrack
#         valid_tuples = rt_tuples[i, :tuple_counts[i]]

#         # print(f"valid_tuples{i}: {valid_tuples}")
#         # print("")
        
#         # Sort tuples by endlen descending, then flips ascending
#         tuples_list = [tuple(x) for x in valid_tuples]
#         tuples_list.sort(key=lambda x: (-x[1], x[2]))

#         # print(f"tuples_list[{i}]: {tuples_list}")
        
#         # Process tuples
#         while tuples_list:
#             tuple_index = tuples_list[0][0]
#             start_index_mid = tuples_list[0][3]
#             final_index_mid = start_index_mid + tuples_list[0][2]
            
#             # Flip bits
#             for idx in range(start_index_mid, final_index_mid):
#                 # print(f"{idx}: flipped {data[idx]} -> {-data[idx]}")
#                 data[idx] = -data[idx]
                
#             # Remove processed tuples
#             tuples_list = [t for t in tuples_list if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]
#             # print(f"array_tuples[{i}]: {tuples_list}")
#             # print("")

#         # print("")

#     # print(f"data shape: {data.shape}")
#     # print(data)
#     print("flipping done")

#     # return data #TODO REMOVE??


# # CUDA claude original
# @cuda.jit
# def process_racetrack_cuda_claude(data, rt_size, rt_tuples, tuple_counts):
#     # Get the racetrack index from thread/block ID
#     i = cuda.blockIdx.x

#     # Initializations
#     count = 0                               
#     k = 0                                   
#     bitgroup = cuda.local.array(3, dtype=numba.int32)
#     i_mid = cuda.local.array(3, dtype=numba.int32)
#     current_sign = data[i * rt_size]

#     # Initialize arrays
#     for idx in range(3):
#         bitgroup[idx] = 0
#         i_mid[idx] = 0

#     # Process racetrack
#     for j in range(rt_size):
#         index = i * rt_size + j
        
#         if index < data.size and data[index] != current_sign:
#             i_mid[(count+1)%3] = index
#             current_sign = -current_sign
#             count += 1

#             if count > 2:
#                 if k < rt_size:  # Prevent buffer overflow
#                     rt_tuples[i, k, 0] = k
#                     rt_tuples[i, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#                     rt_tuples[i, k, 2] = bitgroup[(k+1)%3]
#                     rt_tuples[i, k, 3] = i_mid[(k+1)%3]
#                     k += 1
#                 bitgroup[count%3] = 0
        
#         if index < data.size:
#             bitgroup[count%3] += 1

#     # Handle last tuple
#     if k < rt_size:  # Prevent buffer overflow
#         rt_tuples[i, k, 0] = k
#         rt_tuples[i, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#         rt_tuples[i, k, 2] = bitgroup[(k+1)%3]
#         rt_tuples[i, k, 3] = i_mid[(k+1)%3]
#         tuple_counts[i] = k + 1


# def blockhyp_endlen_algorithm_cuda_claude(data, rt_size):
#     # Calculate number of racetracks
#     rt = max(math.ceil(data.shape[0]/rt_size), 1)
    
#     # Allocate memory for tuples and counts
#     rt_tuples = np.zeros((rt, rt_size, 4), dtype=np.int32)
#     tuple_counts = np.zeros(rt, dtype=np.int32)
    
#     # Copy data to GPU
#     data_gpu = cuda.to_device(data.detach().cpu().numpy())
#     rt_tuples_gpu = cuda.to_device(rt_tuples)
#     tuple_counts_gpu = cuda.to_device(tuple_counts)
    
#     # Launch kernel
#     process_racetrack_cuda_claude[rt, 1](data_gpu, rt_size, rt_tuples_gpu, tuple_counts_gpu)
    
#     # Copy results back
#     rt_tuples = rt_tuples_gpu.copy_to_host()
#     tuple_counts = tuple_counts_gpu.copy_to_host()
#     data = data_gpu.copy_to_host()
    
#     # Process results for each racetrack
#     for i in range(rt):
#         # Get valid tuples for this racetrack
#         valid_tuples = rt_tuples[i, :tuple_counts[i]]
        
#         # Sort tuples by endlen descending, then flips ascending
#         tuples_list = [tuple(x) for x in valid_tuples]
#         tuples_list.sort(key=lambda x: (-x[1], x[2]))
        
#         # Process tuples
#         while tuples_list:
#             tuple_index = tuples_list[0][0]
#             start_index_mid = tuples_list[0][3]
#             final_index_mid = start_index_mid + tuples_list[0][2]
            
#             # Flip bits
#             for idx in range(start_index_mid, final_index_mid):
#                 data[idx] = -data[idx]
                
#             # Remove processed tuples
#             tuples_list = [t for t in tuples_list if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]
    
#     return data



# # CUDA 4o
# @cuda.jit
# def process_racetrack_cuda_4o(data, rt_size, rt_tuples, rt):
#     i = cuda.grid(1)
#     if i < rt:
#         count = 0
#         k = 0
#         bitgroup = cuda.local.array(3, dtype=np.int32)
#         i_mid = cuda.local.array(3, dtype=np.int32)
#         current_sign = data[i * rt_size]

#         for j in range(rt_size):
#             index = i * rt_size + j
#             if data[index] != current_sign:
#                 i_mid[(count+1)%3] = index
#                 current_sign = -current_sign
#                 count += 1
#                 if count > 2:
#                     rt_tuples[i, k, 0] = k
#                     rt_tuples[i, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#                     rt_tuples[i, k, 2] = bitgroup[(k+1)%3]
#                     rt_tuples[i, k, 3] = i_mid[(k+1)%3]
#                     k += 1
#                     bitgroup[count%3] = 0
#             bitgroup[count%3] += 1

#         rt_tuples[i, k, 0] = k
#         rt_tuples[i, k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
#         rt_tuples[i, k, 2] = bitgroup[(k+1)%3]
#         rt_tuples[i, k, 3] = i_mid[(k+1)%3]


# def blockhyp_endlen_algorithm_cuda_4o(data, rt_size):
    
#     rt = max(math.ceil(data.shape[0] / rt_size), 1)
#     rt_tuples = np.zeros((rt, rt_size, 4), dtype=np.int32)

#     d_data = cuda.to_device(data.detach().cpu().numpy())
#     d_rt_tuples = cuda.to_device(rt_tuples)

#     threads_per_block = 128
#     blocks_per_grid = (rt + (threads_per_block - 1)) // threads_per_block

#     process_racetrack_cuda_4o[blocks_per_grid, threads_per_block](d_data, rt_size, d_rt_tuples, rt)

#     d_rt_tuples.copy_to_host(rt_tuples)

#     for i in range(rt):
#         tuples = rt_tuples[i]
#         tuples = sorted(tuples, key=lambda x: (-x[1], x[2]))
#         while len(tuples) > 0 and tuples[0][1] > 0:
#             tuple_index = tuples[0][0]
#             start_index_mid = tuples[0][3]
#             final_index_mid = start_index_mid + tuples[0][2]
#             for idx in range(start_index_mid, final_index_mid):
#                 data[idx] = -data[idx]
#             tuples = [t for t in tuples if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]

#     return data