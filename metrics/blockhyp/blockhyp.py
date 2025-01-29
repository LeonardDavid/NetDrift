import math 

# Implements the blockhyp algorithm for a 1D tensor (e.g. a weight tensor)
# If the tensor is 2D or 3D, it should be reshaped to 1D before calling this function
# The tensor is modified in place
# 
# data: the tensor to be modified
# rt_size: the size of the racetracks
#
def blockhyp_algorithm(data, rt_size):
    
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
        while len(rt_tuples) > 0:

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