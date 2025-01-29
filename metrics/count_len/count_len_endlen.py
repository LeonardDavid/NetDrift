import numpy as np
import math
import time

def load_data_from_file(filename):
  """
  Loads a 3D tensor from a file with the specified format.

  Args:
      filename: The path to the file containing the tensor data.

  Returns:
      A list representing the 3D tensor.
  """
  # Open the file in read mode
  with open(filename, 'r') as f:
    # Read the entire content of the file
    data_str = f.read()

  # Use eval to convert the string representation to a nested list
  # (Caution: eval can be insecure, use with caution on untrusted data)
  data_list = eval(data_str)

  # Convert the nested list to a NumPy array for efficiency
  data = np.array(data_list)

  return data.tolist()


def count_len(data, rt_shape):

    # Initialize the variables to calculate the lengths of bit groups
    current_length = 0
    current_element = None
    
    # Initialize the array to store the lengths of bit groups for each racetrack
    block_gr = []

    # For each racetrack...
    for i in range(rt_shape[0]):
        # Initialize/reset the array to store the lengths of bit groups for the current racetrack
        len_gr = []
        
        # ...of rt_size = rt_shape[1]
        for j in range(rt_shape[1]):

            # iterate through every weight
            data_index = i * rt_shape[1] + j
            # print(data[data_index])

            # Calculate the length of the bitgroup
            if current_element == None:
                current_element = data[data_index]
                current_length = 1

            # If the current data element is the same as the current element, increment the current length
            elif data[data_index] == current_element:
                current_length += 1

            # If the current data element is different from the current element, a different bitgroup started
            # store bitgroup length and reset current element and current length for the next bitgroup
            else:
                len_gr.append(int(current_length*current_element))
                # print(int(current_length*current_element))

                current_element = data[data_index]
                current_length = 1

            # print(str(data[data_index]) + " " + str(data_index-1) + "/" + str(rt_shape[1]) + "=" + str(int((data_index-1) / rt_shape[1])) + "==" + str(i))
            # if data_index % rt_shape[1] == 0:
            #     print("")

        # Handle the last element
        len_gr.append(int(current_length*current_element))
        # print(int(current_length*current_element))

        # Reset current element for next racetrack
        current_element = None
        
        # Store the lengths of bit groups for the current racetrack
        block_gr.append(len_gr)
                
    return block_gr


def create_endlen_tuples(block_gr):
    
    arr_tuples = []

    # for each racetrack
    for i in range(len(block_gr)):
        line_tuples = []

        # for each bitgroup
        for j in range(1, len(block_gr[i])-1): # ignore edges since they are missing one neighbour
            # using a sliding window of size 3, sum up lengths of potential bitgroups, if the middle bitgroup would be flipped 
            # [optional: alternate the signs]    
            endlen = abs(block_gr[i][j-1])+abs(block_gr[i][j])+abs(block_gr[i][j+1])
            
            # create tuple in format: (index, endlen, #flips)
            line_tuples.append((j, endlen, abs(block_gr[i][j])))

        arr_tuples.append(line_tuples)

    # Sort the list of tuples (index, endlen, flips) by endlen descending, then by flips ascending

    for line in arr_tuples:
        line.sort(key=lambda x: (-x[1], x[2]))
    
    return arr_tuples


def find_pos(arr_tuples, block_gr):

    pos_bitlen = []
    position = 0
    
    # for each racetrack
    for i in range(len(arr_tuples)):
        pos = []

        # greedily select the bitgroups with the highest endlen, and the lowest number of flips, as long as there are still available bitgroups left
        while len(arr_tuples[i]) > 0:
            # print(f"({arr_tuples[i][0][0]}, {arr_tuples[i][0][1]}, {arr_tuples[i][0][2]})")

            # first tuple  always contains the maximum (it has been sorted previously)
            index = arr_tuples[i][0][0]
            # store index from tuple for now, expand to all pos_bitlen later
            pos.append(index) 

            # remove selected tuple after storing it, along with its neighbouring tuples
            arr_tuples[i] = [t for t in arr_tuples[i] if t[0] not in [index-1, index, index+1]]
    
        # store all original indices that need to be flipped in the original tensor, according to the indices of the tuples chosen previously
        pos_set = set(pos)  # Convert pos to a set for O(1) lookups
        for j, value in enumerate(block_gr[i]):
            position += abs(value)

            # if the bitgroup index is in the set of selected bitgroups, add all indices of the original weights to the list of positions to flip
            if j in pos_set:
                pos_bitlen.extend(position - p_back - 1 for p_back in range(abs(value)))

    return pos_bitlen


def flip_bits(data, positions):

    if len(positions) > 0:

        positions = set(positions)  # Convert positions to a set for O(1) lookups

        for b in range(len(data)):

            if b in positions:
                # data[b] *= -1    # causes error when training, because of in-place operation *= applied directly to tensor
                data[b] = -data[b]

                # print(f"{b}: flipped {-data[b]} -> {data[b]}")
    else:
        print("No positions to flip")
        return


def apply_1flip(data, rt_shape):
    
    # rt_size = rt_shape[1]
    # total_flips = 0

    start_time = time.time()
    # count lengths of bit groups in each block
    block_groups = count_len(data=data, rt_shape=rt_shape)
    print(f"count_len_new function took {time.time() - start_time:.4f} seconds")

    # print(f"block_groups:")
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")

    start_time = time.time()
    array_tuples = create_endlen_tuples(block_gr=block_groups)
    print(f"create_endlen_tuples function took {time.time() - start_time:.4f} seconds")

    # print("sorted array_tuples #2:")
    # for tuples in array_tuples:
    #     print(tuples)
    # print("")

    start_time = time.time()
    pos1 = find_pos(arr_tuples=array_tuples, block_gr=block_groups)
    print(f"find_pos function took {time.time() - start_time:.4f} seconds")

    # total_flips += len(pos1)
    # print(pos1)
    # print(len(pos1))
    # print("")

    start_time = time.time()
    # flip positions of ones found previously
    flip_bits(data=data, positions=pos1)
    print(f"flip_bits_new function took {time.time() - start_time:.4f} seconds")
    # print(data)
    # print("")

    
    # with open(out_file_flip1, "w") as f:
    #     f.write("[")

    #     # Write the list of integers to the file
    #     for integer in data[:-1]:
    #         f.write(str(integer) + ',\n')

    #     f.write(str(data[-1]) + "]")

    # # count lengths of bit groups in each block
    # # # ONLY IF SUBSEQUENT apply_2flip IS NEEDED (or for debugging)
    # block_groups = count_len(data=data, rt_shape=rt_shape)
    
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")



# Implements the blockhyp algorithm for a 1D tensor (e.g. a weight tensor)
# If the tensor is 2D or 3D, it should be reshaped to 1D before calling this function
# The tensor is modified in place
# 
# data: the tensor to be modified
# rt_size: the size of the racetracks
#
def count_len_create_endlen_tuples(data, rt_size):
    
    # Specify the number of racetracks based on the size of the tensor and the racetrack size
    rt = max(math.ceil(data.shape[0]/rt_size), 1)
    print(f"rt_shape: {rt}x{rt_size}")

    tuples = []

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

        tuples.append(rt_tuples)

    return tuples


def select_flip_bits(data, array_tuples):

    # for each racetrack
    for i in range(len(array_tuples)):
        
        print("")
        print(f"array_tuples[{i}]: {array_tuples[i]}")        

        # greedily select the bitgroups with the highest endlen, and the lowest number of required flips, as long as there are still available bitgroups left
        while len(array_tuples[i]) > 0:

            tuple_index = array_tuples[i][0][0]                         # first tuple  always contains the maximum (it has been sorted previously)
            start_index_mid = array_tuples[i][0][3]                     # starting index of the middle bitgroup to be flipped
            final_index_mid = start_index_mid + array_tuples[i][0][2]   # start index + required bitflips

            print(f"array_tuples[{i}][0]: {array_tuples[i][0]}")
            
            # flip the middle bitgroup
            for idx in range(start_index_mid, final_index_mid):
                print(f"{idx}: flipped {data[idx]} -> {-data[idx]}")
                data[idx] = -data[idx]

            # remove selected tuple after storing it, along with its neighbouring tuples
            array_tuples[i] = [t for t in array_tuples[i] if t[0] not in [tuple_index-1, tuple_index, tuple_index+1]]
            print(f"array_tuples[{i}]: {array_tuples[i]}")


if __name__ == '__main__':
    ## main ##

    # min_bitgroup_size = 0 # which sizes of bit groups to include in metric, set to 0 for all sizes (including groups of 1 bit)

    kernel_size = 3

    ### important variable
    rt_size = 12
    # rt_size = 64
    
    err = 0.1

    gbb = 0.03  # global bitflip budget
    lbb = 1   # local bitflip budget

    total_elems = [0, 576, 36864, 6422528, 20480] #FMNIST

    # total_elems = [0, 401408, 262144, 5120] # MNIST9696
    # total_elems = [0, 401408, 5120] # MNIST9418
    # total_elems = [0, 7840] # MNIST8562


    for layer in range(1,2):

        layer = 2
        # layer = 4
        if layer == 1 or layer == 2 or layer==0:
            array_type = "3D"
        elif layer == 3 or layer == 4:
            array_type = "1D"

        # array_type = "1D"

        total_elem = total_elems[layer]

        print("==========================================")
        print(f"Layer {layer}")
        print("")

        total_flips = 0


        #################
        ### variables ###
        #################

        test_flag = True
        print_flag = True

        _1flip_flag = True
        _1flip_flag_budget = False
        _1flip_flag_col = False
        _1flip_flag_offset = False

        _1eflip_flag = False
        _2flip_flag = False

        
        if _1flip_flag_col:
            bitlen1 = "endlen1_col"
            bitlen2 = "endlen2_col"
            bitlen3 = "endlen3_col"
        else:
            bitlen1 = "endlen1"
            bitlen2 = "endlen2"
            bitlen3 = "endlen3"

        #############
        ### files ###
        #############

        if test_flag:
            in_file1 = 'q_in_test/qweights_append_'+str(layer)+'_lce.txt'
            out_file_flip1 = "q_out_test/qweights_append_"+str(layer)+"_1flip_"+str(bitlen1)+".txt"
            out_file_flip1e = "q_out_test/qweights_append_"+str(layer)+"_1flip_"+str(bitlen1)+".txt"
            out_file_flip2 = "q_out_test/qweights_append_"+str(layer)+"_2flip_"+str(bitlen2)+".txt"
            out_file_flip2e = "q_out_test/qweights_append_"+str(layer)+"_2flip_"+str(bitlen2)+".txt"
            out_file_flip3 = "q_out_test/qweights_append_"+str(layer)+"_3flip_"+str(bitlen3)+".txt"
            out_file_flip3e = "q_out_test/qweights_append_"+str(layer)+"_3flip_"+str(bitlen3)+".txt"
        else:
            in_file1 = 'q_in/qweights_orig_'+str(layer)+'.txt'
            out_file_flip1 = "q_out/qweights_orig_"+str(layer)+"_1flip_"+str(bitlen1)+".txt"
            out_file_flip1e = "q_out/qweights_orig_"+str(layer)+"_1flip_"+str(bitlen1)+".txt"
            out_file_flip2 = "q_out/qweights_orig_"+str(layer)+"_2flip_"+str(bitlen2)+".txt"
            out_file_flip2e = "q_out/qweights_orig_"+str(layer)+"_2flip_"+str(bitlen2)+".txt"
            out_file_flip3 = "q_out/qweights_orig_"+str(layer)+"_3flip_"+str(bitlen3)+".txt"
            out_file_flip3e = "q_out/qweights_orig_"+str(layer)+"_3flip_"+str(bitlen3)+".txt"


        ###################
        ### 1. flip new ###
        ###################

        if _1flip_flag:
            print(in_file1)
            data = load_data_from_file(in_file1)

            data_initial_shape = np.shape(data)
            print(f"data initial shape: {data_initial_shape}")

            if array_type == "3D":               
                data = np.reshape(data, -1)
                print(f"data reshaped: {np.shape(data)}")
                                
            print(f"data before: {data}")
            
            # rt_shape = (max(math.ceil(data.shape[0]/rt_size),1), rt_size)
            # print(rt_shape)

            array_tuples = count_len_create_endlen_tuples(data=data, rt_size=rt_size)

            select_flip_bits(data=data, array_tuples=array_tuples)

            print(f"data after: {data}")

            if array_type == "3D":                
                # print("")
                data=np.reshape(data, data_initial_shape)
                print(f"data reshaped to initial shape: {np.shape(data)}")
                # print(data)

            
        elif _1flip_flag_offset:
            print(in_file1)
            data = load_data_from_file(in_file1)

            glb_bb = 0.01
            loc_bb = 1.0

            ind_off = np.array([[0,-1,-2],[0,1,0]])

            apply_1flip_ind_off(array_type=array_type, rt_size=rt_size, data=data, index_offset=ind_off, global_bitflip_budget=glb_bb, local_bitflip_budget=loc_bb)

        # exit()

        ###################
        ### 1. flip old ###
        ###################

        if _1flip_flag:
            print(in_file1)
            data = load_data_from_file(in_file1)

            # column-wise blockhyp
            if _1flip_flag_col:
                
                # for conv layers: reshape to 1D-type array and transpose (cols->rows), 
                # to execute same steps for blockhyp but for the cols
                if layer == 1 or layer == 2 or layer==0:
                    array_type = "1D"
                    
                    data_initial_shape = np.shape(data)
                    data = np.reshape(data,(data_initial_shape[0],-1)).T

                    print(f"{data_initial_shape} -> Reshape + Transpose -> {np.shape(data)}")

                # for linear layers: already 1D-type array, therefore only transpose (cols->rows), 
                # to execute same steps for blockhyp but for the cols
                elif layer == 3 or layer == 4:
                    array_type = "1D"

                    data_initial_shape = np.shape(data)
                    data = np.transpose(data)

                    print(f"{data_initial_shape} -> Transpose -> {np.shape(data)}")

                            
            # total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
            # total_elem = len(total_ones) * len(total_ones[0]) * rt_size
            
            if test_flag:
                total_ones, total_neg_ones = count_old(data=data, arr_type=array_type, rt_size=rt_size)
                rt_shape = np.shape(total_ones)
                print(np.shape(total_ones))
                
                if array_type == "3D":
                    nr_elem = len(data[0]) * 9 # conv kernels of 3x3, 5x5, 7x7 elements
                elif array_type == "1D":
                    nr_elem = len(data[0]) 
                else:
                    print("Invalid array type")
                    exit()
                rt_shape = (len(data), max(math.ceil(nr_elem/rt_size),1))
                print(rt_shape)
            else:
                if array_type == "3D":
                    nr_elem = len(data[0]) * 9 # conv kernels of 3x3, 5x5, 7x7 elements
                elif array_type == "1D":
                    nr_elem = len(data[0]) 
                else:
                    print("Invalid array type")
                    exit()
            
                rt_shape = (len(data), max(math.ceil(nr_elem/rt_size),1))
            

            # count lengths of bit groups in each block
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=rt_shape, rt_size=rt_size)
            
            if print_flag:
                print("block_groups:")
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            endlen_groups = sum_endlen_old(block_gr=block_groups)

            if print_flag:
                print("endlen_groups:")
                for sum_len in endlen_groups:
                    print(sum_len)
                print("")

            array_tuples = create_tuples_old(block_gr=block_groups, endlen_gr=endlen_groups)
            array_tuples = sort_tuples_old(arr_tuples=array_tuples)

            if print_flag:
                print("sorted array_tuples:")
                for tuples in array_tuples:
                    print(tuples)
                print("")

            if _1flip_flag_budget:
                print(total_elem)
                pos1 = find_with_bitflip_budget(arr_tuples=array_tuples, block_gr=block_groups, rt_size=rt_size, total_elem=total_elem, global_bitflip_budget=gbb, local_bitflip_budget=lbb)
            else:
                pos1 = find_old(arr_tuples=array_tuples, block_gr=block_groups)

            total_flips += len(pos1)
            if print_flag:
                print(pos1)
            print("")
            print(len(pos1))
            print("")

            # flip positions of ones found previously
            flip_bits_old(data=data, positions=pos1, rt_shape=rt_shape, array_type=array_type)

            # column-wise blcokhyp
            if _1flip_flag_col:

                # transpose back (rows->cols), and reshape to original 3D shape
                if layer == 1 or layer == 2 or layer == 0:
                    array_type = "3D"

                    data_before_shape = np.shape(data)
                    data = np.reshape(data.T, data_initial_shape)
                    
                    total_ones, total_neg_ones = count_old(data=data, arr_type=array_type, rt_size=rt_size)
                    
                    print(f"{data_before_shape} -> Transpose + Reshape -> {np.shape(data)}")


                    with open(out_file_flip1, 'w') as f:
                        f.write('[')

                        for i in range(data.shape[0]):
                            f.write('[')
                            for j in range(data.shape[1]):
                                f.write('[')
                                for k in range(data.shape[2]):
                                    # Convert the subarray to a comma-separated string
                                    subarray_str = ','.join(map(str, data[i, j, k]))
                                    # Write the subarray string to the file
                                    f.write('[' + subarray_str + ']')
                                    if k < data.shape[2] - 1:
                                        f.write(',')
                                f.write(']')
                                if j < data.shape[1] - 1:
                                    f.write(',')
                            f.write(']')
                            if i < data.shape[0] - 1:
                                f.write(',\n')

                        f.write(']')

                # only transpose back to original shape (rows->cols)
                elif layer == 3 or layer == 4:
                    array_type = "1D"

                    data_before_shape = np.shape(data)
                    data = np.transpose(data)

                    total_ones, total_neg_ones = count_old(data=data, arr_type=array_type, rt_size=rt_size)
                    
                    print(f"{data_before_shape} -> Transpose -> {np.shape(data)}")


                    with open(out_file_flip1, 'w') as f:
                        f.write('[')

                        for i in range(data.shape[0]):
                            # Convert the subarray to a comma-separated string
                            subarray_str = ','.join(map(str, data[i]))
                            # Write the subarray string to the file
                            f.write('[' + subarray_str + ']')
                            if i < data.shape[0] - 1:
                                f.write(',\n')

                        f.write(']')

            else:
                with open(out_file_flip1, "w") as f:
                    f.write("[")

                    # Write the list of integers to the file
                    for integer in data[:-1]:
                        f.write(str(integer) + ',\n')

                    f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=rt_shape, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

        elif _1flip_flag_offset:
            print(in_file1)
            data = load_data_from_file(in_file1)

            glb_bb = 0.01
            loc_bb = 1.0

            ind_off = np.array([[0,-1,-2],[0,1,0]])

            apply_1flip_ind_off(array_type=array_type, rt_size=rt_size, data=data, index_offset=ind_off, global_bitflip_budget=glb_bb, local_bitflip_budget=loc_bb)

        ###################
        ### clean edges ###
        ###################

        if _1eflip_flag:
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=rt_shape, rt_size=rt_size)

            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            pos_edges = find_edges(bitlen1, block_groups, n_left1, n_right1)

            total_flips += len(pos_edges)
            if print_flag:
                print(pos_edges)
            print(len(pos_edges))
            print("")


            flip_bits_old(data=data, positions=pos_edges, rt_shape=rt_shape, array_type=array_type)
            

            with open(out_file_flip1e, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=rt_shape, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

        # # # 
        # (optional, if wanting to continue improving flipped tensor) update block_groups after bit flips, since fewer but bigger groups are created in the process
        # # #


        ###############
        ### 2. flip ###
        ###############

        if _2flip_flag:
            in_file2 = out_file_flip1e
            print(in_file2)
            data = load_data_from_file(in_file2)

            total_ones, total_neg_ones = count_old(data=data, arr_type=array_type, rt_size=rt_size)
            
            # count lengths of bit groups in each block
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
            pos2 = find_old(block_groups, n_left2, n_right2)

            total_flips += len(pos2)
            if print_flag:
                print(pos2)
            print(len(pos2))
            print("")

            
            # flip positions of ones found previously
            flip_bits_old(data=data, positions=pos2, rt_shape=total_ones, array_type=array_type)


            with open(out_file_flip2, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len_old(array_type=array_type, data=data, rt_shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")




def count_old(data, arr_type, rt_size):

    total_ones = []
    total_neg_ones = []

    for row in data:

        if arr_type == "3D":
            nr_elem = len(row) * 9 # conv kernels of 3x3, 5x5, 7x7 elements

            count = 0
            arr_size = max(math.ceil(nr_elem/rt_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            for element in row:
                for item in element:
                    for weight in item:
                        count += 1
                        if weight == 1:
                            count_ones[int((count-1) / rt_size)] += 1
                        elif weight == -1:
                            count_neg_ones[int((count-1) / rt_size)] += 1
            
        elif arr_type == "1D":
            nr_elem = len(row)

            count = 0
            arr_size = max(math.ceil(nr_elem/rt_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            for item in row:
                count += 1
                if item == 1:
                    count_ones[int((count-1) / rt_size)] += 1
                elif item == -1:
                    count_neg_ones[int((count-1) / rt_size)] += 1

        # print(count_ones)
        # print(count_neg_ones)
            
        total_ones.append(count_ones)
        total_neg_ones.append(count_neg_ones)

        # print(total_ones)
        # print(total_neg_ones)
        
    return total_ones, total_neg_ones


def count_len_old(array_type, data, shape, rt_size):

    block_gr = []

    for i in range(len(shape)):
        for j in range(len(shape[i])):

            cnt = 0
            current_element = None
            current_length = 0
            len_gr = []

            if array_type == "3D":

                positions = []

                for b in range(len(data[i])):
                    # print(element)
                    for c in range(len(data[i][b])):
                        # print(item)
                        for d in range(len(data[i][b][c])):
                            # print(weight)
                            cnt += 1
                            # print(data[i][b][c][d])
                            # print(str(data[i][b][c][d]) + " " + str(cnt-1) + "/" + str(rt_size) + "=" + str(int((cnt-1) / rt_size)) + "==" + str(j))
                            # if cnt % rt_size == 0:
                            #     print("")
                            if int((cnt-1) / rt_size) == j:
                                positions.append((b,c,d)) 
                                # print(str(cnt) + ": " + "(" + str(b) + "," + str(c) + "," + str(d) + "): " + str(data[i][b][c][d]))

                                if current_element == None:
                                    current_element = data[i][b][c][d]
                                    current_length = 1
                                elif data[i][b][c][d] == current_element:
                                    current_length += 1
                                else:
                                    # if current_length > min_bitgroup_size: # which bit group sizes to exclude
                                    #     # instructions

                                    # print(int(current_length*current_element))
                                    len_gr.append(int(current_length*current_element))

                                    current_element = data[i][b][c][d]
                                    current_length = 1


                # Handle the last element

                # print(int(current_length*current_element))
                len_gr.append(int(current_length*current_element)) 

                # if current_length > min_bitgroup_size: # which bit group sizes to exclude
                #     # instructions

                # print(f"This block contains: {nr_groups[i][j]} groups (> {min_bitgroup_size}), total length: {length_groups[i][j]}, average length: {avg_len_groups[i][j]}")

                current_element = None

                # print(positions)   
                # print("")

            elif array_type == "1D":
                positions = []

                for b in range(len(data[i])):
                    # print(element)
                    cnt += 1
                    # print(data[i][b])
                    # print(str(data[i][b]) + " " + str(cnt-1) + "/" + str(rt_size) + "=" + str(int((cnt-1) / rt_size)) + "==" + str(j))
                    # if cnt % rt_size == 0:
                    #     print("")
                    if int((cnt-1) / rt_size) == j:
                        positions.append((b)) 
                        # print(str(cnt) + ": " + "(" + str(b) + "): " + str(data[i][b]))

                        if current_element == None:
                                current_element = data[i][b]
                                current_length = 1
                        elif data[i][b] == current_element:
                            current_length += 1
                        else:
                            # if current_length > min_bitgroup_size: # which bit group sizes to exclude
                            #     # instructions

                            # print(int(current_length*current_element))
                            len_gr.append(int(current_length*current_element))

                            current_element = data[i][b]
                            current_length = 1

                # Handle the last element
                # print(int(current_length*current_element))
                len_gr.append(int(current_length*current_element))

                # if current_length > min_bitgroup_size: # which bit group sizes to exclude
                #     # instructions

                # print(f"This block contains: {nr_groups[i][j]} groups (> {min_bitgroup_size}), total length: {length_groups[i][j]}, average length: {avg_len_groups[i][j]}")

                current_element = None
                
                # print(positions)                                                
                # print("")
            else:
                continue
            
            block_gr.append(len_gr)
    # print("")
    # print("flips: " + str(flips))
        
    return block_gr


def create_tuples_ind_off(block_gr, endlen_gr, index_offset):
    # format: (index, endlen, flips, index_offset)

    arr_tuples = []

    for i in range(len(endlen_gr)):
        line_tuples = []
        for j in range(len(endlen_gr[i])):
            line_tuples.append((j+1, endlen_gr[i][j], abs(block_gr[i][j+1])))
        arr_tuples.append(line_tuples)
    
    return arr_tuples


def sort_tuples_ind_off(arr_tuples):
    # Sorts a list of tuples (index, endlen, flips, index_offset) 
    # by endlen descending, then by index_offset ascending

    for line in arr_tuples:
        line.sort(key=lambda x: (-x[1], x[2]))

    return arr_tuples


def matrix2tuples(ind_off):
    # Flatten the matrix to a 1D array
    flattened_m = ind_off.flatten()

    # Create an array of indices for each element
    indices = np.indices(ind_off.shape)

    tuples = [(abs(val), i, j) for val, i, j in zip(flattened_m, indices[0].flatten(), indices[1].flatten())]

    return tuples


def sum_endlen_old(block_gr):
    # using a sliding window of size 3, sum up lengths form block_gr, alternating the signs

    endlen_gr = []

    for i in range(len(block_gr)):
        sum_gr = []

        for j in range(1, len(block_gr[i])-1): # ignore edges since they are missing one neighbour
            sum_gr.append(abs(block_gr[i][j-1])+abs(block_gr[i][j])+abs(block_gr[i][j+1]))

        endlen_gr.append(sum_gr)
    
    return endlen_gr


def create_tuples_old(block_gr, endlen_gr):
    # format: (index, endlen, flips)

    arr_tuples = []

    for i in range(len(endlen_gr)):
        line_tuples = []
        for j in range(len(endlen_gr[i])):
            line_tuples.append((j+1, endlen_gr[i][j], abs(block_gr[i][j+1])))
        arr_tuples.append(line_tuples)
    
    return arr_tuples


def sort_tuples_old(arr_tuples):
    # Sorts a list of tuples (index, endlen, flips) by endlen descending, then by flips ascending

    for line in arr_tuples:
        line.sort(key=lambda x: (-x[1], x[2]))

    return arr_tuples


def find_old(arr_tuples, block_gr):

    pos_bitlen = []
    position = 0
    
    for i in range(len(arr_tuples)):
        # print(arr_tuples[i])
        # for j in range(len(arr_tuples[i])):
            # print(f"({arr_tuples[i][j][0]}, {arr_tuples[i][j][1]}, {arr_tuples[i][j][2]})")

        pos = []

        while len(arr_tuples[i]) > 0:
            # print(f"({arr_tuples[i][0][0]}, {arr_tuples[i][0][1]}, {arr_tuples[i][0][2]})")
            
            # first tuple  always contains the maximum (it has been sorted previously)
            index = arr_tuples[i][0][0]
            pos.append(index) # store index from tuple for now, expand to all pos_bitlen later
            arr_tuples[i].remove(arr_tuples[i][0]) # remove tuple after storing it

            # find and remove neighbouring tuples
            found = list(filter(lambda t: t[0] in [index-1, index+1], arr_tuples[i]))
            for tuple in found:
                arr_tuples[i].remove(tuple)
            # print(f"results: {results}")

        # store all indices that need to be flipped, according to the chosen indexes previously
        pos.sort()
        for j in range(len(block_gr[i])):
            position += abs(block_gr[i][j])
            for p in range(len(pos)):
                if j == pos[p]:
                    p_back = 0
                    while p_back <= abs(block_gr[i][j])-1:
                        pos_bitlen.append(position-p_back)
                        p_back += 1
        
        # print("")
                
    pos_bitlen.sort() # sort array so that flip() function can work properly
    return pos_bitlen


def find_with_bitflip_budget(arr_tuples, block_gr, rt_size, total_elem, global_bitflip_budget, local_bitflip_budget):

    pos_bitlen = []
    position = 0
    nr_flips_global = 0

    i = 0
    
    while i < len(arr_tuples) and nr_flips_global < total_elem * global_bitflip_budget:
        # print(arr_tuples[i])
        # for j in range(len(arr_tuples[i])):
            # print(f"({arr_tuples[i][j][0]}, {arr_tuples[i][j][1]}, {arr_tuples[i][j][2]})")

        pos = []
        nr_flips_local = 0

        while len(arr_tuples[i]) > 0 and nr_flips_local < rt_size * local_bitflip_budget:
            # print(f"({arr_tuples[i][0][0]}, {arr_tuples[i][0][1]}, {arr_tuples[i][0][2]})")
            
            # first tuple  always contains the maximum (it has been sorted previously)
            index = arr_tuples[i][0][0]
            nr_flips_local += arr_tuples[i][0][2]

            pos.append(index) # store index from tuple for now, expand to all pos_bitlen later
            arr_tuples[i].remove(arr_tuples[i][0]) # remove tuple after storing it

            # find and remove neighbouring tuples
            found = list(filter(lambda t: t[0] in [index-1, index+1], arr_tuples[i]))
            for tuple in found:
                arr_tuples[i].remove(tuple)
            # print(f"results: {results}")

            
        # print(nr_flips_local)
        nr_flips_global += nr_flips_local

        # store all indices that need to be flipped, according to the chosen indexes previously
        pos.sort()
        for j in range(len(block_gr[i])):
            position += abs(block_gr[i][j])
            for p in range(len(pos)):
                if j == pos[p]:
                    p_back = 0
                    while p_back <= abs(block_gr[i][j])-1:
                        pos_bitlen.append(position-p_back)
                        p_back += 1
        
        i += 1
        # print("")
        
    # print(nr_flips_global)
                
    pos_bitlen.sort() # sort array so that flip() function can work properly
    return pos_bitlen


def find_with_bitflip_budget_ind_off(arr_tuples, block_gr, ind_off_tuples, rt_shape, rt_size, total_elem, global_bitflip_budget, local_bitflip_budget):

    pos_bitlen = []
    position = 0
    nr_flips_global = 0
    
    ## global: 0.05
    ## local: 0.2

    # global_bitflip_budget = 0.1
    # local_bitflip_budget = 0.1

    # print(global_bitflip_budget)
    # print(local_bitflip_budget)
    # print(total_elem)
    print("")

    k = 0
    while k < len(ind_off_tuples) and nr_flips_global < total_elem * global_bitflip_budget:
        # print(ind_off_tuples[k])
        # print(f"({ind_off_tuples[k][1]}, {ind_off_tuples[k][2]})")

        index_i = ind_off_tuples[k][1] * rt_shape[1] + ind_off_tuples[k][2]

        # print(f"[{i},{j}]")
        
        # print(arr_tuples[index_i])
        # for j in range(len(arr_tuples[i])):
            # print(f"({arr_tuples[i][j][0]}, {arr_tuples[i][j][1]}, {arr_tuples[i][j][2]})")

        pos = []
        nr_flips_local = 0

        while len(arr_tuples[index_i]) > 0 and nr_flips_local < rt_size * local_bitflip_budget:
            # print(arr_tuples[index_i])
            # print(f"({arr_tuples[i][0][0]}, {arr_tuples[i][0][1]}, {arr_tuples[i][0][2]})")
            
            # first tuple always contains the maximum (it has been sorted previously)
            index = arr_tuples[index_i][0][0]
            nr_flips_local += arr_tuples[index_i][0][2]

            pos.append(index) # store index from tuple for now, expand to all pos_bitlen later
            arr_tuples[index_i].remove(arr_tuples[index_i][0]) # remove tuple after storing it

            # find and remove neighbouring tuples
            found = list(filter(lambda t: t[0] in [index-1, index+1], arr_tuples[index_i]))
            for tuple in found:
                arr_tuples[index_i].remove(tuple)
            # print(f"results: {results}")
        
        # print(nr_flips_local)
        nr_flips_global += nr_flips_local

        # print(pos)
        # store all indices that need to be flipped, according to the chosen indexes previously
        pos.sort()
        for j in range(len(block_gr[index_i])):
            position += abs(block_gr[index_i][j])
            for p in range(len(pos)):
                if j == pos[p]:
                    p_back = 0
                    while p_back <= abs(block_gr[index_i][j])-1:
                        pos_bitlen.append(position-p_back)
                        p_back += 1
        
        k+=1
    
    print(nr_flips_global)

    pos_bitlen.sort() # sort array so that flip() function can work properly
    return pos_bitlen


def flip_bits_old(data, positions, shape, array_type):

    if len(positions) > 0:

        use_ind = 0 # use index at this position for comparison
        cnt = 0

        for i in range(len(shape)):

            if array_type == "3D":

                for b in range(len(data[i])):
                    for c in range(len(data[i][b])):
                        for d in range(len(data[i][b][c])):
                            cnt += 1
                            if cnt == positions[use_ind]:
                                if use_ind < len(positions)-1:
                                    use_ind += 1
                                # data[i][b][c][d] *= -1    # causes error when training, because of in-place operation *= applied directly to tensor
                                data[i][b][c][d] = -data[i][b][c][d]

                            # print(data[i][b][c][d])
                            # print(str(data[i][b][c][d]) + " " + str(cnt-1) + "/" + str(rt_size) + "=" + str(int((cnt-1) / rt_size)) + "==" + str(j))
                            # if cnt % rt_size == 0:
                            #     print("")
                            # if int((cnt-1) / rt_size) == j:
                            #     print(str(cnt) + ": " + "(" + str(b) + "," + str(c) + "," + str(d) + "): " + str(data[i][b][c][d]))

                # print("")

            elif array_type == "1D":

                for b in range(len(data[i])):
                    cnt += 1
                    if cnt == positions[use_ind]:
                        if use_ind < len(positions)-1:
                            use_ind += 1
                        # data[i][b] *= -1    # causes error when training, because of in-place operation *= applied directly to tensor
                        data[i][b] = -data[i][b]

                    # print(data[i][b])
                    # print(str(data[i][b]) + " " + str(cnt-1) + "/" + str(rt_size) + "=" + str(int((cnt-1) / rt_size)) + "==" + str(j))
                    # if cnt % rt_size == 0:
                    #     print("")
                    # if int((cnt-1) / rt_size) == j:
                    #     print(str(cnt) + ": " + "(" + str(b) + "): " + str(data[i][b]))
                                                
                # print("")
            else:
                continue
        # print("")        


def apply_1flip_old(array_type, rt_size, data):
    
    total_flips = 0
    
    # print(array_type)
    # print(data)
    # print("")

    start_time = time.time()
    total_ones, total_neg_ones = count_old(data=data, arr_type=array_type, rt_size=rt_size)
    print(f"count function took {time.time() - start_time:.4f} seconds")

    # print("")
    # print(total_ones)
    # print(f"Shape of total_ones: {np.shape(total_ones)}")
    # print(f"Shape of data: {np.shape(data)}")
    # print("")

    # if array_type == "3D":
    #     nr_elem = len(data[0]) * 9 # conv kernels of 3x3, 5x5, 7x7 elements
    # elif array_type == "1D":
    #     nr_elem = len(data[0]) 
    # else:
    #     print("Invalid array type")
    #     return   
    
    # data_initial_shape = np.shape(data)
    # print(f"data initial shape: {data_initial_shape}")
    
    # rt_shape = (len(data), max(math.ceil(nr_elem/rt_size),1))
    # print(f"rt_shape: {rt_shape}")

    start_time = time.time()
    # count lengths of bit groups in each block
    block_groups = count_len_old(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
    print(f"count_len function took {time.time() - start_time:.4f} seconds")

    # print("block_groups:")
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")

    start_time = time.time()
    endlen_groups = sum_endlen_old(block_gr=block_groups)
    print(f"sum_endlen function took {time.time() - start_time:.4f} seconds")

    # print("endlen_groups:")
    # for sum_len in endlen_groups:
    #     print(sum_len)
    # print("")

    start_time = time.time()
    array_tuples = create_tuples_old(block_gr=block_groups, endlen_gr=endlen_groups)
    print(f"create_tuples function took {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    array_tuples = sort_tuples_old(arr_tuples=array_tuples)
    print(f"sort_tuples function took {time.time() - start_time:.4f} seconds")

    # print("sorted array_tuples:")
    # for tuples in array_tuples:
    #     print(tuples)
    # print("")

    start_time = time.time()
    pos1 = find_old(arr_tuples=array_tuples, block_gr=block_groups)
    print(f"find function took {time.time() - start_time:.4f} seconds")

    total_flips += len(pos1)
    # print(pos1)
    # print(len(pos1))
    # print("")

    start_time = time.time()
    # flip positions of ones found previously
    flip_bits_old(data=data, positions=pos1, shape=total_ones, array_type=array_type)
    print(f"flip_bits function took {time.time() - start_time:.4f} seconds")
    # print(data)
    # print("")

    # with open(out_file_flip1, "w") as f:
    #     f.write("[")

    #     # Write the list of integers to the file
    #     for integer in data[:-1]:
    #         f.write(str(integer) + ',\n')

    #     f.write(str(data[-1]) + "]")

    # # count lengths of bit groups in each block
    # # # ONLY IF SUBSEQUENT apply_2flip IS NEEDED (or for debugging)
    # block_groups = count_len(array_type=array_type, data=data, rt_shape=total_ones, rt_size=rt_size)
    
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")


def apply_1flip_ind_off(array_type, rt_size, data, index_offset, global_bitflip_budget, local_bitflip_budget):
    total_flips = 0
    total_elem = len(index_offset)*len(index_offset[0])*rt_size
    ind_off_shape = (len(index_offset), len(index_offset[0]))


    print(global_bitflip_budget)
    print(local_bitflip_budget)
    # print("")
    # print(index_offset)
    # print(f"ind_off_avg: {np.mean(np.abs(index_offset))}")
    # print("")

    # print(array_type)
    # print(data)
    # print("")

    # count lengths of bit groups in each block
    block_groups = count_len_old(array_type=array_type, data=data, rt_shape=index_offset, rt_size=rt_size)

    # print("block_groups:")
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")

    endlen_groups = sum_endlen_old(block_gr=block_groups)

    # print("endlen_groups:")
    # for sum_len in endlen_groups:
    #     print(sum_len)
    # print("")

    array_tuples = create_tuples_old(block_gr=block_groups, endlen_gr=endlen_groups)
    array_tuples = sort_tuples_old(arr_tuples=array_tuples)

    # print("sorted array_tuples:")
    # print(len(array_tuples))
    # for tuples in array_tuples:
    #     print(tuples)
    #     print("")
    # print("")


    ind_off_tuples = matrix2tuples(ind_off=index_offset)
    
    # # Find blocks in which the most error shifts happened, 
    # # to prioritize creation of larger bitgroups there (endlen bitflip in blocks with bigger numbers)
    
    ind_off_tuples.sort(key=lambda x: x[0], reverse=True)
    
    # # Find blocks in which the fewest error shifts happened (except 0), 
    # # to prioritize creation of larger bitgroups there (endlen bitflip in blocks with smaller numbers - except 0)
    
    # ind_off_tuples.sort(key=lambda x: x[0])
    ind_off_tuples = [t for t in ind_off_tuples if t[0] != 0] # remove index offsets that are 0, to not flip bits there
    # ind_off_tuples = [t for t in ind_off_tuples if t[0]%2 != 0] # remove index offsets that are even (includes 0), to not flip bits there
    # ind_off_tuples = [t for t in ind_off_tuples if t[0] != 0 and t[0]%2 == 0] # remove index offsets that are odd, to not flip bits there
    
    # print(ind_off_tuples)

    pos1 = find_with_bitflip_budget_ind_off(arr_tuples=array_tuples, block_gr=block_groups, ind_off_tuples=ind_off_tuples, rt_shape=ind_off_shape, rt_size=rt_size, total_elem=total_elem, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget)

    total_flips += len(pos1)
    # print(pos1)
    # print(len(pos1))
    # print("")

    # flip positions of ones found previously
    flip_bits_old(data=data, positions=pos1, rt_shape=index_offset, array_type=array_type)
    # print(data)
    # print("")

    # with open(out_file_flip1, "w") as f:
    #     f.write("[")

    #     # Write the list of integers to the file
    #     for integer in data[:-1]:
    #         f.write(str(integer) + ',\n')

    #     f.write(str(data[-1]) + "]")

    # # count lengths of bit groups in each block
    # # # ONLY IF SUBSEQUENT apply_2flip IS NEEDED (or for debugging)
    # block_groups = count_len(array_type=array_type, data=data, rt_shape=index_offset, rt_size=rt_size)
    
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")