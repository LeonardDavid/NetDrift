import numpy as np
import math

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


def count(data, arr_type, rt_size):

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


def count_len(array_type, data, shape, rt_size):

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


def  sum_endlen(block_gr):
    # using a sliding window of size 3, sum up lengths form block_gr, alternating the signs

    endlen_gr = []

    for i in range(len(block_gr)):
        sum_gr = []

        for j in range(1, len(block_gr[i])-1): # ignore edges since they are missing one neighbour
            sum_gr.append(abs(block_gr[i][j-1])+abs(block_gr[i][j])+abs(block_gr[i][j+1]))

        endlen_gr.append(sum_gr)
    
    return endlen_gr


def create_tuples(block_gr, endlen_gr):
    # format: (index, endlen, flips)

    arr_tuples = []

    for i in range(len(endlen_gr)):
        line_tuples = []
        for j in range(len(endlen_gr[i])):
            line_tuples.append((j+1, endlen_gr[i][j], abs(block_gr[i][j+1])))
        arr_tuples.append(line_tuples)
    
    return arr_tuples


def sort_tuples(arr_tuples):
    # Sorts a list of tuples (index, endlen, flips) by endlen descending, then by flips ascending

    for line in arr_tuples:
        line.sort(key=lambda x: (-x[1], x[2]))

    return arr_tuples


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


def find(arr_tuples, block_gr):

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


def find_with_bitflip_budget_ind_off(arr_tuples, block_gr, ind_off_tuples, shape, rt_size, total_elem, global_bitflip_budget, local_bitflip_budget):

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

        index_i = ind_off_tuples[k][1] * shape[1] + ind_off_tuples[k][2]

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


def flip_bits(data, positions, shape, array_type):

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


def apply_1flip(array_type, rt_size, data):
    
    total_flips = 0
    
    # print(array_type)
    # print(data)
    # print("")

    total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)

    # print(total_ones)
    # print("")

    # count lengths of bit groups in each block
    block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)

    # print("block_groups:")
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")

    endlen_groups = sum_endlen(block_gr=block_groups)

    # print("endlen_groups:")
    # for sum_len in endlen_groups:
    #     print(sum_len)
    # print("")

    array_tuples = create_tuples(block_gr=block_groups, endlen_gr=endlen_groups)
    array_tuples = sort_tuples(arr_tuples=array_tuples)

    # print("sorted array_tuples:")
    # for tuples in array_tuples:
    #     print(tuples)
    # print("")

    pos1 = find(arr_tuples=array_tuples, block_gr=block_groups)

    total_flips += len(pos1)
    # print(pos1)
    # print(len(pos1))
    # print("")

    # flip positions of ones found previously
    flip_bits(data=data, positions=pos1, shape=total_ones, array_type=array_type)
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
    # block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
    
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
    block_groups = count_len(array_type=array_type, data=data, shape=index_offset, rt_size=rt_size)

    # print("block_groups:")
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")

    endlen_groups = sum_endlen(block_gr=block_groups)

    # print("endlen_groups:")
    # for sum_len in endlen_groups:
    #     print(sum_len)
    # print("")

    array_tuples = create_tuples(block_gr=block_groups, endlen_gr=endlen_groups)
    array_tuples = sort_tuples(arr_tuples=array_tuples)

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

    pos1 = find_with_bitflip_budget_ind_off(arr_tuples=array_tuples, block_gr=block_groups, ind_off_tuples=ind_off_tuples, shape=ind_off_shape, rt_size=rt_size, total_elem=total_elem, global_bitflip_budget=global_bitflip_budget, local_bitflip_budget=local_bitflip_budget)

    total_flips += len(pos1)
    # print(pos1)
    # print(len(pos1))
    # print("")

    # flip positions of ones found previously
    flip_bits(data=data, positions=pos1, shape=index_offset, array_type=array_type)
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
    # block_groups = count_len(array_type=array_type, data=data, shape=index_offset, rt_size=rt_size)
    
    # for len_gr in block_groups:
    #     print(len_gr)
    # print("")




if __name__ == '__main__':
    ## main ##

    # min_bitgroup_size = 0 # which sizes of bit groups to include in metric, set to 0 for all sizes (including groups of 1 bit)

    # rt_size = 12
    rt_size = 64
    err = 0.1

    gbb = 0.03  # global bitflip budget
    lbb = 1   # local bitflip budget

    # total_elems = [0, 576, 36864, 6422528, 20480] #FMNIST
    # total_elems = [0, 401408, 262144, 5120] # MNIST9696
    # total_elems = [0, 401408, 5120] # MNIST9418
    total_elems = [0, 7840] # MNIST8562


    for layer in range(1,2):
    # for layer in range(1,3):
    # for layer in range(1,4):
    # for layer in range(1,5):

        # layer = 2
        # if layer == 1 or layer == 2 or layer==0:
        #     array_type = "3D"
        # elif layer == 3 or layer == 4:
        #     array_type = "1D"

        array_type = "1D"

        total_elem = total_elems[layer]

        print("==========================================")
        print(f"Layer {layer}")
        print("")

        total_flips = 0


        #################
        ### variables ###
        #################

        test_flag = False
        print_flag = False

        _1flip_flag = True
        _1flip_flag_budget = False
        _1flip_flag_col = False
        _1flip_flag_offset = False

        _1eflip_flag = False
        _2flip_flag = False
        _2eflip_flag = False
        _3flip_flag = False
        _3eflip_flag = False

        
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


        ###############
        ### 1. flip ###
        ###############

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

                            
            total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
            # total_elem = len(total_ones) * len(total_ones[0]) * rt_size
            

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                print("block_groups:")
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            endlen_groups = sum_endlen(block_gr=block_groups)

            if print_flag:
                print("endlen_groups:")
                for sum_len in endlen_groups:
                    print(sum_len)
                print("")

            array_tuples = create_tuples(block_gr=block_groups, endlen_gr=endlen_groups)
            array_tuples = sort_tuples(arr_tuples=array_tuples)

            if print_flag:
                print("sorted array_tuples:")
                for tuples in array_tuples:
                    print(tuples)
                print("")

            if _1flip_flag_budget:
                print(total_elem)
                pos1 = find_with_bitflip_budget(arr_tuples=array_tuples, block_gr=block_groups, rt_size=rt_size, total_elem=total_elem, global_bitflip_budget=gbb, local_bitflip_budget=lbb)
            else:
                pos1 = find(arr_tuples=array_tuples, block_gr=block_groups)

            total_flips += len(pos1)
            if print_flag:
                print(pos1)
            print("")
            print(len(pos1))
            print("")

            # flip positions of ones found previously
            flip_bits(data=data, positions=pos1, shape=total_ones, array_type=array_type)

            # column-wise blcokhyp
            if _1flip_flag_col:

                # transpose back (rows->cols), and reshape to original 3D shape
                if layer == 1 or layer == 2 or layer == 0:
                    array_type = "3D"

                    data_before_shape = np.shape(data)
                    data = np.reshape(data.T, data_initial_shape)
                    
                    total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
                    
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

                    total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
                    
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
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
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
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)

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


            flip_bits(data=data, positions=pos_edges, shape=total_ones, array_type=array_type)
            

            with open(out_file_flip1e, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
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

            total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
            
            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
            pos2 = find(block_groups, n_left2, n_right2)

            total_flips += len(pos2)
            if print_flag:
                print(pos2)
            print(len(pos2))
            print("")

            
            # flip positions of ones found previously
            flip_bits(data=data, positions=pos2, shape=total_ones, array_type=array_type)


            with open(out_file_flip2, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

        ###################
        ### clean edges ###
        ###################

        if _2eflip_flag:
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)

            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            pos_edges = find_edges(bitlen2, block_groups, n_left2, n_right2)

            total_flips += len(pos_edges)
            if print_flag:
                print(pos_edges)
            print(len(pos_edges))
            print("")


            flip_bits(data=data, positions=pos_edges, shape=total_ones, array_type=array_type)
            

            with open(out_file_flip2e, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")


        # # # 
        # (optional, if wanting to continue improving flipped tensor) update block_groups after bit flips, since fewer but bigger groups are created in the process
        # # #

        ###############
        ### 3. flip ###
        ###############

        if _3flip_flag:
            in_file3 = out_file_flip2
            print(in_file3)
            data = load_data_from_file(in_file3)

            total_ones, total_neg_ones = count(data=data, arr_type=array_type, rt_size=rt_size)
            
            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
            pos3 = find(block_groups, n_left2, n_right2)

            total_flips += len(pos3)
            if print_flag:
                print(pos3)
            print(len(pos3))
            print("")

            
            # flip positions of ones found previously
            flip_bits(data=data, positions=pos3, shape=total_ones, array_type=array_type)


            with open(out_file_flip3, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

        ###################
        ### clean edges ###
        ###################

        if _3eflip_flag:
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)

            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

            pos_edges = find_edges(bitlen3, block_groups, n_left3, n_right3)

            total_flips += len(pos_edges)
            if print_flag:
                print(pos_edges)
            print(len(pos_edges))
            print("")


            flip_bits(data=data, positions=pos_edges, shape=total_ones, array_type=array_type)
            

            with open(out_file_flip3e, "w") as f:
                f.write("[")

                # Write the list of integers to the file
                for integer in data[:-1]:
                    f.write(str(integer) + ',\n')

                f.write(str(data[-1]) + "]")

            # count lengths of bit groups in each block
            block_groups = count_len(array_type=array_type, data=data, shape=total_ones, rt_size=rt_size)
            
            if print_flag:
                for len_gr in block_groups:
                    print(len_gr)
                print("")

        print(f"Total flips: {total_flips}")
        print("")