import numpy as np

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


def count(data, arr_type):

    total_ones = []
    total_neg_ones = []

    for row in data:

        if arr_type == "3D":
            nr_elem = len(row) * 9 # 64 kernels of 3x3 elements

            count = 0
            arr_size = max(int(nr_elem/block_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            for element in row:
                for item in element:
                    for weight in item:
                        count += 1
                        if weight == 1:
                            count_ones[int((count-1) / block_size)] += 1
                        elif weight == -1:
                            count_neg_ones[int((count-1) / block_size)] += 1
            
        elif arr_type == "1D":
            nr_elem = len(row)

            count = 0
            arr_size = max(int(nr_elem/block_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            for item in row:
                count += 1
                if item == 1:
                    count_ones[int((count-1) / block_size)] += 1
                elif item == -1:
                    count_neg_ones[int((count-1) / block_size)] += 1

        # print(count_ones)
        # print(count_neg_ones)
            
        total_ones.append(count_ones)
        total_neg_ones.append(count_neg_ones)

        # print(total_ones)
        # print(total_neg_ones)
        
    return total_ones, total_neg_ones


def count_len(array_type, data, total_ones):

    block_gr = []

    for i in range(len(total_ones)):
        for j in range(len(total_ones[i])):

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
                            # print(str(data[i][b][c][d]) + " " + str(cnt-1) + "/" + str(block_size) + "=" + str(int((cnt-1) / block_size)) + "==" + str(j))
                            # if cnt % block_size == 0:
                            #     print("")
                            if int((cnt-1) / block_size) == j:
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
                    # print(str(data[i][b]) + " " + str(cnt-1) + "/" + str(block_size) + "=" + str(int((cnt-1) / block_size)) + "==" + str(j))
                    # if cnt % block_size == 0:
                    #     print("")
                    if int((cnt-1) / block_size) == j:
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


def find(bitlen, block_gr, n_left, n_right):

    # it is guaranteed to contain positions of ones surrounded by different signed groups, else the +/-1 bit would have been already in neighbouring group
    pos_bitlen = []
    pos = 0
    
    for i in range(len(block_gr)):
        for j in range(len(block_gr[i])):
            pos += abs(block_gr[i][j])
            if j > 0 and j < len(block_gr[i])-1:
                if abs(block_gr[i][j]) == bitlen:
                    # print(f"{block_gr[i][j-1]} < {block_gr[i][j]} > {block_gr[i][j+1]}")
                    if abs(block_gr[i][j-1]) >= n_left and abs(block_gr[i][j+1]) >= n_right:
                        # add `bitlen` bits to be flipped, for bitlen >= 2
                        p_back = 0
                        while p_back <= bitlen-1:
                            pos_bitlen.append(pos-p_back)
                            p_back += 1
                
    pos_bitlen.sort() # sort array so that flip() function can work properly
    return pos_bitlen


# def find_ones_edges(block_gr):

#     # it is guaranteed to contain positions of ones surrounded by different signed groups, else the +/-1 bit would have been already in neighbouring group
#     pos_edge = []
#     pos = 0
    
#     for i in range(len(block_gr)):
#         for j in range(len(block_gr[i])):
#             pos += abs(block_gr[i][j])
#             if j == 0 or j == len(block_gr[i])-1:
#                 if abs(block_gr[i][j]) == 1:
#                     pos_edge.append(pos)
#         # print("")

#     return pos_edge


def find_edges(bitlen, block_gr, n_left, n_right):

    # it is guaranteed to contain positions of ones surrounded by different signed groups, else the +/-1 bit would have been already in neighbouring group
    pos_edge = []
    pos = 0
    
    for i in range(len(block_gr)):
        for j in range(len(block_gr[i])):
            pos += abs(block_gr[i][j])
            if j == 0:
                if abs(block_gr[i][j]) == bitlen:
                    # print(f"{i}, {j}, {block_gr[i][j]} | {block_gr[i][j+1]}")
                    if abs(block_gr[i][j+1]) >= n_right:
                        # add `bitlen` bits to be flipped, for bitlen >= 2
                        p_back = 0
                        while p_back <= bitlen-1:
                            pos_edge.append(pos-p_back)
                            p_back += 1

            elif j == len(block_gr[i])-1:
                if abs(block_gr[i][j]) == bitlen:
                    # print(f"{i}, {j}, {block_gr[i][j]} | {block_gr[i][j-1]}")
                    if abs(block_gr[i][j-1]) >= n_left:
                        # add `bitlen` bits to be flipped, for bitlen >= 2
                        p_back = 0
                        while p_back <= bitlen-1:
                            pos_edge.append(pos-p_back)
                            p_back += 1
            else:
                continue            
        # print("")

    pos_edge.sort() # sort array so that flip() function can work properly
    return pos_edge



def flip_bits(positions, total_ones):

    if len(positions) > 0:

        use_ind = 0 # use index at this position for comparison
        cnt = 0

        for i in range(len(total_ones)):

            if array_type == "3D":

                for b in range(len(data[i])):
                    for c in range(len(data[i][b])):
                        for d in range(len(data[i][b][c])):
                            cnt += 1
                            if cnt == positions[use_ind]:
                                if use_ind < len(positions)-1:
                                    use_ind += 1
                                data[i][b][c][d] *= -1

                            # print(data[i][b][c][d])
                            # print(str(data[i][b][c][d]) + " " + str(cnt-1) + "/" + str(block_size) + "=" + str(int((cnt-1) / block_size)) + "==" + str(j))
                            # if cnt % block_size == 0:
                            #     print("")
                            # if int((cnt-1) / block_size) == j:
                            #     print(str(cnt) + ": " + "(" + str(b) + "," + str(c) + "," + str(d) + "): " + str(data[i][b][c][d]))

                # print("")

            elif array_type == "1D":

                for b in range(len(data[i])):
                    cnt += 1
                    if cnt == positions[use_ind]:
                        if use_ind < len(positions)-1:
                            use_ind += 1
                        data[i][b] *= -1

                    # print(data[i][b])
                    # print(str(data[i][b]) + " " + str(cnt-1) + "/" + str(block_size) + "=" + str(int((cnt-1) / block_size)) + "==" + str(j))
                    # if cnt % block_size == 0:
                    #     print("")
                    # if int((cnt-1) / block_size) == j:
                    #     print(str(cnt) + ": " + "(" + str(b) + "): " + str(data[i][b]))
                                                
                # print("")
            else:
                continue
        # print("")        


## main ##

# min_bitgroup_size = 0 # which sizes of bit groups to include in metric, set to 0 for all sizes (including groups of 1 bit)

# block_size = 12
block_size = 64
err = 0.001

# for layer in range(2,3):
for layer in range(1,9):

    # layer = 4
    if layer == 7 or layer == 8:
        array_type = "1D"
    else:
        array_type = "3D"

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
    _1eflip_flag = True
    _2flip_flag = True
    _2eflip_flag = False
    _3flip_flag = True
    _3eflip_flag = False

    bitlen1 = 1
    n_left1 = 2
    n_right1 = 2
    
    bitlen2 = 2
    n_left2 = 2
    n_right2 = 2

    bitlen3 = 3
    n_left3 = 3
    n_right3 = 3

    #############
    ### files ###
    #############

    if test_flag:
        in_file1 = 'q_in_test/qweights_append_'+str(layer)+'_lce.txt'
        out_file_flip1 = "q_out_test/qweights_append_"+str(layer)+"_1flip"+str(bitlen1)+"_"+str(n_left1)+"_"+str(n_right1)+".txt"
        out_file_flip1e = "q_out_test/qweights_append_"+str(layer)+"_1flip"+str(bitlen1)+"e_"+str(n_left1)+"_"+str(n_right1)+".txt"
        out_file_flip2 = "q_out_test/qweights_append_"+str(layer)+"_2flip"+str(bitlen2)+"_"+str(n_left2)+"_"+str(n_right2)+".txt"
        out_file_flip2e = "q_out_test/qweights_append_"+str(layer)+"_2flip"+str(bitlen2)+"e_"+str(n_left2)+"_"+str(n_right2)+".txt"
        out_file_flip3 = "q_out_test/qweights_append_"+str(layer)+"_3flip"+str(bitlen3)+"_"+str(n_left3)+"_"+str(n_right3)+".txt"
        out_file_flip3e = "q_out_test/qweights_append_"+str(layer)+"_3flip"+str(bitlen3)+"e_"+str(n_left3)+"_"+str(n_right3)+".txt"
    else:
        in_file1 = 'q_in/qweights_orig_'+str(layer)+'.txt'
        out_file_flip1 = "q_out/qweights_orig_"+str(layer)+"_1flip"+str(bitlen1)+"_"+str(n_left1)+"_"+str(n_right1)+".txt"
        out_file_flip1e = "q_out/qweights_orig_"+str(layer)+"_1flip"+str(bitlen1)+"e_"+str(n_left1)+"_"+str(n_right1)+".txt"
        out_file_flip2 = "q_out/qweights_orig_"+str(layer)+"_2flip"+str(bitlen2)+"_"+str(n_left2)+"_"+str(n_right2)+".txt"
        out_file_flip2e = "q_out/qweights_orig_"+str(layer)+"_2flip"+str(bitlen2)+"e_"+str(n_left2)+"_"+str(n_right2)+".txt"
        out_file_flip3 = "q_out/qweights_orig_"+str(layer)+"_3flip"+str(bitlen3)+"_"+str(n_left3)+"_"+str(n_right3)+".txt"
        out_file_flip3e = "q_out/qweights_orig_"+str(layer)+"_3flip"+str(bitlen3)+"e_"+str(n_left3)+"_"+str(n_right3)+".txt"


    ###############
    ### 1. flip ###
    ###############

    if _1flip_flag:
        print(in_file1)
        data = load_data_from_file(in_file1)

        total_ones, total_neg_ones = count(data, array_type)
        
        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

        # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
        pos1 = find(bitlen1, block_groups, n_left1, n_right1)

        total_flips += len(pos1)
        if print_flag:
            print(pos1)
        print(len(pos1))
        print("")

        
        # flip positions of ones found previously
        flip_bits(pos1, total_ones)

                
        with open(out_file_flip1, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

    ###################
    ### clean edges ###
    ###################

    if _1eflip_flag:
        block_groups = count_len(array_type, data, total_ones)

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


        flip_bits(pos_edges, total_ones)
        

        with open(out_file_flip1e, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
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

        total_ones, total_neg_ones = count(data, array_type)
        
        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

        # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
        pos2 = find(bitlen2, block_groups, n_left2, n_right2)

        total_flips += len(pos2)
        if print_flag:
            print(pos2)
        print(len(pos2))
        print("")

        
        # flip positions of ones found previously
        flip_bits(pos2, total_ones)


        with open(out_file_flip2, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

    ###################
    ### clean edges ###
    ###################

    if _2eflip_flag:
        block_groups = count_len(array_type, data, total_ones)

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


        flip_bits(pos_edges, total_ones)
        

        with open(out_file_flip2e, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
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

        total_ones, total_neg_ones = count(data, array_type)
        
        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

        # find and store positions of +/-1 bits surrounded by bit-groups of bigger lengths: [n_left | +/-1 | n_right], usually n_left==n_right but can be fine-tuned
        pos3 = find(bitlen3, block_groups, n_left2, n_right2)

        total_flips += len(pos3)
        if print_flag:
            print(pos3)
        print(len(pos3))
        print("")

        
        # flip positions of ones found previously
        flip_bits(pos3, total_ones)


        with open(out_file_flip3, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

    ###################
    ### clean edges ###
    ###################

    if _3eflip_flag:
        block_groups = count_len(array_type, data, total_ones)

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


        flip_bits(pos_edges, total_ones)
        

        with open(out_file_flip3e, "w") as f:
            f.write("[")

            # Write the list of integers to the file
            for integer in data[:-1]:
                f.write(str(integer) + ',\n')

            f.write(str(data[-1]) + "]")

        # count lengths of bit groups in each block
        block_groups = count_len(array_type, data, total_ones)
        
        if print_flag:
            for len_gr in block_groups:
                print(len_gr)
            print("")

    print(f"Total flips: {total_flips}")
    print("")