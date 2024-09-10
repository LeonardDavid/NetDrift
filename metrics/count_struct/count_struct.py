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

    sum_all = []
    alt_len_all = []

    for i in range(len(total_ones)):
        for j in range(len(total_ones[i])):

            cnt = 0
            prev = None

            sum = 0
            alt_len = 0
            sum_arr = []
            alt_len_arr = []

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

                                curr_elem = data[i][b][c][d]
                                if curr_elem != prev:
                                    sum_arr.append(sum)
                                    sum = curr_elem
                                    alt_len += 1
                                else:
                                    sum += curr_elem
                                    alt_len_arr.append(alt_len)
                                    alt_len = 1
                                
                                prev = curr_elem

                # remove initial value of sum 0 at the beginning of the array
                sum_arr.remove(0)
                # Handle the last element
                sum_arr.append(sum)
                alt_len_arr.append(alt_len)

                # print(sum_arr)
                # print(alt_len_arr)
                # print("")

                prev = None

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

                        curr_elem = data[i][b]
                        if curr_elem != prev:
                            sum_arr.append(sum)
                            sum = curr_elem
                            alt_len += 1
                        else:
                            sum += curr_elem
                            alt_len_arr.append(alt_len)
                            alt_len = 1
                        
                        prev = curr_elem

                # remove initial value of sum 0 at the beginning of the array
                sum_arr.remove(0)
                # Handle the last element
                sum_arr.append(sum)
                alt_len_arr.append(alt_len)

                # print(sum_arr)
                # print(alt_len_arr)
                # print("")

                prev = None

                # print(positions)   
                # print("")
            else:
                continue
            
            sum_all.append(np.abs(sum_arr))
            alt_len_all.append(alt_len_arr)

    # print("")
    # print("flips: " + str(flips))
        
    return sum_all, alt_len_all


## main ##

# min_bitgroup_size = 0 # which sizes of bit groups to include in metric, set to 0 for all sizes (including groups of 1 bit)

# block_size = 12
block_size = 64
err = 0.1

# for layer in range(2,3):
for layer in range(1,5):

    # layer = 4
    if layer == 1 or layer == 2:
        array_type = "3D"
    elif layer == 3 or layer == 4:
        array_type = "1D"

    print("==========================================")
    print(f"Layer {layer}")
    print("")

    #################
    ### variables ###
    #################

    test_flag = False
    print_flag = False

    #############
    ### files ###
    #############

    if test_flag:
        in_file1 = 'q_in_test/qweights_append_'+str(layer)+'_lce.txt'
        out_file_sum = "q_out/qweights_append_"+str(layer)+"_sum.txt"
        out_file_altlen = "q_out/qweights_append_"+str(layer)+"_altlen.txt"
    else:
        in_file1 = 'q_in/qweights_orig_'+str(layer)+'.txt'
        out_file_sum = "q_out/qweights_orig_"+str(layer)+"_sum.txt"
        out_file_altlen = "q_out/qweights_orig_"+str(layer)+"_altlen.txt"


    print(in_file1)
    data = load_data_from_file(in_file1)

    total_ones, total_neg_ones = count(data, array_type)
    
    # count lengths of bit groups in each block
    sum_all, alt_len_all = count_len(array_type, data, total_ones)
    
    if print_flag:
        for sum_arr in sum_all:
            print(sum_arr)
        print("")

        for alt_len_arr in alt_len_all:
            print(alt_len_arr)
        print("")

    with open(out_file_sum, "w") as f:
        for sum_arr in sum_all:
            for sum in sum_arr:
                f.write(str(sum) + ' ')
            f.write('\n')

    with open(out_file_altlen, "w") as f:
        for alt_len_arr in alt_len_all:
            for alt_len in alt_len_arr:
                f.write(str(alt_len) + ' ')
            f.write('\n')