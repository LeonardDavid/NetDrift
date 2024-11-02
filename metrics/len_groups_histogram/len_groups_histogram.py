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
    
    length_groups = []
    nr_groups = []
    avg_len_groups = []

    for row in data:

        if arr_type == "3D":
            nr_elem = len(row) * 9 # 64 kernels of 3x3 elements

            count = 0
            arr_size = max(int(nr_elem/block_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            ngroups = np.zeros(arr_size)
            lgroups = np.zeros(arr_size)
            algroups = np.zeros(arr_size)

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

            ngroups = np.zeros(arr_size)
            lgroups = np.zeros(arr_size)
            algroups = np.zeros(arr_size)

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

        nr_groups.append(ngroups)
        length_groups.append(lgroups)
        avg_len_groups.append(algroups)

        # print(total_ones)
        # print(total_neg_ones)
        
    return total_ones, total_neg_ones, nr_groups, length_groups, avg_len_groups


block_size = 64
err = 0.1

min_bitgroup_size = 1 # which sizes of bit groups to include in metric, set to 0 for all sizes (including groups of 1 bit)

# Assuming your file is named "data.txt" and is in the same directory
# index_offset = np.loadtxt("q_index_offset/qweights_shift1_2_ind_off.txt")

# for layer in range(1,5):
for layer in range(2,3):
    print("")
    
    layer = 1
    if layer == 1 or layer == 2:
        array_type = "3D"
    elif layer == 3 or layer == 4:
        array_type = "1D"

    # file = "q/qweights_shift1_"+str(layer)+"_init.txt" # acc: 28.48%, total_shifts: 423
    # file = "q_index_offset/qweights_shift1_"+str(layer)+"_init.txt" # acc: 29.59%, total_shifts: 3652

    # file = 'q_in/qweights_append_'+str(layer)+'_lce.txt'
    file = 'q_in/qweights_append_'+str(layer)+'.txt'
    print(file)

    data = load_data_from_file(file)

    total_ones, total_neg_ones, nr_groups, length_groups, avg_len_groups = count(data, array_type)

    hist_amount = np.zeros(block_size+1)

    # total_ratios = np.divide(total_ones, total_neg_ones)
    # total_ratios = np.around(total_ratios, decimals=2)


    # print(array_type)
    # print(total_ones)
    # print(total_neg_ones)
    # print(total_ratios)

    # # out_ratio_init = "q/qweights_shift1_"+str(layer)+"_ratios_init.txt"
    # out_ratio_init = "q_index_offset/qweights_shift1_"+str(layer)+"_ratios_init.txt"
    # with open(out_ratio_init, "w") as f:
    #     # f.write("[")
    #     for line in total_ratios:
    #         # f.write("[")
    #         for value in line:
    #             f.write(str(value) + " ")
    #         f.write("\n")
    #         # f.write("]\n")
    #     # f.write("]")

    flips = 0

    for i in range(len(total_ones)):
        for j in range(len(total_ones[i])):
            cnt = 0
            # print(total_ones[i][j])
            # if i==0 and j==0:
            if i>=0 and j>=0:
                # print(str(i) + "-" + str(j) + ": " + str(total_ones[i][j]) + " / " + str(total_neg_ones[i][j]) + " = " + str(total_ones[i][j]/total_neg_ones[i][j]))
                # print(data[i])
                # print("")
                
                # lower = 0.75
                # upper = 3
                # ratio = total_ones[i][j]/total_neg_ones[i][j]
                ratio = 1.0

                # if ratio < lower or ratio >= upper:
                if ratio >= 0.0:
                    # print(str(i) + "-" + str(j) + ": " + str(total_ones[i][j]) + " / " + str(total_neg_ones[i][j]) + " = " + str(ratio))
                    # print("index_offset: " + str(index_offset[i][j]))
                    # print(data[i])

                    current_element = None
                    current_length = 0

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
                                            if current_length > min_bitgroup_size: # which bit group sizes to exclude
                                                nr_groups[i][j] += 1
                                                length_groups[i][j] += current_length
                                                hist_amount[current_length] += 1
                                            current_element = data[i][b][c][d]
                                            current_length = 1

                        current_element = None

                        # Handle the last element
                        if current_length > min_bitgroup_size: # which bit group sizes to exclude
                            nr_groups[i][j] += 1
                            length_groups[i][j] += current_length # this array will always have only elements = block_size, unless we exclude groups of bits size 1 or 2...
                            hist_amount[current_length] += 1

                        if nr_groups[i][j] != 0: # can happen if there are no groups in this bigger than the specified min_bitgroup_size
                            avg_len_groups[i][j] = length_groups[i][j]/nr_groups[i][j]
                        else:
                            avg_len_groups[i][j] = 0

                        print(f"This block contains: {nr_groups[i][j]} groups (> {min_bitgroup_size}), total length: {length_groups[i][j]}, average length: {avg_len_groups[i][j]}")


                        # print(positions)
                        
                        # ratio_new = ratio

                        # if index_offset[i][j] < 0.0:
                        #     while ratio_new <= lower or ratio_new >= upper:
                        #         for pos in positions:
                        #             # print(data[i][pos[0]][pos[1]][pos[2]])
                        #                 if ratio_new <= lower:
                        #                     if data[i][pos[0]][pos[1]][pos[2]] == -1.0:
                        #                         data[i][pos[0]][pos[1]][pos[2]] = 1.0
                        #                         total_neg_ones[i][j] -= 1
                        #                         total_ones[i][j] += 1
                        #                         ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                         print("flipped @ " + str(pos) + " from -1 to 1 => " + str(ratio_new))
                        #                         flips += 1
                        #                 elif ratio_new >= upper:
                        #                     if data[i][pos[0]][pos[1]][pos[2]] == 1.0:
                        #                         data[i][pos[0]][pos[1]][pos[2]] = -1.0
                        #                         total_ones[i][j] -= 1
                        #                         total_neg_ones[i][j] += 1
                        #                         ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                         print("flipped @ " + str(pos) + " from 1 to -1 => " + str(ratio_new))
                        #                         flips += 1
                        #                 else:
                        #                     continue
                        # elif index_offset[i][j] > 0.0:
                        #     while ratio_new <= lower or ratio_new >= upper:
                        #         for pos in reversed(positions):
                        #             # print(data[i][pos[0]][pos[1]][pos[2]])
                        #                 if ratio_new <= lower:
                        #                     if data[i][pos[0]][pos[1]][pos[2]] == -1.0:
                        #                         data[i][pos[0]][pos[1]][pos[2]] = 1.0
                        #                         total_neg_ones[i][j] -= 1
                        #                         total_ones[i][j] += 1
                        #                         ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                         print("flipped @ " + str(pos) + " from -1 to 1 => " + str(ratio_new))
                        #                         flips += 1
                        #                 elif ratio_new >= upper:
                        #                     if data[i][pos[0]][pos[1]][pos[2]] == 1.0:
                        #                         data[i][pos[0]][pos[1]][pos[2]] = -1.0
                        #                         total_ones[i][j] -= 1
                        #                         total_neg_ones[i][j] += 1
                        #                         ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                         print("flipped @ " + str(pos) + " from 1 to -1 => " + str(ratio_new))
                        #                         flips += 1
                        #                 else:
                        #                     continue         
                        print("")

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
                                    if current_length > min_bitgroup_size: # which bit group sizes to exclude
                                        nr_groups[i][j] += 1
                                        length_groups[i][j] += current_length
                                        hist_amount[current_length] += 1
                                    current_element = data[i][b]
                                    current_length = 1

                        current_element = None

                        # Handle the last element
                        if current_length > min_bitgroup_size: # which bit group sizes to exclude
                            nr_groups[i][j] += 1
                            length_groups[i][j] += current_length # this array will always have only elements = block_size, unless we exclude groups of bits size 1 or 2...
                            hist_amount[current_length] += 1

                        if nr_groups[i][j] != 0: # can happen if there are no groups in this bigger than the specified min_bitgroup_size
                            avg_len_groups[i][j] = length_groups[i][j]/nr_groups[i][j]
                        else:
                            avg_len_groups[i][j] = 0

                        print(f"This block contains: {nr_groups[i][j]} groups (> {min_bitgroup_size}), total length: {length_groups[i][j]}, average length: {avg_len_groups[i][j]}")

                        
                        # print(positions)

                        # ones = total_ones[i][j]
                        # neg_ones = total_neg_ones[i][j]
                        
                        # ratio_new = ratio

                        # if index_offset[i][j] != 0.0:
                        #     while ratio_new <= lower or ratio_new >= upper:
                        #         for pos in positions:
                        #             # print(data[i][pos[0]][pos[1]][pos[2]])
                        #             if ratio_new <= lower:
                        #                 if data[i][pos] == -1.0:
                        #                     data[i][pos] = 1.0
                        #                     total_neg_ones[i][j] -= 1
                        #                     total_ones[i][j] += 1
                        #                     ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                     print("flipped @ " + str(pos) + " from -1 to 1 => " + str(ratio_new))
                        #                     flips += 1
                        #             elif ratio_new >= upper:
                        #                 if data[i][pos] == 1.0:
                        #                     data[i][pos] = -1.0
                        #                     total_ones[i][j] -= 1
                        #                     total_neg_ones[i][j] += 1
                        #                     ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                        #                     print("flipped @ " + str(pos) + " from 1 to -1 => " + str(ratio_new))
                        #                     flips += 1
                        #             else:
                        #                 continue
                                                
                        print("")
                    else:
                        continue

        print("")
    print("flips: " + str(flips))

    # print("")
    # for index, nr in enumerate(hist_amount):
    #     print(f"{index}: {nr}")
    # print(hist_amount)

    out_len = "q_out/qweights_append_"+str(layer)+"_len_groups.txt"
    with open(out_len, "w") as f:
        # f.write("[")
        for line in length_groups:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")

    out_nr = "q_out/qweights_append_"+str(layer)+"_nr_groups.txt"
    with open(out_nr, "w") as f:
        # f.write("[")
        for line in nr_groups:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")

    avg_len_groups = np.around(avg_len_groups, decimals=2)
    out_avg = "q_out/qweights_append_"+str(layer)+"_avg_len_groups.txt"
    with open(out_avg, "w") as f:
        # f.write("[")
        for line in avg_len_groups:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")

    out_hist = "q_out/qweights_append_"+str(layer)+"_hist_amount.txt"
    with open(out_hist, "w") as f:
        # f.write("[")
        for value in hist_amount:
            f.write(str(value) + " ")
        # f.write("\n")
        # f.write("]\n")
    # f.write("]")


