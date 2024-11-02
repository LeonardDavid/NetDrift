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
    
    table_lce = []

    for row in data:

        if arr_type == "3D":
            nr_elem = len(row) * 9 # 64 kernels of 3x3 elements

            count = 0
            arr_size = max(int(nr_elem/block_size),1)
            count_ones = np.zeros(arr_size)
            count_neg_ones = np.zeros(arr_size)

            lce = np.zeros(arr_size)

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

            lce = np.zeros(arr_size)

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

        table_lce.append(lce)

        # print(total_ones)
        # print(total_neg_ones)
        
    return total_ones, total_neg_ones, table_lce


block_size = 64
err = 0.1

# Assuming your file is named "data.txt" and is in the same directory
# index_offset = np.loadtxt("q_index_offset/qweights_shift1_2_ind_off.txt")

# for layer in range(1,5):
for layer in range(1,2):
    print("")
    
    layer = 1
    if layer == 1 or layer == 2:
        array_type = "3D"
    elif layer == 3 or layer == 4:
        array_type = "1D"

    # file = 'q_in/qweights_append_'+str(layer)+'_lce.txt'
    file = 'q_in/qweights_append_'+str(layer)+'.txt'
    print(file)

    data = load_data_from_file(file)

    total_ones, total_neg_ones, table_lce = count(data, array_type)

    # total_ratios = np.divide(total_ones, total_neg_ones)
    # total_ratios = np.around(total_ratios, decimals=2)


    # print(array_type)
    # print(total_ones)
    # print(total_neg_ones)
    # print(total_ratios)

    # out_ratio_init = "q_out/qweights_shift1_"+str(layer)+"_ratios_init.txt"
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

                    longest_element = None
                    longest_length = 0
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

                                        if data[i][b][c][d] == current_element:
                                            current_length += 1
                                        else:
                                            longest_element = current_element if current_length > longest_length else longest_element
                                            longest_length = max(current_length, longest_length)
                                            current_element = data[i][b][c][d]
                                            current_length = 1

                        # Handle the last element
                        longest_element = current_element if current_length > longest_length else longest_element
                        longest_length = max(current_length, longest_length)

                        if longest_element < 0.0:
                            table_lce[i][j] = -longest_length
                        else:
                            table_lce[i][j] = longest_length

                        print(f"Longest consecutive occurrences: element {longest_element}, length {longest_length}")


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

                                if data[i][b] == current_element:
                                    current_length += 1
                                else:
                                    longest_element = current_element if current_length > longest_length else longest_element
                                    longest_length = max(current_length, longest_length)
                                    current_element = data[i][b]
                                    current_length = 1

                        # Handle the last element
                        longest_element = current_element if current_length > longest_length else longest_element
                        longest_length = max(current_length, longest_length)

                        if longest_element < 0.0:
                            table_lce[i][j] = -longest_length
                        else:
                            table_lce[i][j] = longest_length

                        print(f"Longest consecutive occurrences: element {longest_element}, length {longest_length}")

                        
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

    out_lce = "q_out/qweights_append_"+str(layer)+"_lce_table.txt"
    with open(out_lce, "w") as f:
        # f.write("[")
        for line in table_lce:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")


    # total_ratios = np.divide(total_ones, total_neg_ones)
    # total_ratios = np.around(total_ratios, decimals=2)
    
    # out_ratio_mod = "q_out/qweights_shift1_"+str(layer)+"_ratios_mod.txt"
    # with open(out_ratio_mod, "w") as f:
    #     # f.write("[")
    #     for line in total_ratios:
    #         # f.write("[")
    #         for value in line:
    #             f.write(str(value) + " ")
    #         f.write("\n")
    #         # f.write("]\n")
    #     # f.write("]")


    # out_file_mod = "q_out/qweights_shift1_"+str(layer)+"_mod.txt"
    # with open(out_file_mod, "w") as f:
    #     f.write("[")

    #     # Write the list of integers to the file
    #     for integer in data[:-1]:
    #         f.write(str(integer) + ',\n')

    #     f.write(str(data[-1]) + "]")



