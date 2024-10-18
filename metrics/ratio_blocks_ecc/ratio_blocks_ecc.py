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


block_size = 64
err = 0.1


for layer in range(1,5):
# for layer in range(2,3):
    print("")
    # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_append_"+str(layer)+".txt"
    # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift1_"+str(layer)+".txt"
    # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift10_"+str(layer)+".txt"
    # file = "q_in/qweights_shift1_4_init.txt"
    file = "q_in/qweights_orig_"+str(layer)+".txt"
    print(file)

    if layer == 1 or layer == 2:
        array_type = "3D"
    elif layer == 3 or layer == 4:
        array_type = "1D" 

    data = load_data_from_file(file)

    total_ones, total_neg_ones = count(data, array_type)

    total_ratios = np.divide(total_ones, total_neg_ones)
    total_ratios = np.around(total_ratios, decimals=2)

    out_ratio_init = "q_in/qweights_shift1_4_ratios_init.txt"
    with open(out_ratio_init, "w") as f:
        # f.write("[")
        for line in total_ratios:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")

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
                
                lower = 0.9
                upper = 1.4
                ratio = total_ones[i][j]/total_neg_ones[i][j]

                if ratio < lower or ratio >= upper:
                    # print(str(i) + "-" + str(j) + ": " + str(total_ones[i][j]) + " / " + str(total_neg_ones[i][j]) + " = " + str(ratio))
                    # print(data[i])

                    if array_type == "3D":

                        positions = []

                        for b in range(len(data[i])):
                            # print(element)
                            for c in range(len(data[i][b])):
                                # print(item)
                                for d in range(len(data[i][b][c])):
                                    # print(weight)
                                    cnt += 1
                                    # mod_block.append(data[i][b][c][d])
                                    # print(data[i][b][c][d])
                                    # print(str(data[i][b][c][d]) + " " + str(cnt-1) + "/" + str(block_size) + "=" + str(int((cnt-1) / block_size)) + "==" + str(j))
                                    # if cnt % block_size == 0:
                                    #     print("")
                                    if int((cnt-1) / block_size) == j:
                                        positions.append((b,c,d)) 
                                        # print(str(cnt) + ": " + "(" + str(b) + "," + str(c) + "," + str(d) + "): " + str(data[i][b][c][d]))

                        # print(positions)
                        
                        ratio_new = ratio
                        interrupt = 0

                        while ratio_new <= lower or ratio_new >= upper or interrupt<5:
                            interrupt += 1
                            for pos in positions:
                                # print(data[i][pos[0]][pos[1]][pos[2]])
                                if ratio_new <= lower:
                                    if data[i][pos[0]][pos[1]][pos[2]] == -1.0:
                                        data[i][pos[0]][pos[1]][pos[2]] = 1.0
                                        total_neg_ones[i][j] -= 1
                                        total_ones[i][j] += 1
                                        ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                                        # neg_ones -= 1
                                        # ones += 1
                                        # ratio_new = ones/neg_ones
                                        # print("flipped @ " + str(pos) + " from -1 to 1 => " + str(ratio_new))
                                        flips += 1
                                elif ratio_new >= upper:
                                    if data[i][pos[0]][pos[1]][pos[2]] == 1.0:
                                        data[i][pos[0]][pos[1]][pos[2]] = -1.0
                                        total_ones[i][j] -= 1
                                        total_neg_ones[i][j] += 1
                                        ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                                        # ones -= 1
                                        # neg_ones += 1
                                        # ratio_new = ones/neg_ones
                                        # print("flipped @ " + str(pos) + " from 1 to -1 => " + str(ratio_new))
                                        flips += 1
                                else:
                                    continue
                            
                                                
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
                        
                        # print(positions)

                        # ones = total_ones[i][j]
                        # neg_ones = total_neg_ones[i][j]
                        
                        ratio_new = ratio
                        interrupt = 0

                        while ratio_new <= lower or ratio_new >= upper or interrupt<5:
                            interrupt += 1
                            for pos in positions:
                                # print(data[i][pos[0]][pos[1]][pos[2]])
                                if ratio_new <= lower:
                                    if data[i][pos] == -1.0:
                                        data[i][pos] = 1.0
                                        total_neg_ones[i][j] -= 1
                                        total_ones[i][j] += 1
                                        ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                                        # neg_ones -= 1
                                        # ones += 1
                                        # ratio_new = ones/neg_ones
                                        # print("flipped @ " + str(pos) + " from -1 to 1 => " + str(ratio_new))
                                        flips += 1
                                elif ratio_new >= upper:
                                    if data[i][pos] == 1.0:
                                        data[i][pos] = -1.0
                                        total_ones[i][j] -= 1
                                        total_neg_ones[i][j] += 1
                                        ratio_new = total_ones[i][j]/total_neg_ones[i][j]
                                        # ones -= 1
                                        # neg_ones += 1
                                        # ratio_new = ones/neg_ones
                                        # print("flipped @ " + str(pos) + " from 1 to -1 => " + str(ratio_new))
                                        flips += 1
                                else:
                                    continue
                                                
                        # print("")
                    else:
                        continue

        # print("")
    print(f"ratios € [{lower}, {upper}]")
    print("flips: " + str(flips))

    total_ratios = np.divide(total_ones, total_neg_ones)
    total_ratios = np.around(total_ratios, decimals=2)

    # out_ratio_mod = "q_out/qweights_ratio_ratio_"+str(layer)+".txt"
    # with open(out_ratio_mod, "w") as f:
    #     # f.write("[")
    #     for line in total_ratios:
    #         # f.write("[")
    #         for value in line:
    #             f.write(str(value) + " ")
    #         f.write("\n")
    #         # f.write("]\n")
    #     # f.write("]")


    out_file_mod = "q_out/qweights_ratio_ecc"+str(layer)+".txt"
    with open(out_file_mod, "w") as f:
        f.write("[")

        # Write the list of integers to the file
        for integer in data[:-1]:
            f.write(str(integer) + ',\n')

        f.write(str(data[-1]) + "]")



