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


def count(data, array_type):
   
    total_ones = []
    total_neg_ones = []

    for row in data:

        if array_type == "3D":
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
        elif array_type == "1D":
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

for layer in range(1,9):
# for layer in range(1,5):
    print("")

    file = "q_in/qweights_orig_"+str(layer)+".txt"
    print(file)
    data = load_data_from_file(file)

    # if layer == 1 or layer == 2:
    #     array_type = "3D"
    # elif layer == 3 or layer == 4:
    #     array_type = "1D" 

    if layer == 7 or layer == 8:
        array_type = "1D"
    else:
        array_type = "3D" 

    total_ones, total_neg_ones = count(data, array_type)

    total_ratios = np.divide(total_ones, total_neg_ones)
    # Round the result to 2 decimal places using np.around()
    total_ratios = np.around(total_ratios, decimals=2)
    # np.set_printoptions(threshold=np.inf)

    # print global ratios
    _p1s = np.sum(total_ones)
    _m1s = np.sum(total_neg_ones)

    if _m1s == 0:
        total_ratio="inf"
    else:
        total_ratio="{:.2f}".format(_p1s/_m1s)
    print("1s: " + str(_p1s) + " | -1s: " + str(_m1s) + " | 1/-1 ratio: " + str(total_ratio))
    print("")

    # # print ratios per racetrack/block
    # out_file = "q_out/ratios_orig"+str(layer)+".txt"
    # with open(out_file, "w") as f:
    #     # f.write("[")
    #     for line in total_ratios:
    #         # f.write("[")
    #         for value in line:
    #             f.write(str(value) + " ")
    #         f.write("\n")
    #         # f.write("]\n")
    #     # f.write("]")

