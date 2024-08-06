import numpy as np

def calculate_ratio(data, arr_type):
  count_ones = 0
  count_neg_ones = 0
  if arr_type is "3D":
    count_ones = sum(item == 1 for row in data for element in row for item in element)
    count_neg_ones = sum(item == -1 for row in data for element in row for item in element)
  elif arr_type is "1D":
    count_ones = sum(item == 1 for item in data)
    count_neg_ones = sum(item == -1 for item in data)

  return count_ones, count_neg_ones

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


block_size = 64
err = 0.1


# for layer in range(1,5):
for layer in range(1,5):
    print("")
    # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_append_"+str(layer)+".txt"
    # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift1_"+str(layer)+".txt"
    file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift10_"+str(layer)+".txt"
    print(file)

    if layer == 1 or layer == 2:
        array_type = "3D"
    elif layer == 3 or layer == 4:
        array_type = "1D" 

    total_ones = []
    total_neg_ones = []


    data = load_data_from_file(file)

    for row in data:

        if array_type is "3D":
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
        elif array_type is "1D":
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

        # ratios = np.divide(count_ones, count_neg_ones)
        # Round the result to 2 decimal places using np.around()
        # ratios = np.around(ratios, decimals=2)
        # print(ratios)
        # Try-except block to handle division by zero
        # try:
        #     ratios = np.divide(count_ones, count_neg_ones)
        #     # Round the result to 2 decimal places using np.around()
        #     ratios = np.around(ratios, decimals=2)
        #     # print(ratios)
        # except ZeroDivisionError:
        #     print("Error: Division by zero encountered.")
            

        total_ones.append(count_ones)
        total_neg_ones.append(count_neg_ones)

        # print(total_ones)
        # print(total_neg_ones)

    total_ratios = np.divide(total_ones, total_neg_ones)
    # Round the result to 2 decimal places using np.around()
    total_ratios = np.around(total_ratios, decimals=2)
    # np.set_printoptions(threshold=np.inf)

    # out_file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/ratios_append_"+str(layer)+".txt"
    # out_file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/ratios_shift1_"+str(layer)+".txt"
    out_file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/ratios_shift10_"+str(layer)+".txt"
    with open(out_file, "w") as f:
        # f.write("[")
        for line in total_ratios:
            # f.write("[")
            for value in line:
                f.write(str(value) + " ")
            f.write("\n")
            # f.write("]\n")
        # f.write("]")

    # try:
    #     total_ratios = np.divide(total_ones, total_neg_ones)
    #     # Round the result to 2 decimal places using np.around()
    #     total_ratios = np.around(total_ratios, decimals=2)
    #     print(total_ratios)
    # except ZeroDivisionError:
    #     print("Error: Division by zero encountered.")


