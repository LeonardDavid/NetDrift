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


block_sizes = [64, 32, 16, 8, 4, 2]
errs = [0.1, 0.01, 0.001, 0.0001]
shift = 10

# block_sizes = [64]
# errs = [0.1]

for block_size in block_sizes:
    for err in errs:
        
        bitflips_arr = []
        
        for layer in range(1,5):
            print("")

            # layer = 4
            if layer == 1 or layer == 2:
                array_type = "3D"
            else:
                array_type = "1D" 

            file_init = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_append_"+str(layer)+".txt"
            file_shift = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift"+str(shift)+"_"+str(layer)+".txt"
            # file_init = "q_in/qweights_append_"+str(layer)+".txt"
            # file_shift = "q_in/qweights_shift1_"+str(layer)+".txt"
            print(file_init)
            print(file_shift)


            data_init = load_data_from_file(file_init)
            data_shift = load_data_from_file(file_shift)

            bitflips = 0

            # Process each line
            for row_init, row_shift in zip(data_init, data_shift):
                
                if array_type == "3D":
                    # print(row_init)
                    # print(row_shift)
                    # break
                    for element_init, element_shift in zip(row_init, row_shift):
                        # print(element_init)
                        # print(element_shift)
                        # break
                        for item_init, item_shift in zip(element_init, element_shift):
                            # print(item_init)
                            # print(item_shift)
                            # break
                            for weight_init, weight_shift in zip(item_init, item_shift):
                                # print(weight_init)
                                # print(weight_shift)
                                # break
                                if weight_init != weight_shift:
                                    bitflips += 1
                elif array_type == "1D":
                    # print(row_init)
                    # print(row_shift)
                    # break
                    for item_init, item_shift in zip(row_init, row_shift):
                        if item_init != item_shift:
                            bitflips += 1
            print(bitflips)
            bitflips_arr.append(bitflips)


        out_file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/bitflips"+str(shift)+".txt"
        # out_file = "q_out/bitflips_"+ str(block_size) +"_"+ str(err)+".txt"
        with open(out_file, "w") as f:
            for value in bitflips_arr:
                f.write(str(value) + " ")
                f.write("\n")

        out_file2 = "metrics/nr_bitflips/bitflips/"+str(shift)+"/bitflip_" + str(block_size) + "_" + str(err) +".txt"
        # out_file = "q_out/bitflips_"+ str(block_size) +"_"+ str(err)+".txt"
        with open(out_file2, "w") as f2:
            for value in bitflips_arr:
                f2.write(str(value) + " ")
                f2.write("\n")
