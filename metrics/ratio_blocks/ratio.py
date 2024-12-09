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

# array_type = "3D"
# file = "qweights/32/qweights_0.1/qweights_append_1.txt" # 1s: 273 | -1s: 303 | 1/-1 ratio: 0.900990099009901
# file = "qweights/32/qweights_0.01/qweights_shift1_1.txt" # 1s: 254 | -1s: 313 | 1/-1 ratio: 0.8115015974440895

# file = "qweights/32/qweights_0.1/qweights_append_2.txt" # 1s: 17801 | -1s: 19063 | 1/-1 ratio: 0.9337984577453706
# file = "qweights/32/qweights_0.01/qweights_shift1_2.txt" # 1s: 17768 | -1s: 19096 | 1/-1 ratio: 0.9304566401340595


# array_type = "1D" # 1D | 3D
# file = "qweights/32/qweights_0.1/qweights_append_3.txt" # 1s: 3208141 | -1s: 3214387 | 1/-1 ratio: 0.9980568612304617
# file = "qweights/32/qweights_0.01/qweights_shift1_3.txt" # 1s: 3208757 | -1s: 3213771 | 1/-1 ratio: 0.9984398390551162

# file = "qweights/32/qweights_0.1/qweights_append_4.txt" # 1s: 10097 | -1s: 10383 | 1/-1 ratio: 0.9724549744775113
# file = "qweights/32/qweights_0.01/qweights_shift1_4.txt" # 1s: 10100 | -1s: 10380 | 1/-1 ratio: 0.9730250481695568


block_size = 64
err = 0.1

for layer in range(1,5):
  print("")
  # file = "qweights/" + str(block_size) + "/qweights_"+ str(err) +"/qweights_shift1_"+str(layer)+".txt"
  file = "q_in/qweights_orig_"+str(layer)+".txt"
  print(file)

  if layer == 1 or layer == 2:
    array_type = "3D"
  else:
    array_type = "1D" 

  # Read data from file
  with open(file, "r") as file:
    lines = file.readlines()

  total_ones = 0
  total_neg_ones = 0

  # Process each line
  for line in lines:
    # Convert line string to a list representing the 3D array
    data = eval(line.strip())
    ones, neg_ones = calculate_ratio(data, array_type)
    
    total_ones = total_ones + ones
    total_neg_ones = total_neg_ones + neg_ones

    if neg_ones == 0:
      ratio="inf"
    else:
      ratio="{:.2f}".format(ones/neg_ones)
    # print("1s: " + str(ones) + " | -1s: " + str(neg_ones) + " | 1/-1 ratio: " + str(ratio))

  print("==TOTALS==")
  if total_neg_ones == 0:
      total_ratio="inf"
  else:
      total_ratio="{:.2f}".format(total_ones/total_neg_ones)
  print("1s: " + str(total_ones) + " | -1s: " + str(total_neg_ones) + " | 1/-1 ratio: " + str(total_ratio))

