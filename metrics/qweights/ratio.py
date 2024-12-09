import ast
import numpy as np

def calculate_ratio(data):
  """
  Calculates the ratio of 1s to -1s in a 3D array, handling cases with only 1s.

  Args:
      data: A list representing a 3D array.

  Returns:
      The ratio of 1s to -1s as a float, or -1 if there are only 1s.
  """
#   total_elements = len(data) * len(data[0]) * len(data[0][0])
  count_ones = sum(item == 1 for row in data for element in row for item in element)
  count_neg_ones = sum(item == -1 for row in data for element in row for item in element)

#   # Check for only 1s present (all elements are 1)
#   if count_neg_ones == 0 and count_ones == total_elements:
#     return -1  # Special case: all 1s, ratio undefined
#   elif count_neg_ones == 0:
#     return float('inf')  # Ratio is infinite if no -1s present (other cases)

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

# file = "qweights_append_1.txt" # 1s: 273 | -1s: 303 | 1/-1 ratio: 0.900990099009901
# file = "qweights_shift1_1.txt" # 1s: 254 | -1s: 313 | 1/-1 ratio: 0.8115015974440895
# file = "qweights_append_2.txt" # 1s: 17801 | -1s: 19063 | 1/-1 ratio: 0.9337984577453706
file = "qweights_shift.txt" # 1s: 17768 | -1s: 19096 | 1/-1 ratio: 0.9304566401340595

data = load_data_from_file(file)
# print(data)

for row in data:
  print(row)
  for element in row:
    print(element)
    for item in element:
      print(item)

# # Read data from file
# with open(file, "r") as file:
#   lines = file.readlines()
#   # lines2=ast.literal_eval(lines)

# # print(lines2)

# total_ones = 0
# total_neg_ones = 0

# # Process each line
# for line in lines:
#   print(line.strip())
#   # Convert line string to a list representing the 3D array
#   data = eval(line.strip())
#   # data = ast.literal_eval(line.strip())

#   # data_string = line.strip()
#   # data = eval(data_string[1:-1])
#   ones, neg_ones = calculate_ratio(data)
  
#   total_ones = total_ones + ones
#   total_neg_ones = total_neg_ones + neg_ones

#   if neg_ones == 0:
#     ratio="inf"
#   else:
#     ratio=ones/neg_ones
#   print("1s: " + str(ones) + " | -1s: " + str(neg_ones) + " | 1/-1 ratio: " + str(ratio))

# print("==TOTALS==")
# if total_neg_ones == 0:
#     total_ratio="inf"
# else:
#     total_ratio=total_ones/total_neg_ones
# print("1s: " + str(total_ones) + " | -1s: " + str(total_neg_ones) + " | 1/-1 ratio: " + str(total_ratio))

