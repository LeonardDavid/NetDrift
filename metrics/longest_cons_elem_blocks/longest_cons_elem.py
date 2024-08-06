def find_longest_consecutive(data, arr_type):
  """
  This function finds the longest consecutive occurrences of the same element (1 or -1) in a list of lists.

  Args:
      data: A list of lists, where each inner list represents a row in the data.

  Returns:
      A tuple containing the element value (1 or -1) and the length of the longest consecutive occurrence.
  """
  longest_element = None
  longest_length = 0
  current_element = None
  current_length = 0

  if arr_type == "3D":

    for row in data:
        for row2 in row:
            for row3 in row2:
                for element in row3:
                    if element == current_element:
                        current_length += 1
                    else:
                        longest_element = current_element if current_length > longest_length else longest_element
                        longest_length = max(current_length, longest_length)
                        current_element = element
                        current_length = 1

    # Handle the last element
    longest_element = current_element if current_length > longest_length else longest_element
    longest_length = max(current_length, longest_length)

  elif arr_type == "1D":
     
    for row in data:
        for element in row:
            if element == current_element:
                current_length += 1
            else:
                longest_element = current_element if current_length > longest_length else longest_element
                longest_length = max(current_length, longest_length)
                current_element = element
                current_length = 1

    # Handle the last element
    longest_element = current_element if current_length > longest_length else longest_element
    longest_length = max(current_length, longest_length)
     

  return longest_element, longest_length

def read_data_from_file(filename):
  """
  This function reads the data from a file in the specified format.

  Args:
      filename: The name of the file containing the data.

  Returns:
      A list of lists representing the data in the file.
  """
  with open(filename, 'r') as f:
    data = eval(f.read())
  return data

def main():
  layer = 3
  if layer == 1 or layer == 2:
    arr_type = "3D"
  elif layer == 3 or layer == 4:
    arr_type = "1D"
  data = read_data_from_file('q_in/qweights_append_'+str(layer)+'.txt')
  element, length = find_longest_consecutive(data, arr_type)
  print(f"Longest consecutive occurrences: element {element}, length {length}")

if __name__ == "__main__":
  main()
