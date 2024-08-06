import sys
import numpy as np

def calculate_averages(filename):
  """
  Reads data from a file with an array of arrays format, calculates the average 
  of each element along the corresponding index, and returns the averages.

  Args:
      filename: The path to the file containing the data.

  Returns:
      A numpy array containing the averages of each element.
  """

  # Read the data from the file using numpy.loadtxt
  data = np.loadtxt(filename, delimiter=',')

  # Calculate the averages along axis 0 (columns)
  averages = np.mean(data, axis=0)

  return averages

# Example usage
layer = sys.argv[1]

filename = "all_results_"+str(layer)+".txt"  # Replace with your actual filename
averages = calculate_averages(filename)

# Print the averages
print(str(layer) + ": ")
print(averages)
