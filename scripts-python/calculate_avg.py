import sys
import os
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

script_dir = os.path.dirname(os.path.abspath(__file__))
in_file = os.path.join(script_dir, "..", sys.argv[1])

averages = calculate_averages(in_file)

print(averages)
