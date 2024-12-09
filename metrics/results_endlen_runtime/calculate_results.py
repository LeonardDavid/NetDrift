import sys
import numpy as np

def calculate_averages(data):

  # Calculate the averages along axis 0 (columns)
  averages = np.mean(data, axis=0)

  return averages


def calculate_max(data):

  # Calculate the max values along axis 0 (columns)
  maximums = np.max(data, axis=0)

  return maximums


def calculate_min(data):

  # Calculate the min values along axis 0 (columns)
  minimums = np.min(data, axis=0)

  return minimums

# layer = sys.argv[1]

# filename = "all_results_2-1-0.1.txt"
# filename = "all_results_2-1-0.05.txt"
# filename = "all_results_2-1-0.025.txt"

# filenames = ["all_results_2-1-0.1.txt", "all_results_2-1-0.05.txt", "all_results_2-1-0.035.txt", "all_results_2-1-0.03.txt", "all_results_2-1-0.025.txt", "all_results_2-1-0.01.txt"]
filenames = ["all_results_2-0.1-1.txt", "all_results_2-0.05-1.txt", "all_results_2-0.035-1.txt", "all_results_2-0.03-1.txt", "all_results_2-0.025-1.txt", "all_results_2-0.01-1.txt"]

for filename in filenames:
    data = np.loadtxt(filename, delimiter=',')

    averages = calculate_averages(data)
    maximums = calculate_max(data)
    minimums = calculate_min(data)

    print(str(filename) + ": ")
    print(f"minimums: {minimums}")
    print(f"averages: {averages}")
    print(f"maximums: {maximums}")
