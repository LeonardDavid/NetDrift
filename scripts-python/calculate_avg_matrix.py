import sys
import ast
import numpy as np
import math
import os

def read_data_from_file(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            data.append(ast.literal_eval(line))
        return data

def calculate_mean(data):
    means = []
    for i in range(len(data[0])):
        index_values = [array[i] for array in data if array[i]]
        if index_values:
            mean_values = [math.ceil(np.mean([val for val in sublist if val])) for sublist in zip(*index_values)]
            means.append(mean_values)
        else:
            means.append([])
    return means

script_dir = os.path.dirname(os.path.abspath(__file__))
in_file = os.path.join(script_dir, "..", sys.argv[1])

data = read_data_from_file(in_file)
means = calculate_mean(data)

for mean in means:
    print(mean)

# print("")

if sys.argv[2] == "1":
    # print("Total average per layer:")
    print("=============================")
    print([sum(mean) for mean in means])
    print("=============================")