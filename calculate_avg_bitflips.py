import sys
import ast
import numpy as np
import math

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

layer = sys.argv[1]

filename = "all_bitflips_"+str(layer)+".txt"
data = read_data_from_file(filename)
means = calculate_mean(data)

for mean in means:
    print(mean)