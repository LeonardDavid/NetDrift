import matplotlib.pyplot as plt
import os

import sys
import ast

# Check if an input file is provided as an argument
# if len(sys.argv) != 2:
#     print("Usage: " + sys.argv[0] + " <input_file>")
#     sys.exit(1)

# Check if the input file exists
try:
    with open(sys.argv[1]) as f:
        pass
except FileNotFoundError:
    print("Input file not found: " + sys.argv[1])
    sys.exit(1)

# Read the penultimate line from the input file
with open(sys.argv[1]) as f:
    lines = f.readlines()
    penultimate_line = lines[-2].strip()

# Extract the double values from the string representation of the array
y_values = ast.literal_eval(penultimate_line)

print(y_values)

# # Print the double values in the array
# for value in y_values:
#     print(value)

loops = sys.argv[4]
perror = sys.argv[5]
unproc_layer = sys.argv[6]

output_dir = sys.argv[2] + "/figures/" + unproc_layer + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nn_model = sys.argv[3]
if(nn_model == "FMNIST"):
    baseline = 91.08 # 9108/8582/7694
elif (nn_model == "CIFAR"):
    baseline = 85.85
elif (nn_model == "RESNET"):
    baseline = 76.94
else:
    baseline = 0.0

filename = "graph_" + str(nn_model) + "-" + str(unproc_layer) + "-" + str(loops) + "-" + str(perror)

x_values = range(len(y_values))


plt.plot(x_values, y_values)
plt.yticks(range(0, 100, 5))
# plt.yticks(range(80, 90, 1))
arg3 = int(len(x_values)/10)
if arg3 == 0:
    arg3 = 1
plt.xticks(range(0, len(x_values) + 1, arg3))
plt.axhline(y=baseline, color='r', linestyle='--')

plt.grid(True)
plt.xlabel('Iteration #')
plt.ylabel('Accuracy')
plt.title('Accumulated shift for ' + str(nn_model) + ' with unprotected layer ' + str(unproc_layer) + ' over ' + str(loops) + ' inference iterations with error ' + str(perror))

fig = plt.gcf()
fig.set_size_inches(15,9)
plt.savefig(output_dir + filename + ".png")

# plt.show()