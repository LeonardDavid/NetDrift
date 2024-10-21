import os
import numpy as np
import matplotlib.pyplot as plt

# Custom sorting function to extract the integer part from the filename
def sort_key(filename):
    return int(filename.split('_run_')[1].split('.txt')[0])


for layer in range(1,5):

    # folder = "edges_8020/"
    # folder = "mid_8020/"
    # folder = "edges_5050/"
    folder = "mid_5050/"

    # Directory containing the files
    directory = folder + str(layer)
    print(directory)

    # Initialize a dictionary to store data from each file
    data_dict = {}

    # Read each file in the directory
    for filename in os.listdir(directory):
        # print(filename)
        if filename.endswith('.txt'):  # Assuming the files have a .txt extension
            filepath = os.path.join(directory, filename)
            data = np.loadtxt(filepath)
            data_dict[filename] = data.flatten()

    # Sort data_dict by filename using the custom sorting function
    data_dict = dict(sorted(data_dict.items(), key=lambda item: sort_key(item[0])))

    # # Plot histogram for each file
    # plt.figure(figsize=(10, 6))
    # for filename, data in data_dict.items():
    #     plt.hist(data, bins=20, alpha=0.5, label=filename)

    # Create a bar plot for each file
    plt.figure(figsize=(16, 6))
    bin_edges = np.arange(-10, 12)  # Adjust the range and number of bins as needed
    # print(bin_edges)
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers = (bin_edges[:-1])
    # print(bin_centers)
    width = (bin_edges[1] - bin_edges[0]) / (len(data_dict) + 1)
    # width = 0.1
    # print(width)

    out_file = folder + "data_hist"+str(layer)+".txt"

    with open(out_file, 'w') as outfile:
        for i, (filename, data) in enumerate(data_dict.items()):
            # Clip data to the range -10 to 10
            data_clipped = np.clip(data, -10, 10)
            
            # Calculate histogram
            hist, bins = np.histogram(data_clipped, bins=bin_edges)

            for bin, h in zip(bins, hist):
                if bin == -10:
                    outfile.write(f"($\leq{bin}$, {h}) ")
                elif bin == 10:
                    outfile.write(f"($\geq{bin}$, {h}) ")
                else:
                    outfile.write(f"({bin}, {h}) ")

            outfile.write("\n")

            # plot the histogram
            plt.bar(bin_centers + (i-len(data_dict)/2+0.5)*width, hist, width=width, alpha=0.5, label=filename)

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of 2D Arrays from Multiple Files')
    # plt.xticks(np.arange(-11, 12, 1))  # Set x-axis ticks to show all integer values from -10 to 10
    plt.xticks(np.arange(-10, 11, 1), labels=['more-', *map(str, range(-9, 10)), 'more+'])  # Set x-axis ticks to show all integer values from -10 to 10 and "more"
    plt.legend(loc='upper right')

    # Show the plot
    plt.show()
    plt.savefig(folder +"histogram"+str(layer)+".png")
