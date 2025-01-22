import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def calculate_averages(results_dir):
    # Check if the directory exists
    if not os.path.isdir(results_dir):
        print(f"The directory {results_dir} does not exist.")
        return

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    all_averages = []

    # Iterate over each file in the directory
    for filename in os.listdir(results_dir):
        file_path = os.path.join(results_dir, filename)
        # print(file_path)
        
        # Check if the file is a TXT file
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            # Read the TXT file into a DataFrame, skipping the first line
            df = pd.read_csv(file_path, delimiter=',', header=None, skiprows=1)
            
            # Read the first line separately to get the first float
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                first_float = float(first_line.split(',')[0])
            
            # Calculate the average over the columns
            averages = df.mean(axis=0).tolist()
            
            # Append the first float at the front of the averages list
            averages.insert(0, first_float)

            all_averages.append(averages)

            # # Print the averages
            # print(f"Averages for {filename}:")
            # print(averages)
            # print()

            # plt.plot(averages, label=filename)

            marker = markers[len(all_averages) % len(markers)]
            plt.plot(averages, label=filename, marker=marker)


    x_values = range(len(averages))

    plt.yticks(range(0, 105, 5))
    # plt.yticks(range(80, 90, 1))
    arg3 = int(len(x_values)/10)
    if arg3 == 0:
        arg3 = 1
    plt.xticks(range(0, len(x_values) + 1, arg3))
    # plt.axhline(y=baseline, color='r', linestyle='--') # basline is different for every model

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.65)
    plt.xlabel('Iteration #')
    plt.ylabel('Accuracy')
    
    last_two_dirs = '/'.join(results_dir.rstrip('/').split('/')[-2:])
    plt.title(f"Averages over 10 runs from {last_two_dirs}")

    fig = plt.gcf()
    fig.set_size_inches(15,9)
    
    # Save the figure in the same directory
    output_path = os.path.join(results_dir, 'averages_plot.png')
    plt.savefig(output_path)

    return all_averages
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_auto.py <RESULTS_DIR>")
    else:
        results_dir = sys.argv[1]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        results_dir = os.path.join(parent_dir, results_dir)

        # print(results_dir)

        all_averages = calculate_averages(results_dir)
