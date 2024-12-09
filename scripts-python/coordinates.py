# Function to read data from the file and return as a list of lists
def read_data_from_file(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        # Split each line by tab or space to get individual values
        data = [list(map(float, line.split())) for line in lines]
    return data

# Function to transpose data (swap rows and columns)
def transpose_data(data):
    # Use zip to transpose the data
    transposed = list(zip(*data))
    return transposed

# Function to format the transposed data and write to output file
def write_formatted_data_to_file(transposed_data, output_file):
    with open(output_file, 'w') as file:
        for row in transposed_data:
            # Create formatted strings for each row
            formatted_row = " ".join(f"({i}, {val})" for i, val in enumerate(row))
            file.write(formatted_row + "\n")

# Main function to process the input and generate the output
def process_file(input_file, output_file):
    # Read data from input file
    data = read_data_from_file(input_file)
    
    # Transpose the data
    transposed_data = transpose_data(data)
    
    # Write the formatted data to output file
    write_formatted_data_to_file(transposed_data, output_file)

# Input and output file names
input_file = 'coordinates/coordinates_in.txt'  # Replace with the path to your input file
output_file = 'coordinates/coordinates_out.txt'  # Replace with the desired output file path

# Run the process
process_file(input_file, output_file)
