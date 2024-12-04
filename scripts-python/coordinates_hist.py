# Function to read from the input file and write to the output file
def reformat_data(input_file, output_file):
    try:
        # Open the input file for reading
        with open(input_file, 'r') as infile:
            # Read lines from the file
            lines = infile.readlines()

        # Open the output file for writing
        with open(output_file, 'w') as outfile:
            # Initialize an empty list to store formatted data
            formatted_data = []

            # Iterate over each line in the input file
            for line in lines:
                # Split each line into two parts (x and y values)
                x, y = line.strip().split()
                # Format as (x, y) and append to the list
                formatted_data.append(f"({x}, {y})")

            # Join the formatted data with spaces and write to output file
            outfile.write(" ".join(formatted_data))
        
        print(f"Data successfully written to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

# Input and output file names
input_file = 'coordinates_hist/coordinates_hist_in.txt'   # Replace with your input file path
output_file = 'coordinates_hist/coordinates_hist_out.txt' # Replace with your output file path

# Call the function to process the data
reformat_data(input_file, output_file)
