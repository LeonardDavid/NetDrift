#!/bin/bash

# Base filename (without the number)
base_filename="run_auto_all_"

# Parameters to pass to the scripts
parameters="CIFAR 10 64 CUSTOM 0"

# Loop through numbers 1 to 8
for i in {1..8}; do
  # Construct the full filename
  filename="${base_filename}${i}.sh"

  # Check if the file exists
  if [[ -f "$filename" ]]; then
    # Execute the script with parameters
    ./"$filename" $parameters
  else
    echo "File $filename not found."
  fi
done

python_script="calculate_avg.py"

for i in {1..8}; do
  python "$python_script" $i
done