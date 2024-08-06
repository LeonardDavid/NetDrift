import os

def modify_file(filepath):
  if not filepath.endswith(".py"):
    print(filepath)
    with open(filepath, 'r') as f:
      lines = f.readlines()
    modified_lines = [line.rstrip() + ",\n" for line in lines[:-1]]  # Add comma and newline

    modified_lines.append(lines[-1])

    with open(filepath, 'w') as f:
      f.write("[")
      f.write("".join(modified_lines))
      f.write("]")

def iterate_folders(folder_path):
  for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        modify_file(filepath)

# Replace 'path/to/your/folder' with your actual folder path
iterate_folders('./qwe')
