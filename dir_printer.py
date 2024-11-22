import os
import sys

def print_py_tree(directory, prefix=""):
    contents = sorted(os.listdir(directory))
    for i, item in enumerate(contents):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            if i == len(contents) - 1:
                print(f"{prefix}└── {item}/")
                print_py_tree(path, prefix + "    ")
            else:
                print(f"{prefix}├── {item}/")
                print_py_tree(path, prefix + "│   ")
        elif item.endswith('.py'):
            if i == len(contents) - 1:
                print(f"{prefix}└── {item}")
            else:
                print(f"{prefix}├── {item}")

def process_py_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print("\n" + "=" * 100 + "\n")
                print(f"File: {file}")
                print(f"Directory: {root}")
                print("\nContent:")
                with open(file_path, 'r') as f:
                    print(f.read())
                print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    print("Directory tree of .py files:")
    print_py_tree(directory)
    print("\nProcessing .py files:")
    process_py_files(directory)