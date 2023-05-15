from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import os
import glob
import subprocess
import shutil




api = KaggleApi()
api.authenticate()

kernels = kaggle.api.kernels_list(search= "countries")



print(len(kernels))
for i in range(len(kernels)):
    if(kernels[i].ref != ""):
        kaggle.api.kernels_pull(kernels[i].ref,"kernel_extraction", metadata=True)


# Directory containing the .ipynb files
notebook_dir = "kernel_extraction"

# Get a list of all .ipynb files in the directory
notebook_files = glob.glob(f"{notebook_dir}/*.ipynb")


# Run the nbconvert command on each file
for notebook_file in notebook_files:
    print(notebook_file)
    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "script", notebook_file],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"Converted {notebook_file} to .py")
    else:
        print(f"Error converting {notebook_file} to .py:")
        print(result.stderr)

# # List all files in the directory
# files = glob.glob(os.path.join(notebook_dir, "*"))
#

destination_dir = "converted_python_files"

# Filter out .py files
py_files = [file for file in os.listdir(notebook_dir) if file.endswith(".py")]

for py_file in py_files:
    shutil.move(os.path.join(notebook_dir, py_file), os.path.join(destination_dir, py_file))


#
# # Delete all non-.py files
# for file in non_py_files:
#     os.remove(file)