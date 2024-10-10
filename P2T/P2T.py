# Import necessary modules and functions from various files
from file_ans import *
from Subfunctions import *
from code_generate import *
from new_polyhedralmodel import *
import argparse
from convert_to_compute import *
import os
import sys

# Create an argument parser to parse command-line arguments
parser = argparse.ArgumentParser(description="Parse polyhedral model and array size from files and apply optimization.")
# Argument for the polyhedral model file
parser.add_argument("model_file", type=str, help="The name of the file to parse the polyhedral model")
# Argument for the array size file
parser.add_argument("array_size_file", type=str, help="The name of the file to parse the array size")
# Argument for selecting the optimization method (padding, shifting, or none)
parser.add_argument("optimization_method", type=str, choices=["padding", "shifting", "none"], help="The optimization method to apply")

# Parse the command-line arguments
args = parser.parse_args()

# Parse the polyhedral model from the specified file and extract the function name and models
function_name, models = parse_polyhedral_model(args.model_file)
# Parse the array size information from the specified file
array_size = parse_array_size(args.array_size_file)
# Get the chosen optimization method from the arguments
optimization_method = args.optimization_method

# Extract the schedule from the first model
sch = models[0]['schedule']
# Extract the iterators from the first model
iterators = models[0]['iterators']
new_models = {}

# Check if the schedule and iterators are identical
if set(sch) == set(iterators):
    # If the schedule and iterators are identical, use the existing models
    new_models = models
else:
    # If they are different, create new polyhedral models
    new_models = new_polyhedral_model(models)
    # Check if the schedule is a subset of the iterators in the new models
    if sch == set(new_models[0]['iterators']).issubset:
        # Raise an error if the schedule is not a subset of the iterators
        raise ValueError("The schedule is not a subset of the iterators.")

# Convert the models to computation code based on the provided optimization method
tc = convert_to_compute(function_name, new_models, array_size, optimization_method)

# Print the generated computation code
file_path = os.path.join('new_compute',function_name+'.py')
with open(file_path,'w') as f:
    print(tc,file = f)
      
print(tc)
