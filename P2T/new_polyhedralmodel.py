# Import necessary files and functions
from file_ans import *
from Subfunctions import *
from code_generate import *
import argparse

def new_polyhedral_model(models):  
    new_models = []

    for model in models:
        new_model = {}
        print(model)
        # Parse domain constraints for the model to obtain conditions and inequalities (var_iqe)
        var_iqe, if_condition = parse_domain_constraints(model)

        
        # Determine the range of the variables based on the parsed domain constraints
        var_range = Determine_scope(var_iqe)
        # Determine the axis ranges for reduction based on the model and variable ranges
        axis_ranges = determine_reduce_axis(model, var_range)
        
        # Create new variables and mappings for the model's variable ranges
        new_var_range, mapping = create_new_variables(var_range, model)
        
        # Store the new iterators (keys of the new variable range) in the new model
        new_model['iterators'] = list(new_var_range.keys())
        # Store the new variable range in the new model
        new_model['var_range'] = new_var_range

        # new_model['domain_constraints'] = 
        
        # Extract the original iterator variables
        variables = model['iterators']
        # Inverse mapping for schedule variables based on the original variables and mapping
        inverse_map = inverse_mapping(model['schedule'], mapping, variables)

        domain_constraints, if_condition= update_domain_constraints(new_var_range,if_condition,inverse_map,new_model['iterators'])

        new_model['domain_constraints'] = domain_constraints
        
        # Parse the statement in the model to extract useful information
        statement_info = parse_statement(model)
        # Replace variables in the statement using the inverse mapping
        new_statement_info = replace_variables(statement_info, inverse_map)
        variables = model['iterators']
        new_model['reads'] = {}
        new_model['writes'] = {}
        update_array_access_index(model,new_model,inverse_map)
        # Store the new statement in the new model
        new_model['statement'] = new_statement_info

        
        # Add the new model to the list of new models
        new_models.append(new_model)


    # Return the updated models with new variables and statements
    return new_models
