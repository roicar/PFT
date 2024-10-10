# Import preparation files
from file_ans import *
from Subfunctions import *
from code_generate import *


def convert_to_compute(function_name, models, array_size, optimization_method):

    code = ""

    code += function_application(function_name, 'n')
    code += array_application(array_size)

    if optimization_method == "padding":
        for model in models:
            new_axis_ranges = determine_new_reduce_axis(model['statement'], model['var_range'])

            code += reduce_axis_application(new_axis_ranges)
            Arrays_Indices, newsize = calculate_array_size(model['statement'], model['var_range'])

            needpadding_arrays = find_which_array_needpadding(array_size, newsize, Arrays_Indices)

            padding_if_condition, padding_index = find_padding_if_conditon(array_size, needpadding_arrays, model['statement'],model['var_range'])  
            tvm_code, updated_statement_info = padding(padding_if_condition, padding_index, needpadding_arrays, model['statement'])

            # Append the generated code to the existing code
            code += tvm_code
            code += compute_application(updated_statement_info, model['var_range'], new_axis_ranges)
    
    elif optimization_method == "shifting":
        for model in models:
            var_iqe, if_condition = parse_domain_constraints(model)

            # Determine the variable range
            var_range = Determine_scope(var_iqe)

            statement_info = parse_statement(model)
            
            new_array_sizes, max_neg = find_shifting_size(statement_info, array_size)
            new_statement_info = shift_replace_variables(statement_info, max_neg)
            code += shifting(new_statement_info, new_array_sizes)
            

    elif optimization_method == "none":
        # No optimization
        for model in models:
            var_iqe,if_condition = parse_domain_constraints(model)

            #确定变量范围
            var_range = Determine_scope(var_iqe)
            print(var_range)

            axis_ranges =determine_reduce_axis(model,var_range)

           
            axis_ranges =determine_reduce_axis(model,var_range)
            code+=reduce_axis_application(axis_ranges)
            
            statement_info = parse_statement(model)
            print(statement_info)
            
            code+=compute_application(statement_info,var_range,axis_ranges)
            


    code += return_array_application(array_size)
    return code    
