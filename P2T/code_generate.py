from file_ans import *
from Subfunctions import *

def function_application(function_name, params):
    # Convert the parameter list to a comma-separated string
    param_str = ", ".join(params)
    
    # Generate the function definition code
    function_code = f"def {function_name}({param_str}):\n"
    
    return function_code



def array_application(array_info):
    tvm_code = ""
    
    for array_name, size_exprs in array_info.items():
        # Convert size expressions to symbolic expressions and add 1 to each expression
        size_exprs_sym = [sp.sympify(size_expr) for size_expr in size_exprs]
        size_exprs_str = [str(sp.simplify(size_expr)) for size_expr in size_exprs_sym]
        size_str = ", ".join(size_exprs_str)

        # Generate TVM placeholder code
        code = f'{array_name} = te.placeholder(({size_str},), name="{array_name}", dtype="float32")\n'
        tvm_code += code

    return tvm_code

def reduce_axis_application(axis_info):
    tvm_code = ""
    
    for axis_name, axis_range in axis_info.items():
        # Convert range expressions to symbolic expressions and simplify
        min_val = sp.sympify(axis_range[0])
        max_val = sp.sympify(axis_range[1]) + 1

        # Generate TVM reduce_axis code
        code = f'{axis_name} = te.reduce_axis(({min_val}, {max_val}), name="{axis_name}")\n'
        tvm_code += code

    return tvm_code


def padding(padding_if_condition, padding_index, needpadding_arrays, statement_info):    
    tvm_code = ""
    n = sp.symbols('n')
    for array_name, padding_info in needpadding_arrays.items():
        new_size = padding_info['new_size'][0]
        conditions = padding_if_condition[array_name]
        indices = padding_index[array_name][0]
        
        # Change the original array name to the new array name
        new_array_name = array_name + '1'

        # Generate TVM compute code
        code = f"""
{new_array_name} = te.compute(
    ({new_size},), 
    lambda i:
        te.if_then_else(
            tvm.tir.all({conditions}), 
            {array_name}[{indices}], 
            0
    ), 
    name='{new_array_name}'
)
"""     
        tvm_code += code

        # Update only the right_part
        right_part = statement_info['right_part']
        
        # Replace the array name
        updated_right_part = right_part.replace(array_name, new_array_name)
        
        # Use regular expressions to match the array indices of new_array_name
        def replace_negative_indices(match):
            expr = match.group(1)
            # Parse the expression using SymPy
            sympy_expr = sp.sympify(expr)
    
            # 获取 n 的系数
            coeff_n = sympy_expr.coeff(n)
        
            # 只有当 n 的系数为负时才移除 n 项
            if coeff_n < 0:
                simplified_expr = sympy_expr - coeff_n * n
            else:
                simplified_expr = sympy_expr
        
            return f"[{simplified_expr}]"
        

        
        # Match only the index part of new_array_name and handle negative values
        updated_right_part = re.sub(rf'{new_array_name}\[([^\]]+)\]', lambda m: f"{new_array_name}{replace_negative_indices(m)}", updated_right_part)
        
        # Update the right_part in statement_info
        statement_info['right_part'] = updated_right_part
        
    return tvm_code, statement_info



# Input statement info and new array size information after shifting
def shifting(statement_info, new_array_size):
    tvm_code = ""
    n = sp.symbols('n')
    right_part = statement_info['right_part']
    
    left_part = statement_info['left_part']
    array_name, left_part_indices = list(left_part.items())[0] 
    new_left_part_indices = ', '.join(map(str, left_part_indices))

    code = f"""
{array_name} = te.compute(
    ({new_array_size}, 
    lambda {new_left_part_indices}:
        (
        {right_part}
    ), 
    name='{array_name}'
)
"""  
    tvm_code += code
    return tvm_code
    

def compute_application(statement_info, var_range, axis_info):
    left_part = statement_info['left_part']
    right_part = statement_info['right_part']

    # Extract array name and indices from left_part
    array_name, indices = list(left_part.items())[0]
    indices_str = ", ".join(indices)
    
    # Get the maximum value of each index variable from var_range and simplify
    array_sizes = [sp.sympify(f"{var_range[idx][1]} + 1") for idx in indices]
    array_sizes_str = ", ".join([str(size.simplify()) for size in array_sizes])

    # Extract information from axis_info
    axis_name = list(axis_info.keys())[0]

    # Generate te.compute code
    compute_expr = f'te.sum({right_part}, axis={axis_name})'
    compute_code = f'{array_name} = te.compute(({array_sizes_str}), lambda {indices_str}: {compute_expr}, name="{array_name}")\n'
    
    return compute_code

def return_array_application(array_size):
    # Extract all array names
    array_names = list(array_size.keys())
    
    # Join array names into a string to generate a return statement
    return_statement = "return [" + ", ".join(array_names) + "]"
    
    return return_statement
