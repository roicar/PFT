from file_ans import *
import sympy as sp
import re

def parse_domain_constraints(polyhedral_model):

    n = sp.symbols('n')
    I = sp.symbols('I', real=True)
    # List of constraint strings
    iterators = polyhedral_model['iterators']

    iter_symbols = [sp.Symbol(var) for var in iterators]
    
    inequalities = []
    for constraint in polyhedral_model['domain_constraints']:
        constraint_expr = sp.sympify(constraint)
        if any(symbol in constraint_expr.free_symbols for symbol in iter_symbols): 
            # Remove extra spaces from the string,remove the n>0
            clean_constraint = constraint.replace(" ", "")
            lhs, rhs = clean_constraint.split(">=")
            lhs_expr = sp.sympify(lhs)
            rhs_expr = sp.sympify(rhs)
            inequalities.append(sp.Ge(lhs_expr, rhs_expr))
    # Define two parts
    part1 = []
    part2 = []
    # Classify the inequalities
    for inequality in inequalities:
        lhs = inequality.lhs
        # Extract variables in the inequality (excluding n)
        vars_in_inequality = lhs.free_symbols - {n}
        # Check if it only contains one variable and its coefficient is ±1
        if len(vars_in_inequality) == 1:
            var = list(vars_in_inequality)[0]
            coeff = lhs.as_coefficients_dict().get(var, 0)
            if coeff == 1 or coeff == -1:
                part1.append(inequality)
            else:
                part2.append(inequality)
        else:
            part2.append(inequality)

    part1_str = [f"{ineq.lhs} >= {ineq.rhs}" for ineq in part1]
    part2_str = [f"{ineq.lhs} >= {ineq.rhs}" for ineq in part2]
    return part1_str, part2_str


# def parse_domain_constraint(polyhedral_model):

#     n = sp.symbols('n')
#     I = sp.symbols('I', real=True)
#     # List of constraint strings
#     iterators = polyhedral_model['iterators']

#     iter_symbols = [sp.Symbol(var) for var in iterators]
    
#     inequalities = []
#     variable_pattern = re.compile(r'\b[a-zA-Z_]\w*\b')
#     for constraint in polyhedral_model['domain_constraints']:
#         variables_in_constraint = set(variable_pattern.findall(constraint)) 

#         if any(symbol in variables_in_constraint for symbol in iter_symbols): 

#             clean_constraint = constraint.replace(" ", "") 
#             lhs, rhs = clean_constraint.split(">=")

#             lhs_expr = sp.sympify(lhs) 
#             rhs_expr = sp.sympify(rhs) 
#             inequalities.append(sp.Ge(lhs_expr, rhs_expr))

#             # clean_constraint = constraint.replace(" ", "")
#             # lhs, rhs = clean_constraint.split(">=")
#             # lhs_expr = sp.sympify(lhs)
#             # rhs_expr = sp.sympify(rhs)
#             # inequalities.append(sp.Ge(lhs_expr, rhs_expr))
#     # Define two parts
#     part1 = []
#     # Classify the inequalities
#     for inequality in inequalities:
#         lhs = inequality.lhs
#         # Extract variables in the inequality (excluding n)
#         vars_in_inequality = lhs.free_symbols - {n}
#         # Check if it only contains one variable and its coefficient is ±1
#         part1.append(inequality)


#     part1_str = [f"{ineq.lhs} >= {ineq.rhs}" for ineq in part1]

#     return part1_str

# Determine the range of variables
def Determine_scope(part1_str):
    range_dict = {}
    for constraint in part1_str:
        # Parse constraints in the form of "+i >= 0" or "-i +n -1 >= 0"
        match = re.match(r"(?:\s*)([+-]?)([a-zA-Z])(?:\s*)([+-]?)(.*?)(?:\s*)([><=]+)(?:\s*)(\d+)", constraint)
        if match:
            sign, var, op0, minmax, op1, value = match.groups()

            if var not in range_dict:
                range_dict[var] = [float('inf'), float('-inf')]  # Initialize minimum and maximum values
            if sign == '-':
                range_dict[var][1] = minmax  
            else:
                if op0 == "":
                    range_dict[var][0] = value
                elif op0 == '-':
                    range_dict[var][0] = minmax
                elif op0 == '+':
                    range_dict[var][0] = '-' + minmax              
    return range_dict




def determine_reduce_axis(polyhedral_model, range_dict):
    axes = set()
    axis_ranges = {}

    # Extract all iterators from the model info
    all_iterators = set(polyhedral_model['iterators'])
        
    # Extract the write array information
    write_array = polyhedral_model['writes']

    # Iterate over the write arrays to collect used axes
    reax = set()
    for value_list in write_array.values():
        reax.update(value_list)

        # Determine the reduce axes by finding the difference
    axes.update(all_iterators - reax)

    for axe in axes:
        if axe in range_dict:  # Ensure the axis is present in range_dict
            axis_ranges[axe] = range_dict[axe]

    # Return the reduce axis names and their ranges
    return axis_ranges



def determine_new_reduce_axis(new_statement_info, new_var_range):
    # Extract variable names from left_part (parse index expressions)
    left_part_vars = set()
    for array_name, indices in new_statement_info['left_part'].items():
        for index_expr in indices:
            # Convert index expression to a sympy expression
            expr = sp.sympify(index_expr)
            # Extract variables from the expression
            left_part_vars.update([str(v) for v in expr.free_symbols])

    # Calculate new_reduce_axis
    new_reduce_axis = {var: range_ for var, range_ in new_var_range.items() if var not in left_part_vars}

    return new_reduce_axis    

def parse_range(range_str):
    # Parse a single range string and convert it into a computable range
    range_str = range_str.replace(' ', '')
    match = re.match(r'([^\-]+)-(.+)', range_str)
    if not match:
        raise ValueError(f"Invalid range format: {range_str}")
    start, end = match.groups()
    start = sp.sympify(start)
    end = sp.sympify(end)
    return start, end

# Calculate new variable intervals based on the schedule
def interval_operation(ranges, expressions, results):
    # Create symbolic variables
    variables = {var: sp.Symbol(var) for var in ranges.keys()}
    n = sp.Symbol('n')
    
    # Calculate result intervals
    for expression in expressions:
        # Parse the expression
        
        expr = sp.sympify(expression, locals=variables)
        min_vals = {}
        max_vals = {}
        for var, (start, end) in ranges.items():
            if "-" in str(expression) and "n" in str(expression):
                min_vals[variables[var]] = end  # Swap minimum and maximum values
                max_vals[variables[var]] = start
            else:
                min_vals[variables[var]] = start
                max_vals[variables[var]] = end
        
        result_min = expr.subs(min_vals)
        result_max = expr.subs(max_vals)
        
        oppsite_min = -result_min
        final_min = sp.simplify(result_min+oppsite_min)
        final_max = sp.simplify(result_max+oppsite_min)
        
        results.append((final_min, final_max))
    # Return result intervals
    return results

def generate_variable_names(count, prefix='t'):
    return [f"{prefix}{i+1}" for i in range(count)]

def create_new_variables(intervals, model):
    ranges = {}
    new_variables = {}
    mapping = {}
    for var, range_str in intervals.items():

        ranges[var] = parse_range(range_str[0] + "-" + range_str[1])

    expressions = model['schedule']

    # n-j
    interval_results = interval_operation(ranges, expressions, [])

    # Generate new variable names
    variable_names = generate_variable_names(len(expressions))
    for idx, (result_min, result_max) in enumerate(interval_results):
        new_variables[variable_names[idx]] = (result_min, result_max)
        mapping[expressions[idx]] = variable_names[idx]
    

    return new_variables, mapping

def inverse_mapping(expressions, mapping, variables):
    # Create symbolic variables
    symbols = {var: sp.Symbol(var) for var in variables}
    new_symbols = {mapping[expr]: sp.Symbol(mapping[expr]) for expr in expressions}
    
    # Build the system of equations
    equations = []
    for expr in expressions:
        lhs = sp.sympify(expr, locals=symbols)
        rhs = sp.sympify(mapping[expr], locals=new_symbols)
        equations.append(sp.Eq(lhs, rhs))
    
    # Solve the system of equations
    solutions = sp.solve(equations, list(symbols.values()))
    
    # Construct the inverse mapping dictionary
    inverse_mapping_dict = {}
    for var in variables:
        inverse_mapping_dict[var] = str(solutions[symbols[var]])
    
    return inverse_mapping_dict



# Extracts the statement and divides it into left and right parts
def parse_statement(polyhedral_info):
    result = {}
    
    statement = polyhedral_info['statement']
    # Split the statement at the '=' sign
    if isinstance(statement,dict):
        left_dict = statement['left_part']
        right_part = statement['right_part']
    else:
        left_part, right_part = statement.split('=')
    # elif  in statement:
    #     left_part, right_part = statement.split('=')
    # Remove leading and trailing whitespace
        left_part = left_part.strip()
        right_part = right_part.strip()
    
    # Initialize dictionaries for left part
        left_dict = {}
    
        # Match array name and loop variables in the left part
        if '][' not in left_part:
            pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[(.*?)\]'
            match = re.match(pattern, left_part)
            if match:
                array_name = match.group(1)
                index = match.group(2)
                left_dict={array_name: [index]}
            else:
                raise ValueError("The input string format is incorrect")
        else:
            # Extract multi-dimensional array index part
            pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[(.+)\]'
            match = re.match(pattern, left_part)
            if match:
                array_name = match.group(1)
                indices_part = match.group(2)
                # Split the index part
                indices = indices_part.split('][')
                left_dict ={array_name: indices}
            else:
                raise ValueError("The input string format is incorrect")
       
    result['left_part'] = left_dict
    result['right_part'] = right_part
    return result


def replace_variables(statement_info, old_to_new_mapping):
    def replace_expression(expression, mapping):
        sorted_mapping = sorted(mapping.items(), key=lambda x: -len(x[0]))
        for old_var, new_var in sorted_mapping:
            expression = re.sub(r'\b{}\b'.format(re.escape(old_var)), f'({new_var})', expression)
        return expression
    
    def simplify_expression(expression):
        try:
            # Use regular expressions to extract all sub-expressions
            pattern = re.compile(r'(\w+)\[([^\[\]]+)\]')
            matches = pattern.findall(expression)
            
            simplified_expression = expression
            for var, sub_expr in matches:
                simplified_sub_expr = sp.simplify(sub_expr)
                simplified_expression = simplified_expression.replace(sub_expr, str(simplified_sub_expr))
            
            # Simplify the entire expression
            sympy_expr = sp.sympify(simplified_expression)
            simplified_expr = sp.simplify(sympy_expr)
            return str(simplified_expr)
        except Exception as e:
            print(f"Error simplifying expression {expression}: {e}")
            return expression
    
    new_statement_info = {'left_part': {}, 'right_part': ''}

    # Replace and simplify the left part
    new_left_part = {}
    for key, value in statement_info['left_part'].items():
        new_key = old_to_new_mapping.get(key, key)
        new_value = [simplify_expression(replace_expression(v, old_to_new_mapping)) for v in value]
        new_left_part[new_key] = new_value
        
    new_statement_info['left_part'] = new_left_part

    # Replace and simplify the right part
    right_part = replace_expression(statement_info['right_part'], old_to_new_mapping)
    simplified_right_part = right_part
    
    # Use regular expressions to extract all sub-expressions and simplify
    pattern = re.compile(r'(\w+)\[([^\[\]]+)\]')
    matches = pattern.findall(right_part)
    
    for var, sub_expr in matches:
        simplified_sub_expr = str(sp.simplify(sub_expr))
        simplified_right_part = simplified_right_part.replace(sub_expr, simplified_sub_expr)

    new_statement_info['right_part'] = simplified_right_part

    return new_statement_info


     
def calculate_array_size(new_statement_info, new_var_range):

    def get_max_size(indices, var_ranges):
        exprs = [sp.sympify(idx) for idx in indices.split(',')]
        
        min_sizes = []
        max_sizes = []
        
        for expr in exprs:
            min_expr = expr
            max_expr = expr
            
            for var, (min_val, max_val) in var_ranges.items():
                var_sym = sp.symbols(var)
                coeff = expr.coeff(var_sym)
                
                if coeff > 0:
                    min_expr = min_expr.subs(var_sym, min_val)
                    max_expr = max_expr.subs(var_sym, max_val)
                else:
                    min_expr = min_expr.subs(var_sym, max_val)
                    max_expr = max_expr.subs(var_sym, min_val)
            
            min_sizes.append(min_expr)
            max_sizes.append(max_expr)
        
        max_sizes = [sp.simplify(max_expr - min_expr + 1) for max_expr, min_expr in zip(max_sizes, min_sizes)]
        return max_sizes

    left_part = new_statement_info['left_part']
    right_part = new_statement_info['right_part']
    
    pattern = re.compile(r'(\w+)\[([^\]]+)\]')
    matches = pattern.findall(right_part.replace('][', ','))
    
    Arrays_Indices = {}
    new_array_size = {}
    
    for array, indices in matches:
        indices = indices.strip()
        Arrays_Indices[array] = indices
        new_array_size[array] = get_max_size(indices, new_var_range)
    
    for var, indices_list in left_part.items():
        indices = ', '.join(indices_list)
        Arrays_Indices[var] = indices
        new_array_size[var] = get_max_size(indices, new_var_range)

    return Arrays_Indices, new_array_size

def find_which_array_needpadding(array_size, new_array_size, Arrays_Indices):
    old_array_size = array_size
    arrays_need_padding = {}

    n = sp.symbols('n')

    for array_name, old_size in old_array_size.items():
        if array_name in new_array_size:
            new_size = new_array_size[array_name]

            if isinstance(old_size, list) and isinstance(new_size, list):
                old_size = [sp.sympify(os) for os in old_size]
                new_size = [sp.sympify(ns) for ns in new_size]
                if len(old_size) != len(new_size) or any(sp.simplify(os - ns) != 0 for os, ns in zip(old_size, new_size)):
                    arrays_need_padding[array_name] = {
                        'indices': Arrays_Indices[array_name],
                        'new_size': new_size
                    }
            else:
                old_size = sp.sympify(old_size)
                new_size = sp.sympify(new_size)
                if sp.simplify(old_size - new_size) != 0:
                    arrays_need_padding[array_name] = {
                        'indices': Arrays_Indices[array_name],
                        'new_size': new_size
                    }

    return arrays_need_padding

def array_index_convert_if_condition(array_name,info, var_range):
    range_dict = {}
   
    indices_expr = sp.sympify(info['indices'])

    # 计算索引表达式的取值范围
    min_val = indices_expr
    max_val = indices_expr

    for var, (start, end) in var_range.items():
        var_symbol = sp.Symbol(var)
        coeff = indices_expr.coeff(var_symbol)
        if coeff >= 0:
            # 如果系数为正，则min使用start，max使用end
            min_val = min_val.subs(var_symbol, start)
            max_val = max_val.subs(var_symbol, end)
        
        else:
            # 如果系数为负，则min使用end，max使用start
            min_val = min_val.subs(var_symbol, end)
            max_val = max_val.subs(var_symbol, start)
    # 计算取值范围
    min_range = sp.simplify(min_val)
    max_range = sp.simplify(max_val)

    range_dict[array_name] = (min_range,max_range)
    return range_dict

def find_padding_if_conditon(oldsize, needpadding_arrays, statement_info,new_ranges):
    
    n = sp.symbols('n')
    padding_if_condition = {}
    padding_index = {}
    for array_name, padding_info in needpadding_arrays.items():
        indices = padding_info['indices']
        new_sizes = padding_info['new_size']
        index_exprs = [sp.sympify(idx) for idx in indices.split(', ')]

        conditions = []
        index_expressions = []

        for index_expr, old_size in zip(index_exprs, oldsize[array_name]):
            coeff_n = index_expr.coeff(n)
            old_size = sp.sympify(f"{old_size}-1")
            if coeff_n != 0:
                if coeff_n > 0:
                    if coeff_n == 1:
                        condition = f"0 <= i + n <= {old_size}"
                        index_expr_str = f"i +n"
                    else:    
                        condition = f"0 <= i + {coeff_n}*n <= {old_size}"
                        index_expr_str = f"i + {coeff_n}*n"
                else:
                    if coeff_n == -1:
                        condition = f"0 <= i - n <= {old_size}"
                        index_expr_str = f"i - n"
                    else:
                        condition = f"0 <= i - {abs(coeff_n)}*n <= {old_size}"
                        index_expr_str = f"i - {abs(coeff_n)}*n"
            else:
                range_dict = array_index_convert_if_condition(array_name,padding_info,new_ranges)

                for arrray_name,(min_range,max_range) in range_dict.items():
                    if min_range ==0:
                        condition = f"0 <= i <= {old_size}"
                        index_expr_str = f"i"
                    else:
                        condition = f"{-min_range}<= i <= {max_range}"
                        index_expr_str = f"i-{-min_range}"
                        index_expr_str = str(sp.sympify(index_expr_str))
                        ringht_part = statement_info['right_part']
                        new_indexs = f"{indices}-({min_range})"
                        new_indexs = str(sp.sympify(new_indexs))

                        ringht_part = ringht_part.replace(indices,new_indexs)
                        statement_info['right_part'] = ringht_part
            conditions.append(condition)
            index_expressions.append(index_expr_str)
            
        if conditions:
            padding_if_condition[array_name] = ' and '.join(conditions)
            padding_index[array_name] = index_expressions
    
    return padding_if_condition, padding_index    

def find_shifting_size(statement_info, array_sizes):
    array_sizes0 = {key: (sp.symbols('n'), sp.symbols('n')) for key in array_sizes}
    
    right_part = statement_info['right_part']
    
    left_part = statement_info['left_part']
    
    array_name, left_part_indices = list(left_part.items())[0] 

    
    array_size = array_sizes0[array_name]

    num_dimensions = len(array_size)
    updated_size = list(array_size) 
    max_neg_list = [0] * num_dimensions
    
    for dim in range(num_dimensions):
        index = left_part_indices[dim]
        
        # Regular expression to match all +n and -n operations for the current dimension
        pattern = re.compile(rf'\b{index}\s*([\+\-]\s*\d+)')
        matches = pattern.findall(right_part)
        
        max_neg = 0  # Record the maximum negative value for the current dimension
        max_pos = 0  # Record the maximum positive value for the current dimension
        
        for match in matches:
            offset = match.replace(' ', '')  # Remove spaces
            if offset.startswith('-'):
                num = int(offset[1:])  # Get the absolute value
                max_neg = max(max_neg, num)  # Update the maximum negative value
            elif offset.startswith('+'):
                num = int(offset[1:])  # Get the absolute value
                max_pos = max(max_pos, num)  # Update the maximum positive value

        max_neg_list[dim] = max_neg

        # Update size
        if max_neg > 0:
            updated_size[dim] -= max_neg  # Subtract for the maximum negative value
        if max_pos > 0:
            updated_size[dim] -= max_pos  # Subtract for the maximum positive value
    
    return tuple(updated_size), max_neg_list
def shift_replace_variables(statement_info, max_n):
    # Parse the left-side variables
    left_part = statement_info['left_part']
    
    array_name, left_part_indices = list(left_part.items())[0] 
    # Extract the offset amounts

    # Get the right-side part
    right_part = statement_info['right_part']
    for i in range(len(max_n)):
        right_part_str =f'({max_n[i]} + {left_part_indices[i]})'
        right_part = right_part.replace(left_part_indices[i], right_part_str)
        
    # Perform string replacement
    # modified_right_part = right_part.replace('i', f'({offset_i} + i)').replace('j', f'({offset_j} + j)')


    # Create the new statement
    new_statement = {
        'left_part': left_part,
        'right_part': str(right_part)
    }
    
    return new_statement



def update_domain_constraints(new_ranges,if_condition,inverse_map,new_variables):
    inequalities = []
    replaced_conditions = []
    simplified_conditions = []
    n = sp.Symbol('n') # Define symbolic variable 'n'
    variables = {name: sp.Symbol(name) for name in new_variables}  # Create new symbolic variables
    
    for var, (lower, upper) in new_ranges.items():
        # Generate inequalities for lower and upper bounds
        lower_bound = f"{var} >= {lower}"
        upper_bound = f"-{var} + {upper} >= 0"

    # Append the inequalities to the list
        inequalities.append(lower_bound)
        inequalities.append(upper_bound)
    
    for condition in if_condition:
        # Perform string replacement for each variable in inverse_map
        replaced_condition = condition
        for old_var, new_expr in inverse_map.items():
            replaced_condition = replaced_condition.replace(old_var, new_expr)
        
        replaced_conditions.append(replaced_condition)
    

    # Parse and simplify each replaced condition
    for condition in replaced_conditions:
        # Convert the replaced string condition into a symbolic expression
        expr = sp.sympify(condition, locals=variables)
        # Simplify the expression
        simplified_expr = sp.simplify(expr)
        # Append the simplified expression to the list
        simplified_conditions.append(simplified_expr)
    return inequalities,simplified_conditions

def update_array_access_index(model,new_model, inverse_map):
    #  reads and writes
    for access_type in ['reads', 'writes']:
        for array_name, indices in model[access_type].items():
            updated_indices = []

            for index_expr in indices:
                # replace index
                updated_expr = index_expr
                for old_var, new_expr in inverse_map.items():
                    updated_expr = updated_expr.replace(old_var, new_expr)
                
                # sympy
                simplified_expr = sp.simplify(sp.sympify(updated_expr))
                
                # 
                updated_indices.append(str(simplified_expr))
            
            #
            new_model[access_type][array_name] = updated_indices

    
