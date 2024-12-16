import inspect

# Helper Function: Get the arguments of a function and the values of the arguments
# Function to create a dictionary with the actual arguments and their values
def get_function_args(func, *args, **kwargs):
    """
    Creates a dictionary with the function's parameters as keys and their corresponding 
    argument values as values. Handles positional arguments, keyword arguments, and default values.
    
    Arguments:
        func (Callable): The target function whose arguments are to be extracted.
        *args (tuple): Positional arguments passed to the function.
        **kwargs (dict): Keyword arguments passed to the function.
    
    Returns:
        dict: A dictionary with parameter names as keys and their corresponding argument
              values (or defaults if not provided) as values.
    """
    # Get the parameters of the function
    sig = inspect.signature(func)
    # Get the names of the parameters
    param_names = list(sig.parameters.keys())

    # Create a dictionary to store the arguments and their values
    args_dict = {}
    
    # Update the dictionary with the positional arguments
    for i, arg in enumerate(args):
        args_dict[param_names[i]] = arg

    # Update the dictionary with the keyword arguments
    for name, value in kwargs.items():
        args_dict[name] = value

    # Set default values for arguments not passed
    for name, param in sig.parameters.items():
        if name not in args_dict:
            if param.default is inspect.Parameter.empty:
                args_dict[name] = None
            else:
                args_dict[name] = param.default

    return args_dict


