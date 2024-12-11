import inspect

# Helper Function: Get the arguments of a function and the values of the arguments
# Function to create a dictionary with the actual arguments and their values
def get_function_args(func, *args, **kwargs):
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