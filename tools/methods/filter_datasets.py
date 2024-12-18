# Iterate over each dataset and apply the function with error handling
def filter_datasets(datasets, run_Telefonica_data, run_other_data, run_dataset=None, exclude_datasets=None):
    """
    Filters datasets based on the provided inclusion and exclusion criteria.

    Args:
        datasets (list): A list of dictionaries, where each dictionary represents a dataset. 
                         Each dictionary should contain at least the keys:
                         - "name" (str): The name of the dataset.
                         - "Telefoncia_data" (bool): Indicates if the dataset belongs to Telefonica data.
        run_Telefonica_data (bool): If False, exclude Telefonica datasets.
        run_other_data (bool): If False, exclude non-Telefonica datasets.
        run_dataset (list, optional): A list of dataset names to explicitly include. Defaults to None.
        exclude_datasets (list, optional): A list of dataset names to explicitly exclude. Defaults to None.

    Returns:
        list: A filtered list of datasets based on the provided criteria.
    """
    # Filter specific datasets if specified
    if run_dataset:
        datasets = [dataset for dataset in datasets if dataset["name"] in run_dataset]
    
    # Exclude specific datasets if specified
    if exclude_datasets:
        datasets = [dataset for dataset in datasets if dataset["name"] not in exclude_datasets]
    
    # Filter based on Telefonica data flag
    if not run_Telefonica_data:
        datasets = [dataset for dataset in datasets if not dataset["Telefoncia_data"]]
    if not run_other_data:
        datasets = [dataset for dataset in datasets if dataset["Telefoncia_data"]]
    
    return datasets


