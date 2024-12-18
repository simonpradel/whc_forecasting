import os
import warnings

def count_files_in_folders(datasets, current_dir, check, max_files=None, return_datasets_below_threshold=False):
    """
    Counts the number of files in each dataset folder and optionally returns a list of dataset names 
    where the number of files is below the given threshold 'max_files'.

    Args:
        datasets (list): A list of dataset names. Each dataset corresponds to a folder in the directory.
        current_dir (str): The base directory where the dataset folders are located.
        check (str): A subdirectory or specific folder within each dataset folder where files will be counted.
        max_files (int, optional): The threshold for the maximum number of files allowed per dataset folder. 
                                    If provided, datasets with fewer files than this threshold will be tracked.
        return_datasets_below_threshold (bool, optional): If True, the function returns a list of dataset names
                                                           where the number of files is less than 'max_files'.

    Returns:
        dict: A dictionary where the keys are dataset names, and the values are the number of files in each dataset folder.
        list (optional): If 'return_datasets_below_threshold' is True, it returns a list of dataset names where the 
                         number of files is below 'max_files'.
    """
    
    # Initialize a dictionary to store the number of files for each dataset
    file_counts = {}
    
    # List to keep track of datasets that have fewer files than 'max_files' (if specified)
    below_threshold_datasets = []

    # Loop through each dataset in the provided list
    for dataset in datasets:
        dataset_dir = os.path.join(current_dir, dataset, check)  # Get the path to the dataset folder

        try:
            # Count the number of files in the directory
            file_count = len([f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))])
            file_counts[dataset] = file_count  # Store the count for the current dataset
            
            # If a max_files threshold is provided, check if the file count is below the threshold
            if max_files is not None and file_count < max_files:
                below_threshold_datasets.append(dataset)  # Add to the list if file count is below threshold
                
        except FileNotFoundError:
            # Warning if the dataset folder does not exist
            warnings.warn(f"Directory for dataset '{dataset}' not found.")
    
    # If 'return_datasets_below_threshold' is True, return both the file counts and the list of datasets below threshold
    if return_datasets_below_threshold:
        return file_counts, below_threshold_datasets

    # If not, return only the file counts
    return file_counts
