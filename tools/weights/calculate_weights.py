import numpy as np
import pickle
import copy
from scipy.optimize import differential_evolution
import os
import time
import torch
import torch.nn.functional as F

def calculate_weights(results, optim_method="ensemble_selection", include_groups = None, remove_groups=False, hillclimbsets=1, max_iterations=50, saveResults=True, save_path=os.getcwd(), verbosity = 0):
    """
    Applies an optimization method on the aggregated forecasts and includes handling of 'remove' methods.

    Parameters:
    - results (dict): Dictionary containing results from calculate_weights, including:
        - 'aggregated_forecast': Dictionary with each group name as key and corresponding DataFrame (columns: 'date', 'pred', 'total') as value.
        - 'Y_test_values': Actual test values.
        - 'forecast_method': Method used for forecast level.
    - optim_method (str): The optimization method ('differential_evolution', 'ensemble_selection', 'optimize_nnls').
    - hillclimbsets (int): Number of sets for ensemble selection method.
    - max_iterations (int): Maximum number of iterations for optimization.
    - saveResults (bool): If True, results will be saved to a pickle file.
    - save_path (str): Directory where the results will be saved.

    Returns:
    - results (dict): Dictionary with the optimized weights added.
    """
    # Create a deep copy of the results dictionary to avoid modifying the original data
    results_copy = copy.deepcopy(results)
    print(results_copy['meta_data'].keys())
    print(results_copy['meta_data']["time_limit"])
    start_time = time.time()

    Y_test_values = results_copy['aggregated_forecast'][('dataset',)].loc[~results_copy['aggregated_forecast'][('dataset',)]['pred'].isna(), 'total'].values

    if include_groups is not None:
        if isinstance(include_groups, str):
            include_groups = [include_groups]  # Convert single string to list
        
        # Only keep keys that contain all elements in include_groups
        keys_to_remove = [key for key in results_copy['aggregated_forecast'].keys() if not all(group in key for group in include_groups)]
        
        for key in keys_to_remove:
            del results_copy['aggregated_forecast'][key]

        if verbosity >= 4:
            print(f"Included groups after filtering: {list(results_copy['aggregated_forecast'].keys())}")

    remove_group_alias = ""
    if(remove_groups):
        results_copy['aggregated_forecast'] = iteratively_qualify_and_remove_weaker_groups(
            results_copy['aggregated_forecast'], 
            Y_test_values
        )
        remove_group_alias = "_rm"

        if(verbosity >= 4):
            print("qualified groups:")
            print(results_copy['aggregated_forecast'].keys())

    all_selected_groups = []
    hat_Y_test = []
    
    # Iterate through the groups in the 'aggregated_forecast' dictionary
    for group_name, df in results_copy['aggregated_forecast'].items():
        # Remove rows where 'pred' is NaN
        df_clean = df.dropna(subset=['pred'])
        
        if not df_clean.empty:
            # Append the group name to the list of selected groups
            all_selected_groups.append(group_name)
            
            # Append the 'pred' column to the multivariate array (hat_Y_test)
            hat_Y_test.append(df_clean['pred'].values)
    
    # Convert the list of arrays (hat_Y_test) to a 2D array for optimization
    hat_Y_test = np.column_stack(hat_Y_test)

    bounds = [(0, 1) for _ in range(hat_Y_test.shape[1])]
    print(f"\nStart optimization with {optim_method}...")
    
    # Optimization logic based on the selected method
    if optim_method == "differential_evolution":
        scaled_weights = optimize_differential_evolution(Y_test_values, hat_Y_test, bounds, n_sets=hillclimbsets)
        optim_method_alias = "de"
    elif optim_method == "ensemble_selection":
        scaled_weights = optimize_ensemble_selection(Y_test_values, hat_Y_test, max_iterations=max_iterations, n_sets=hillclimbsets)
        optim_method_alias = "es"
    elif optim_method == "optimize_nnls":
        scaled_weights = optimize_nnls(Y_test_values, hat_Y_test, n_sets=hillclimbsets)
        optim_method_alias = "nnls"
    else:
        optim_method_alias = ""
    
    # Combine selected groups with their corresponding weights
    results_copy['selected_groups_with_weights'] = [(group, weight) for group, weight in zip(all_selected_groups, scaled_weights) if weight != 0]
    
    if(verbosity >= 4):
        print("selected_groups_with_weights", results_copy['selected_groups_with_weights'])

    best_model_alias = results_copy["meta_data"]["best_model_alias"]
    include_models_alias = results_copy["meta_data"]["include_models_alias"] 

    end_time = time.time()
    elapsed_time = end_time - start_time
    results_copy["meta_data"]["elapsed_time"] = results_copy["meta_data"]["elapsed_time"] + elapsed_time

    filename = f"Weights_{results_copy['Input_Arguments']['model']}_{results_copy['Input_Arguments']['forecast_method']}_{include_models_alias}{best_model_alias}{optim_method_alias}{remove_group_alias}_{results_copy['Input_Arguments']['fold_length']}_{results_copy['Input_Arguments']['n_splits']}_{results_copy['meta_data']['time_limit']}"
    results_copy["meta_data"]["file_name"] = filename
    results_copy['Input_Arguments']['optim_method'] = optim_method
    results_copy['Input_Arguments']['remove_groups'] = remove_groups 
    results_copy['Input_Arguments']['hillclimbsets'] = hillclimbsets
    results_copy['Input_Arguments']['max_iterations'] = max_iterations
    results_copy['Input_Arguments']['include_groups'] = include_groups
    results_copy['Input_Arguments']['time_limit'] = results_copy['meta_data']["time_limit"]

    # Save the results if the flag is set to True
    if saveResults:
        weights_path = os.path.join(save_path, "weights")
        pkl_filename = os.path.join(weights_path, filename + ".pkl")
        os.makedirs(weights_path, exist_ok=True)

        with open(pkl_filename, 'wb') as f:
            pickle.dump(results_copy, f)
        print(f"Results saved as '{pkl_filename}'.")

    return results_copy

# Weight Optimizer  
def consolidate_forecasts(all_selected_groups, forecast_cache):
    """
    Consolidates the forecasts from all selected groups into a single forecast array.

    Parameters:
    - all_selected_groups (list): List of selected groups for forecasting.
    - forecast_cache (dict): Cached forecasts for each group.

    Returns:
    - array: A 2D array where each column corresponds to the forecast of a selected group.
    """
    hat_Y_test = []
    
    for group_key in all_selected_groups:
        group_forecasts = forecast_cache[group_key]
        hat_Y_test.append(group_forecasts)

    return np.column_stack(hat_Y_test)

def calculate_loss(w, Y_t, hat_Y_t_k):
    """
    Calculates the loss (Mean Absolute Error) with L1 regularization for weight optimization.

    Parameters:
    - w (array): Array of weights.
    - Y_t (array): Actual values of the time series.
    - hat_Y_t_k (array): Forecasted values from different models.

    Returns:
    - float: The calculated loss with regularization.
    """
    hat_Y_t = np.dot(hat_Y_t_k, w)
    return np.mean(np.abs(Y_t - hat_Y_t)) 

def optimize_differential_evolution(Y_test_values, hat_Y_test, bounds, n_sets=1):
    """
    Optimizes the weights using the differential evolution algorithm, considering multiple data subsets.

    Parameters:
    - Y_test_values (array): Actual values of the test data.
    - hat_Y_test (array): Forecasted values from different models.
    - bounds (list of tuple): Bounds for each weight, typically (0, 1) for all weights.
    - n_sets (int): Number of sets to split the test data for optimization (default: 1, no split).

    Returns:
    - array: Optimized and normalized weights.
    """
    np.random.seed(42)  # For reproducibility

    # Convert inputs to float for computation
    Y_test_values = np.array(Y_test_values, dtype=float)
    hat_Y_test = np.array(hat_Y_test, dtype=float)

    # Initialize variables
    set_size = len(Y_test_values) // n_sets
    accumulated_weights = np.zeros(hat_Y_test.shape[1])

    start_idx = 0

    for set_idx in range(n_sets):
        # Determine indices for current set
        if set_idx == n_sets - 1:  # Include remainder in the last set
            end_idx = len(Y_test_values)
        else:
            end_idx = start_idx + set_size

        # Subset Y_test_values and hat_Y_test for the current set
        Y_test_set = Y_test_values[start_idx:end_idx]
        hat_Y_test_set = hat_Y_test[start_idx:end_idx]

        # Define the loss function
        def calculate_loss(weights, Y_test_set, hat_Y_test_set):
            weights = np.clip(weights, 0, 1)  # Ensure weights are non-negative
            weights = weights / np.sum(weights)  # Normalize weights
            combined_forecast = np.dot(hat_Y_test_set, weights)
            loss = np.mean(np.abs(Y_test_set - combined_forecast))
            return loss

        # Perform differential evolution
        result = differential_evolution(
            calculate_loss,
            bounds,
            args=(Y_test_set, hat_Y_test_set),
            strategy='best1bin',
            maxiter=1000
        )
        # Extract and normalize weights
        scaled_weights = result.x / np.sum(result.x)

        # Handle edge case for invalid weights
        scaled_weights = [1.0] if np.isnan(scaled_weights[0]) else scaled_weights

        # Accumulate weights
        accumulated_weights += scaled_weights

        # Update start index for next set
        start_idx = end_idx

    # Average weights across all sets and normalize again
    averaged_weights = accumulated_weights / n_sets
    final_weights = averaged_weights / np.sum(averaged_weights)

    return final_weights


def optimize_nnls(Y_test_values, hat_Y_test, n_sets=1):
    """
    Optimizes the weights using Non-Negative Least Squares (NNLS) from PyTorch.
    Splits the test data into multiple sets, calculates weights for each set, 
    and combines the results.

    Parameters:
    - Y_test_values (array): Actual values of the test data.
    - hat_Y_test (array): Forecasted values from different models.
    - n_sets (int): Number of sets to split the test data for optimization (default: 1, no split).

    Returns:
    - array: Optimized and normalized weights that sum to 1.
    """
    # Convert inputs to np.float32 to avoid object types
    Y_test_values = np.array(Y_test_values, dtype=np.float32).flatten()  # Ensure it's 1D
    hat_Y_test = np.array(hat_Y_test, dtype=np.float32)

    # Initialize variables
    set_size = len(Y_test_values) // n_sets
    accumulated_weights = np.zeros(hat_Y_test.shape[1])

    start_idx = 0

    for set_idx in range(n_sets):
        # Determine indices for current set
        if set_idx == n_sets - 1:  # Include remainder in the last set
            end_idx = len(Y_test_values)
        else:
            end_idx = start_idx + set_size

        # Subset Y_test_values and hat_Y_test for the current set
        Y_test_set = Y_test_values[start_idx:end_idx]
        hat_Y_test_set = hat_Y_test[start_idx:end_idx]

        # Convert to torch tensors
        Y_test_set = torch.tensor(Y_test_set, dtype=torch.float32).unsqueeze(1)  # (21,) -> (21, 1)
        hat_Y_test_set = torch.tensor(hat_Y_test_set, dtype=torch.float32)  # (21, 64)

        # Apply NNLS using torch's lstsq
        solution = torch.linalg.lstsq(hat_Y_test_set, Y_test_set, driver='gelsd')
        weights = solution.solution.flatten()

        # Ensure weights are non-negative
        weights = F.relu(weights)

        # Normalize the weights to sum to 1
        scaled_weights = weights / torch.sum(weights)
        scaled_weights = scaled_weights.detach().numpy()

        # Handle edge case for invalid weights
        scaled_weights = [1.0] if np.isnan(scaled_weights[0]) else scaled_weights

        # Accumulate weights
        accumulated_weights += scaled_weights

        # Update start index for next set
        start_idx = end_idx

    # Average weights across all sets and normalize again
    averaged_weights = accumulated_weights / n_sets
    final_weights = averaged_weights / np.sum(averaged_weights)

    return final_weights

def optimize_ensemble_selection(
    Y_test_values, hat_Y_test, n_sets=1, max_iterations=50, verbose=True):
    """
    Optimizes model weights using a hill-climbing approach with ensemble selection.

    The method iteratively selects models based on minimizing the mean absolute error (MAE)
    over the test dataset. Models can be selected multiple times, and the resulting weights
    are determined based on the frequency of model selections.

    Parameters:
    - Y_test_values (array-like): 
        The actual target values for the test dataset. Shape: (n_samples,).
    - hat_Y_test (array-like): 
        Forecasted values produced by different models. 
        Shape: (n_samples, n_models), where each column corresponds to a model's predictions.
    - n_sets (int, optional): 
        Number of subsets to split the test data for independent optimizations. 
        If `n_sets = 1`, no split is performed. Default is 1.
    - max_iterations (int, optional): 
        Maximum number of iterations for the hill-climbing ensemble selection per subset. 
        Default is 50.
    - verbose (bool, optional): 
        If True, prints progress information such as the current iteration and selected models.
        Default is True.

    Returns:
    - numpy.ndarray: 
        Normalized weights for each model based on their selection frequency. 
        Shape: (n_models,), where each entry corresponds to a model's weight.
    """
    # Determine the size of each subset if n_sets > 1
    set_size = len(Y_test_values) // n_sets
    selected_combinations = []  # List to store selected models across all subsets
    model_count = {i: 0 for i in range(hat_Y_test.shape[1])}  # Count model selections

    start_idx = 0  # Initialize start index for data splitting

    # Loop over subsets of the test data
    for set_idx in range(n_sets):
        if verbose:
            print(f"\nProcessing Subset {set_idx + 1} of {n_sets}")
        
        # Define the end index for the current subset
        if set_idx == n_sets - 1:  # Last subset includes remaining data points
            end_idx = len(Y_test_values)
        else:
            end_idx = start_idx + set_size

        Y_test_set = Y_test_values[start_idx:end_idx]  # Actual values for the subset
        selected_combination = []  # Store selected models for this subset

        # Perform hill-climbing ensemble selection for the current subset
        for iteration in range(1, max_iterations + 1):
            # Print progress every 50 iterations if verbose
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration} / {max_iterations}")

            best_loss = float('inf')  # Initialize the best loss as infinity
            best_model_idx = None  # To track the index of the best model

            # Evaluate adding each model to the current combination
            for model_idx in range(hat_Y_test.shape[1]):
                # Compute the combined forecast for the current combination
                current_combination = selected_combination + [model_idx]
                combined_forecast = np.mean(
                    hat_Y_test[start_idx:end_idx, current_combination], axis=1
                )
                
                # Compute the Mean Absolute Error (MAE) for the current combination
                current_loss = np.mean(
                    np.abs(np.array(Y_test_set, dtype=float) - np.array(combined_forecast, dtype=float))
                )

                # Update the best model if the current combination improves the loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model_idx = model_idx

            # Add the best-performing model to the combination
            if best_model_idx is not None:
                selected_combination.append(best_model_idx)
                model_count[best_model_idx] += 1  # Increment the count of the selected model

        # Store the selected model indices for the current subset
        selected_combinations.extend(selected_combination)
        start_idx = end_idx  # Update start index for the next subset

    # Determine model selection frequency and normalize weights
    unique_combinations, counts = np.unique(selected_combinations, return_counts=True)
    print(f"Models selected: {unique_combinations}")
    print(f"Selection counts: {counts}")
    
    # Initialize weights and scale based on selection frequency
    scaled_weights = np.zeros(hat_Y_test.shape[1])
    scaled_weights[unique_combinations] = counts / len(selected_combinations)

    return scaled_weights

def iteratively_qualify_and_remove_weaker_groups(aggregated_forecast, Y_test_values, verbosity=0):
    """
    Iteratively qualifies and removes weaker groups by comparing the MAE (Mean Absolute Error) 
    of the current level with the average MAE of the previous levels. Groups that exceed 
    the qualification threshold are disqualified in later iterations.

    Parameters:
    - aggregated_forecast (dict): 
        A dictionary where each key represents a group name (a string) and each value 
        is a pandas DataFrame containing the following columns:
          - 'date': Date or time of the forecast.
          - 'pred': Predicted values for the corresponding time periods.
          - 'total': Actual values for the corresponding time periods.

    - Y_test_values (np.array): 
        An array of true test values (ground truth) used for evaluation of forecasts. 
        This parameter is not explicitly used in this implementation but remains for compatibility.

    - verbosity (int, optional, default=0): 
        The level of output verbosity during execution:
          - 0: Silent (no output).
          - 4: Outputs detailed information about the qualification process, 
               including group losses and disqualified groups.

    Returns:
    - aggregated_forecast (dict): 
        A filtered dictionary containing only the qualified groups. Disqualified groups 
        (those with higher errors) are removed.
    """

    # List to store group names along with their corresponding losses (MAE)
    all_selected_groups_with_losses = []
    for group_name, df in aggregated_forecast.items():
        # Drop rows with missing predictions to ensure clean comparisons
        df_clean = df.dropna(subset=['pred'])
        
        if df_clean.empty:
            mae = float('inf')  # Assign a high error if no predictions are available
        else:
            # Extract predictions and actual values
            y_pred = df_clean['pred'].values
            y_true = df_clean['total'].values
            mae = np.mean(np.abs(y_pred - y_true))  # Calculate MAE
        
        # Append the group name and its loss as a tuple
        all_selected_groups_with_losses.append((group_name, mae))

    # Determine the maximum depth/level based on the group names
    max_level = max(len(group_name) for group_name, _ in all_selected_groups_with_losses)

    # Iteratively qualify groups starting from level 2 (levels represent grouping hierarchies)
    for level in range(2, max_level + 1):
        # Filter groups at the current level
        current_level_groups_with_losses = [
            (g, loss) for g, loss in all_selected_groups_with_losses if len(g) == level
        ]

        if not current_level_groups_with_losses:
            continue  # Skip if no groups exist at the current level

        # Calculate the mean loss for all groups at previous levels
        previous_level_losses = [
            loss for g, loss in all_selected_groups_with_losses if len(g) < level
        ]
        mean_previous_level_loss = np.mean(previous_level_losses)
        
        if verbosity >= 4:
            print(f"Mean loss for level <= {level}: {mean_previous_level_loss}")

        # Disqualify groups whose loss exceeds the mean loss of previous levels
        disqualified_groups_with_losses = [
            (g, loss) for g, loss in current_level_groups_with_losses if loss > mean_previous_level_loss
        ]

        # Keep only the qualified groups
        all_selected_groups_with_losses = [
            (g, loss) for g, loss in all_selected_groups_with_losses 
            if g not in [group for group, _ in disqualified_groups_with_losses]
        ]

        if verbosity >= 4:
            print(f"Level {level}: Qualified groups:")
            for group, loss in all_selected_groups_with_losses:
                if len(group) <= level:
                    print(f"  Group: {group}, Loss: {loss}")

        # Recalculate the mean loss for qualified groups across all levels up to the current level
        mean_loss_qualified_groups = np.mean([
            loss for g, loss in all_selected_groups_with_losses if len(g) <= level
        ])

        if verbosity >= 4:
            print(f"New mean loss for qualified groups in level <= {level}: {mean_loss_qualified_groups}")

        # Remove groups from previous levels if their losses exceed the new mean loss
        for prev_level in range(1, level):
            to_remove = [
                (g, loss) for g, loss in all_selected_groups_with_losses 
                if len(g) <= prev_level and loss > mean_loss_qualified_groups
            ]

            # Remove disqualified groups
            all_selected_groups_with_losses = [
                tup for tup in all_selected_groups_with_losses if tup not in to_remove
            ]

            if verbosity >= 4 and to_remove:
                print(f"Removed weaker groups from level {prev_level}: {to_remove} "
                      f"against the new mean {mean_loss_qualified_groups}")

    # Filter the original dictionary to include only the qualified groups
    aggregated_forecast = {
        group: aggregated_forecast[group] for group, _ in all_selected_groups_with_losses
    }

    return aggregated_forecast
