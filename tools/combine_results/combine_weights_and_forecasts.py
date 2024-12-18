import pandas as pd
from itertools import chain, combinations

def combine_weights_and_forecasts(chosenWeights, dict_forecast, verbosity=0, target_col='weighted_pred'):
    """
    Combine forecasts from different sources using specified weights and produce weighted predictions.

    This function aggregates forecasts based on selected weights, applies these weights to generate weighted forecasts,
    and outputs a new dictionary with the aggregated and weighted predictions for different combinations.

    Args:
        chosenWeights (list of tuples):
            A list of tuples where each tuple consists of a key combination and the corresponding weight.
            Example: [(('key1', 'key2'), 0.5), (('key3',), 0.3)]
        dict_forecast (dict):
            A dictionary where keys are tuples representing unique combinations, and values are pandas DataFrames
            containing forecasts. The DataFrames must have a 'pred' column with forecasted values.
        verbosity (int, optional):
            Controls the level of verbosity for debugging output. Higher values print more details. Default is 0.
        target_col (str, optional):
            The name of the column in the output DataFrames that will store the weighted predictions. Default is 'weighted_pred'.

    Returns:
        dict: A new dictionary with the same keys as `dict_forecast` but containing DataFrames with weighted forecasts.
    """

    # Step 0: Extract keys from chosenWeights and find common elements across all keys
    selected_combinations = [set(tup[0]) for tup in chosenWeights]
    common_elements = set.intersection(*selected_combinations)  # Find intersection of all key sets

    # Create a sorted list of common elements for consistent use
    common_elements = sorted(common_elements)

    if verbosity > 5:
        print("Step 0: Common elements")
        print(common_elements)
        print()

    # Step 1: Initialize a new dictionary with keys from dict_forecast and values set to None
    new_forecast_dict = {key: None for key in dict_forecast.keys()}

    if verbosity > 5:
        print("Step 1: New dictionary with empty values")
        print(new_forecast_dict)
        print()

    # Step 2: Aggregate all DataFrames on the common level (common_elements)
    def aggregate_predictions(df, group_by_cols):
        return df.groupby(group_by_cols)['pred'].sum().reset_index()

    aggregated_forecasts = {}
    for key, df in dict_forecast.items():
        if set(common_elements).issubset(set(key)):
            group_by_cols = list(common_elements) + ['date']
            aggregated_df = aggregate_predictions(df, group_by_cols)
            aggregated_forecasts[key] = aggregated_df

    if verbosity > 5:
        print("Step 2: Aggregated forecasts")
        print(aggregated_forecasts)
        print()

    # Step 3: Apply weights to the aggregated forecasts
    weighted_forecasts = {}
    for key, df in aggregated_forecasts.items():
        weight = dict(chosenWeights).get(key, 0)
        df[target_col] = df['pred'] * weight
        weighted_forecasts[key] = df

    if verbosity > 5:
        print("Step 3: Weighted forecasts")
        print(weighted_forecasts)
        print()

    # Step 4: Compute the sum of weighted forecasts for identical combinations
    combined_weighted_forecasts = pd.concat(weighted_forecasts.values(), ignore_index=True)
    summed_weighted_forecasts = combined_weighted_forecasts.groupby(common_elements + ['date'])[target_col].sum().reset_index()

    if verbosity > 5:
        print("Step 4: Summed weighted forecasts")
        print(summed_weighted_forecasts)
        print()

    # Step 5: Compute the sum of weighted forecasts for all subsets of `common_elements`
    def get_all_subsets(s):
        """Helper function to compute all subsets of a set."""
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    subsets = list(get_all_subsets(common_elements))

    if verbosity > 5:
        print("Step 5: All subsets of common elements")
        print(subsets)
        print()

    for subset in subsets:
        subset = list(subset)
        grouped_forecast = summed_weighted_forecasts.groupby(subset + ['date'])[target_col].sum().reset_index()

        for key in new_forecast_dict.keys():
            if set(key) == set(subset):
                new_forecast_dict[key] = grouped_forecast

    if verbosity > 5:
        print("Step 5: Updated dictionary with weighted forecasts")
        print(new_forecast_dict)
        print()

    # Step 6: Remove all empty entries from the new dictionary
    new_forecast_dict = {key: value for key, value in new_forecast_dict.items() if value is not None}

    if verbosity > 5:
        print("Step 6: Cleaned dictionary")
        print(new_forecast_dict)
        print()

    return new_forecast_dict


# Function to match dictionaries and add 'forecast_weighted' to the results
def create_weights_forecast_dict(
    weights_result, 
    forecast_dic, 
    weighted_losses, 
    keys_to_check, 
    additional_keys_weights=None, 
    additional_keys_forecast=None, 
    verbosity=0
):
    """
    Matches weights and forecast dictionaries based on specific keys and computes combined weighted forecasts.
    
    Args:
        weights_result (dict): Dictionary containing weight-related results with keys and metadata, including 
                               'Input_Arguments', 'meta_data', and 'selected_groups_with_weights'.
        forecast_dic (dict): Dictionary of forecast data, where each key contains 'Input_Arguments', 
                             'meta_data', and 'predicted_dic'.
        weighted_losses (dict): A dictionary mapping weight keys to their respective loss values, used for 
                                selecting the best weights.
        keys_to_check (list of str): List of keys to match between the weights and forecast dictionaries.
        additional_keys_weights (list of str, optional): Additional keys from 'weights_result' to include 
                                                         in the output. Defaults to an empty list.
        additional_keys_forecast (list of str, optional): Additional keys from 'forecast_dic' to include 
                                                          in the output. Defaults to an empty list.
        verbosity (int, optional): Level of verbosity for debugging purposes. Values >= 5 provide detailed logs. 
                                   Defaults to 0.

    Returns:
        dict: A dictionary where keys are unique matching identifiers and values contain:
            - 'weights_key': The key of the matched weights dictionary.
            - 'forecast_key': The key of the matched forecast dictionary.
            - 'weights_dict': The matched weights dictionary.
            - 'forecast_dict': The matched forecast dictionary.
            - 'forecast_weighted': Weighted forecast values using the given weights.
            - 'forecast_equal_weights': Forecast values using equal weights for all groups.
            - 'matched_values': Dictionary of key-value pairs that matched.
            - 'additional_weights_values': Additional key-value pairs from weights_result.
            - 'additional_forecast_values': Additional key-value pairs from forecast_dic.
    """
    matches = {}

    # Initialize optional parameters if not provided
    if additional_keys_weights is None:
        additional_keys_weights = []
    if additional_keys_forecast is None:
        additional_keys_forecast = []

    # Iterate through weights_result and forecast_dic to find matches
    for weight_key, weight_val in weights_result.items():
        weight_args = weight_val['Input_Arguments']

        # Extract time_limit from metadata and add it to weight_args
        time_limit = weight_val.get('meta_data', {}).get('time_limit')
        if time_limit is not None:
            weight_args['time_limit'] = time_limit

        if verbosity >= 5:
            print("Weighted Input Arguments Keys:")
            print(weight_args.keys())

        for forecast_key, forecast_val in forecast_dic.items():
            forecast_args = forecast_val['Input_Arguments']

            # Extract time_limit from metadata and add it to forecast_args
            time_limit = forecast_val.get('meta_data', {}).get('time_limit')
            if time_limit is not None:
                forecast_args['time_limit'] = time_limit

            if verbosity >= 5:
                print("Forecast Input Arguments Keys:")
                print(forecast_args.keys())

            # Check for matching keys
            all_match = True
            matched_values = {}
            additional_weights_values = {}
            additional_forecast_values = {}

            for key in keys_to_check:
                if weight_args.get(key) == forecast_args.get(key):
                    matched_values[key] = weight_args[key]
                else:
                    all_match = False
                    break

            if all_match:
                # Collect additional keys from weights and forecasts
                for add_key in additional_keys_weights:
                    if add_key in weight_args:
                        additional_weights_values[add_key] = weight_args[add_key]

                for add_key in additional_keys_forecast:
                    if add_key in forecast_args:
                        additional_forecast_values[add_key] = forecast_args[add_key]

                # Create a unique string to identify the match
                match_values = [weight_args.get(key) for key in keys_to_check]
                match_values += [weight_args.get(add_key) for add_key in additional_keys_weights if add_key in weight_args]
                match_values += [forecast_args.get(add_key) for add_key in additional_keys_forecast if add_key in forecast_args]
                base_match_string = "_".join(map(str, match_values))

                # Compute weighted and equal-weighted forecasts
                chosenWeights = weight_val["selected_groups_with_weights"]
                dict_forecast = forecast_val["predicted_dic"]

                if verbosity >= 5:
                    print("Combining weights and forecasts with weighted predictions.")
                forecast_weighted = combine_weights_and_forecasts(chosenWeights, dict_forecast, verbosity, "weighted_pred")

                if verbosity >= 5:
                    print("Combining weights and forecasts with equal weights.")
                equalWeights = create_weighted_tuples(dict_forecast)
                forecast_equal_weights = combine_weights_and_forecasts(equalWeights, dict_forecast, verbosity, "equal_weights_pred")

                # Add or update match entry
                if base_match_string in matches:
                    current_weight_key = matches[base_match_string]['weights_key']
                    current_loss = weighted_losses.get(current_weight_key, float('inf'))
                    new_loss = weighted_losses.get(weight_key, float('inf'))

                    # Update only if the new loss is smaller
                    if new_loss < current_loss:
                        matches[base_match_string] = {
                            'weights_key': weight_key,
                            'forecast_key': forecast_key,
                            'weights_dict': weight_val,
                            'forecast_dict': forecast_val,
                            'forecast_weighted': forecast_weighted,
                            'forecast_equal_weights': forecast_equal_weights,
                            'matched_values': matched_values,
                            'additional_weights_values': additional_weights_values,
                            'additional_forecast_values': additional_forecast_values
                        }
                else:
                    matches[base_match_string] = {
                        'weights_key': weight_key,
                        'forecast_key': forecast_key,
                        'weights_dict': weight_val,
                        'forecast_dict': forecast_val,
                        'forecast_weighted': forecast_weighted,
                        'forecast_equal_weights': forecast_equal_weights,
                        'matched_values': matched_values,
                        'additional_weights_values': additional_weights_values,
                        'additional_forecast_values': additional_forecast_values
                    }

    return matches


def create_weighted_tuples(dict_forecast):
    """
    Creates a list of tuples with equal weights for each key in a forecast dictionary.

    Args:
        dict_forecast (dict): A dictionary containing forecast data.

    Returns:
        list: A list of tuples where each tuple contains a key from the dictionary and its equal weight.
    """
    # Calculate the number of keys in the dictionary
    num_keys = len(dict_forecast)

    # Compute equal weight
    equal_weight = 1 / num_keys if num_keys > 0 else 0

    # Create list of weighted tuples
    weighted_tuples = [(key, equal_weight) for key in dict_forecast.keys()]

    return weighted_tuples
