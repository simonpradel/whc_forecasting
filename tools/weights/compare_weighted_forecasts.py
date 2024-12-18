def compare_weighted_forecasts(files_dict):
    """
    Processes files, extracts forecasts, and calculates the weighted loss.

    Args:
        files_dict (dict): A dictionary where keys are file names and values are dictionaries containing the content
                           of each file. Each file content is expected to have:
                           - 'selected_groups_with_weights': A list of tuples, where each tuple contains a group key and its associated weight.
                           - 'aggregated_forecast': A dictionary where keys are group names and values are data frames containing 'pred' (forecast) and 'total' (actual values).

    Returns:
        tuple: A tuple containing two sorted dictionaries:
            1. weighted_losses (dict): Dictionary where keys are file names and values are the calculated loss (MAE) for each file, sorted by loss.
            2. detailed_results (dict): Dictionary where keys are file names and values are dictionaries containing detailed results:
               - 'groups_weights': A list of tuples with group names and their associated weights.
               - 'loss': The calculated loss (MAE) for each file, sorted by loss.
    """
    weighted_losses = {}
    detailed_results = {}

    for file_name, file_content in files_dict.items():
        try:
            # Extract selected groups and their weights, and aggregated forecast data
            selected_groups_with_weights = file_content.get('selected_groups_with_weights', [])
            aggregated_forecast = file_content.get('aggregated_forecast', {})

            # Filter the keys in 'aggregated_forecast' based on 'selected_groups_with_weights'
            filtered_forecasts = {
                key: value for key, value in aggregated_forecast.items()
                if any(group[0] == key for group in selected_groups_with_weights)
            }

            combined_preds = 0
            combined_totals = 0
            groups_weights = []  # Stores group names and their associated weights

            # Iterate over the filtered forecasts
            for key, df in filtered_forecasts.items():
                # Filter the DataFrame to include rows where 'pred' is not 'na'
                df_filtered = df[df['pred'].notna()]

                # Extract predictions ('pred') and actual totals ('total')
                preds = df_filtered['pred'].to_numpy()
                totals = df_filtered['total'].to_numpy()

                # Get the weight for the current group
                weight = next(group[1] for group in selected_groups_with_weights if group[0] == key)

                # Combine predictions and totals according to the weight
                combined_preds += preds * weight
                combined_totals += totals * weight

                # Store the group name and its weight
                groups_weights.append((key, weight))

            # Calculate the loss (Mean Absolute Error)
            loss = (abs(combined_preds - combined_totals)).mean()

            # Store the weighted loss for the current file
            weighted_losses[file_name] = loss

            # Store detailed results (groups, weights, and loss) for the current file
            detailed_results[file_name] = {
                "groups_weights": groups_weights,
                "loss": loss
            }

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Sort weighted_losses and detailed_results by the loss (MAE)
    sorted_weighted_losses = dict(sorted(weighted_losses.items(), key=lambda x: x[1]))
    sorted_detailed_results = dict(sorted(detailed_results.items(), key=lambda x: x[1]['loss']))

    return sorted_weighted_losses, sorted_detailed_results
