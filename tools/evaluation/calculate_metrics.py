import pandas as pd

def calculate_metric(df, column, metric, verbosity, test_period, future_periods):
    """
    Calculates the specified metric for evaluating forecast accuracy.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the actual values and predictions.
    - column (str): The column name containing the forecasted values.
    - metric (str): The name of the metric to calculate. Supported metrics: 'MASE', 'RMSSE', 'SMAPE', 'WAPE'.
    - verbosity (int): Level of logging detail. Higher values provide more detailed logs.
    - test_period (int): Number of periods used for testing (validation).
    - future_periods (int): Number of forecasted periods in the DataFrame.

    Returns:
    - float: The calculated value of the metric.

    Raises:
    - ValueError: If an unknown metric is specified or if data consistency checks fail.
    """
    df = df.copy()  # Work on a copy of the DataFrame to avoid modifying the original

    # Extract actual values and forecasted column
    df_actuals = df[['y', column]]

    # Split the data into training and testing based on forecast horizon
    y_train = df_actuals['y'].iloc[:-future_periods]

    # Determine out-of-sample predictions based on test period
    out_of_sample_predictions = future_periods - test_period

    # Select the last rows corresponding to the forecast periods
    df_pred_inc_oos = df_actuals.iloc[-future_periods:]
    print(df_pred_inc_oos)

    # Define y_test and y_pred based on whether there are out-of-sample predictions
    if out_of_sample_predictions:
        y_test = df_pred_inc_oos['y'][:-out_of_sample_predictions]
        y_pred = df_pred_inc_oos[column][:-out_of_sample_predictions]
    else:
        y_test = df_pred_inc_oos['y']
        y_pred = df_pred_inc_oos[column]

    # Remove rows with NaN values in y_pred
    non_nan_indices = y_pred.dropna().index
    y_test = y_test.loc[non_nan_indices]
    y_pred = y_pred.loc[non_nan_indices]

    # Remove rows with NaN values in y_test
    non_nan_indices = y_test.dropna().index
    y_test = y_test.loc[non_nan_indices]
    y_pred = y_pred.loc[non_nan_indices]

    # Ensure y_pred and y_test are of equal length and have matching indices
    if verbosity >= 3:
        if len(y_pred) == len(y_test):
            if not all(y_pred.index == y_test.index):
                raise ValueError("y_pred and y_test have matching lengths but non-matching indices.")
        else:
            raise ValueError("y_pred and y_test are not of the same length.")

        if verbosity >= 5:
            print("metric:\n", metric)
            print("y_train:\n", y_train)
            print("y_test:\n", y_test)
            print("y_pred:\n", y_pred)

    # Calculate the specified metric
    if metric == 'MASE':
        from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
        results = MeanAbsoluteScaledError()
        return results(y_test, y_pred, y_train=y_train)

    elif metric == 'RMSSE':
        from sktime.performance_metrics.forecasting import MeanSquaredScaledError
        results = MeanSquaredScaledError(square_root=True)
        return results(y_test, y_pred, y_train=y_train)

    elif metric == 'SMAPE':
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        results = MeanAbsolutePercentageError(symmetric=True)
        return results(y_test, y_pred)

    elif metric == 'WAPE':
        absolute_deviation = abs(y_test - y_pred)
        return absolute_deviation.sum() / abs(y_test).sum()

    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_metrics(final_result, method_list, metrics, dataset_name, date_column='date', actual_column='y', verbosity=0, test_period=None, future_periods=None):
    """
    Computes multiple forecast evaluation metrics across different methods and models.

    Parameters:
    - final_result (dict): Dictionary containing the forecasting results for each model.
    - method_list (list): List of methods (columns) containing forecasted values to evaluate.
    - metrics (list): List of metrics to calculate for each method. Supported: 'MASE', 'RMSSE', 'SMAPE', 'WAPE'.
    - dataset_name (str): Name of the dataset being evaluated.
    - date_column (str, optional): Column name for date values. Defaults to 'date'.
    - actual_column (str, optional): Column name for actual values. Defaults to 'y'.
    - verbosity (int, optional): Level of logging detail. Defaults to 0.
    - test_period (int, optional): Number of periods used for testing (validation).
    - future_periods (int, optional): Number of forecasted periods in the dataset.

    Returns:
    - pd.DataFrame: A DataFrame containing the calculated metrics for each model and method.
    """
    results = []
    models = list(final_result.keys())

    if verbosity > 3:
        print("/fct: calculate_metrics:")
        print(models)

    for model in models:
        # Extract optimization and forecast method details
        optim_method = final_result[model]["weights_dict"]["Input_Arguments"].get('optim_method', None)
        forecast_method = final_result[model]["weights_dict"]["Input_Arguments"].get('forecast_method', None)
        forecast_model = final_result[model]["weights_dict"]["Input_Arguments"].get('model', None)
        elapsed_time = (final_result[model]['weights_dict']["meta_data"].get('elapsed_time', None) +
                        final_result[model]["forecast_dict"]["meta_data"].get('elapsed_time', None))
        time_limit = final_result[model]['weights_dict']["meta_data"].get('time_limit', None)

        model_data = final_result[model]["combined_results"][('dataset',)]

        for method in method_list:
            for metric in metrics:
                if verbosity >= 5:
                    print("calculate_metric:", metric)
                metric_value = calculate_metric(model_data, method, metric, verbosity, test_period, future_periods)
                if verbosity >= 5:
                    print("calculate_metric End:", metric, metric_value)

                dataset_type = "private" if dataset_name in (
                    "hardware_revenue", "mobile_service_revenue", "fbb_fixed_other_revenue",
                    "cos", "commercial_costs", "bad_debt", "non_commercial_costs", "non_recurrent_income_cost"
                ) else "public"

                results.append({
                    'Model': model,
                    'optim_method': optim_method,
                    'forecast_method': forecast_method,
                    'forecast_model': forecast_model,
                    'elapsed_time': elapsed_time,
                    'method': method,
                    'metric': metric,
                    'dataset': dataset_name,
                    'dataset_type': dataset_type,
                    'time_limit': time_limit,
                    'value': metric_value
                })

    return pd.DataFrame(results)

