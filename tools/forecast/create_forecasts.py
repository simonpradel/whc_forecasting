import os
import pickle
from datetime import datetime
import time
from tools.models.AutoGluon import train_autogluon_and_forecast 
from tools.models.AutoTS import train_autots_and_forecast 
from tools.models.AutoARIMA import train_AutoARIMA_and_forecast 
from tools.models.AutoETS import train_AutoETS_and_forecast 
from tools.transformations.transform_aggregated_data import transform_long_to_dict, transform_dict_to_long
from tools.methods.get_function_args import get_function_args


def create_forecasts(
    train_dic, test_dic=None, future_periods=12, freq="D", model="AutoGluon", 
    train_end_date=None, includeModels=None, excludeModels=None, 
    forecast_method="global", saveResults=True, save_path=os.getcwd(), 
    time_limit=None, verbosity=0
):
    """
    Generates forecasts for specified time series groups and updates the test dataset with forecasted values. 
    This function supports both global and local forecasting methods and multiple forecasting models.

    Parameters:
    ----------
    train_dic : dict
        A dictionary containing time series data for training, structured with hierarchical keys and DataFrame values. 
        Each DataFrame should include at least 'date' and 'total' columns.
    test_dic : dict, optional
        A dictionary of DataFrames containing the testing datasets. If provided, forecasted values will be merged 
        into these DataFrames based on 'date' and hierarchical keys. Defaults to None.
    future_periods : int, optional
        Number of future periods for which forecasts should be generated. Defaults to 12.
    freq : str, optional
        The frequency of the time series, e.g., 'D' for daily or 'M' for monthly. Defaults to 'D'.
    model : str, optional
        The forecasting model to use. Supported values include 'AutoGluon', 'AutoTS', 'AutoARIMA', and 'AutoETS'. Defaults to 'AutoGluon'.
    train_end_date : str, optional
        The last date of the training data to restrict the training period. Format: 'YYYY-MM-DD'. Defaults to None.
    includeModels : list, optional
        A list of specific models to include in the AutoML process. Relevant only for AutoGluon and AutoTS. Defaults to None.
    excludeModels : list, optional
        A list of specific models to exclude from the AutoML process. Relevant only for AutoGluon and AutoTS. Defaults to None.
    forecast_method : str, optional
        The forecasting approach to use: "global" for a single model across all series or "local" for individual models per series. Defaults to "global".
    saveResults : bool, optional
        If True, saves the forecast results as a `.pkl` file to the specified `save_path`. Defaults to True.
    save_path : str, optional
        Directory path where results will be saved if `saveResults` is True. Defaults to the current working directory.
    time_limit : int, optional
        Time limit in seconds for training models. Relevant for AutoGluon and AutoTS. Defaults to None.
    verbosity : int, optional
        Level of verbosity for logging during model training. Higher values produce more detailed output. Defaults to 0.

    Returns:
    -------
    dict
        An updated version of `test_dic` or a new dictionary with forecasted values merged into the original data. 
        Keys correspond to the original hierarchical keys in `train_dic` or `test_dic`.
    """

    # Start timing to measure the execution time of the function
    start_time = time.time()

    # Initialize the predicted_dic with a copy of test_dic if it exists; otherwise, create an empty dictionary
    if test_dic is not None:
        predicted_dic = test_dic.copy()
    else:
        predicted_dic = {}

    # Placeholder to store the best model (if applicable)
    best_model = None

    # If the forecast method is "global", combine all series into a single DataFrame for joint forecasting
    if forecast_method == "global":
        # Convert the hierarchical dictionary to a long-format DataFrame for global modeling
        mapping, df_long = transform_dict_to_long(
            train_dic, id_col='ts_id', date_col='date', actuals_col="total"
        )
        print("Starting global forecast...")
        
        # Choose the forecasting model based on the specified argument
        if model == "AutoTS":
            print(df_long.info())  # Debugging: Display information about the long-format DataFrame
            predicted_values, best_model = train_autots_and_forecast(
                df_long, future_periods, freq=freq, includeModels=includeModels,
                excludeModels=excludeModels, date_col="date", actuals_col="total", 
                id_col="ts_id", set_index=False, time_limit=time_limit
            )
        elif model == "AutoGluon":
            predicted_values, best_model = train_autogluon_and_forecast(
                df_long, future_periods, freq=freq, includeModels=includeModels,
                excludeModels=excludeModels, date_col="date", actuals_col="total", 
                id_col="ts_id", set_index=False, time_limit=time_limit
            )
        elif model == "AutoETS":
            # Train and forecast using the ETS model
            predicted_values, fitted_values = train_AutoETS_and_forecast(
                df_long, future_periods, freq=freq, date_col="date", id_col="ts_id", 
                actuals_col="total", set_index=False, verbosity=verbosity
            )
        elif model == "AutoARIMA":
            # Train and forecast using the ARIMA model
            predicted_values, fitted_values = train_AutoARIMA_and_forecast(
                df_long, future_periods, freq=freq, date_col="date", id_col="ts_id", 
                actuals_col="total", set_index=False, verbosity=verbosity
            )
        
        print("Global forecast completed.")

        # Convert the forecasted values back to a dictionary format using the original mapping
        predicted_dic = transform_long_to_dict(
            predicted_values, mapping=mapping, id_col='ts_id', date_col='date', 
            actuals_col="pred"
        )

        if test_dic is not None:
            # Merge the forecasted values into the existing test data for each group key
            for key, df in predicted_dic.items():
                columns_to_fill = list(key)  # Extract the key components as columns
                columns_to_fill.append('date')
                predicted_dic[key] = test_dic[key].merge(
                    df, on=columns_to_fill, how='outer'
                )
                # Forward-fill the 'ts_id' column to ensure continuity across grouped data
                if 'ts_id' in predicted_dic[key].columns:
                    predicted_dic[key]['ts_id'] = predicted_dic[key].groupby(list(key))['ts_id'].ffill()
        else:
            # Handle cases where test_dic is not provided
            for group_key, df in predicted_dic.items():
                relevant_dates = df['date'].unique()  # Identify forecasted dates
                columns_to_fill = list(group_key)  # Extract the group key as columns
                columns_to_fill.append('date')
                predicted_dic[group_key] = train_dic[group_key].merge(
                    df, on=columns_to_fill, how='outer'
                )
                if 'ts_id' in predicted_dic[group_key].columns:
                    predicted_dic[group_key]['ts_id'] = predicted_dic[group_key].groupby(list(group_key))['ts_id'].ffill()
                # Retain only the forecasted dates
                predicted_dic[group_key] = predicted_dic[group_key][
                    predicted_dic[group_key]['date'].isin(relevant_dates)
                ]
    # Handling the forecast for level-specific groups
    elif forecast_method == "level":
        print("Starting level-specific forecast for each group...")
    
        total_iterations = len(train_dic.keys())  # Total number of groups to iterate over
        current_iteration = 0  # Initialize the current iteration counter
    
        for key, df in train_dic.items():
            current_iteration += 1
            print(f"Processing group {current_iteration}/{total_iterations}: {key}")
            df = df.copy()  # Create a copy of the dataframe to avoid modifying the original
    
            # If a training end date is provided, filter the data up to that date
            if train_end_date is not None:
                df = df[df['date'] <= train_end_date]
    
            # Perform forecasting based on the selected model
            if model == "AutoTS":
                predicted_values, fitted_values = train_autots_and_forecast(
                    df, future_periods, freq=freq, includeModels=includeModels,
                    excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit=time_limit
                )
            elif model == "AutoGluon":
                predicted_values, fitted_values = train_autogluon_and_forecast(
                    df, future_periods, freq=freq, includeModels=includeModels,
                    excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit=time_limit
                )
            elif model == "AutoETS":
                predicted_values, fitted_values = train_AutoETS_and_forecast(
                    df, future_periods, freq=freq, date_col="date", id_col="ts_id", actuals_col="total", set_index=False, verbosity=verbosity
                )
            elif model == "AutoARIMA":
                predicted_values, fitted_values = train_AutoARIMA_and_forecast(
                    df, future_periods, freq=freq, date_col="date", id_col="ts_id", actuals_col="total", set_index=False, verbosity=verbosity
                )
    
            # Store the predicted values for each group
            predicted_dic[key] = predicted_values
    
            # If test data is provided, merge predictions with actual test data
            if test_dic is not None:
                predicted_dic[key] = predicted_dic[key].merge(test_dic[key], on=['ts_id', 'date'], how='outer')
    
                # Forward-fill columns based on 'ts_id' to ensure continuity in the time series
                for col in key:
                    if col in predicted_dic[key].columns:
                        predicted_dic[key][col] = predicted_dic[key].groupby('ts_id')[col].ffill()
            else:
                # If no test data is provided, merge the predictions with the training data
                relevant_dates = predicted_dic[key]['date'].unique()  # Retain only relevant dates from predicted values
                predicted_dic[key] = predicted_dic[key].merge(train_dic[key], on=['ts_id', 'date'], how='outer')
    
                # Forward-fill columns based on 'ts_id' to ensure continuity in the time series
                for col in key:
                    if col in predicted_dic[key].columns:
                        predicted_dic[key][col] = predicted_dic[key].groupby('ts_id')[col].ffill()
    
                # Filter to keep only the relevant dates
                predicted_dic[key] = predicted_dic[key][predicted_dic[key]['date'].isin(relevant_dates)]
    
    # Process results after all forecasts are completed
    if train_end_date is None:
        train_end_date = max([df['date'].max() for df in train_dic.values()])
    else:
        train_end_date = train_end_date  # If a train end date is provided, use it
    
    if test_dic is not None:
        test_end_date = max([df['date'].max() for df in test_dic.values()])  # Determine the latest test date
    else:
        test_end_date = None

    out_of_sample = ""
    if test_dic is None:
        test_end_date = None
        out_of_sample = "OOS_"
        
    # Calculate and display execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Convert elapsed time into hours and minutes
    hours, rem = divmod(elapsed_time, 3600)
    minutes = rem // 60
    
    # Format the elapsed time as "hours:minutes"
    elapsed_time_str = f"{int(hours)}:{int(minutes):02d} h"
    print(f"Total execution time: {elapsed_time_str}")
    
    # Store the current date for result metadata
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Handle the 'includeModels' argument to ensure proper naming in the output file
    if isinstance(includeModels, str):
        includeModels = [includeModels]  # If a single model is provided, convert to a list
    
    if includeModels and len(includeModels) == 1:
        model_suffix = includeModels[0]  # If only one model is specified, add it to the file name
        file_name = f"{out_of_sample}Forecast_{model}_{model_suffix}_{forecast_method}_{future_periods}_{freq}_{time_limit}.pkl"
    else:
        file_name = f"{out_of_sample}Forecast_{model}_{forecast_method}_{future_periods}_{freq}_{time_limit}.pkl"
    
    # Prepare the combined result dictionary
    combined_result = {
        "predicted_dic": predicted_dic,
        "meta_data": {
            "Train_end_date": train_end_date,
            "Test_end_date": test_end_date,
            "elapsed_time": elapsed_time,
            "time_limit": time_limit,
            "date": current_date,
            "file_name": file_name,
            "best_model": best_model
        },
        "Input_Arguments": get_function_args(
            create_forecasts, train_dic=train_dic, test_dic=test_dic, future_periods=future_periods, freq=freq, model=model,
            train_end_date=train_end_date, includeModels=includeModels, excludeModels=excludeModels, forecast_method=forecast_method, saveResults=saveResults, save_path=save_path
        )
    }
    
    # Save the results to a file if 'saveResults' is True
    if saveResults:
        # Ensure the target directory exists
        os.makedirs(save_path, exist_ok=True)
    
        # Create the full path for the output file
        pkl_filename = os.path.join(save_path, file_name)
    
        # Save the combined result to a pickle file
        with open(pkl_filename, 'wb') as f:
            pickle.dump(combined_result, f)
        print(f"Results saved in '{pkl_filename}' as a pkl file.")
    
    # Return the combined result
    return combined_result
