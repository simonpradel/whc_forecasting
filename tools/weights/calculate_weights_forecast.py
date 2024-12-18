import pickle
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tools.models.AutoGluon import train_autogluon_and_forecast 
from tools.models.AutoTS import train_autots_and_forecast 
from tools.models.AutoARIMA import train_AutoARIMA_and_forecast 
from tools.models.AutoETS import train_AutoETS_and_forecast 
from tools.transformations.transform_aggregated_data import transform_dict_to_long, transform_long_to_dict
from tools.methods.get_function_args import get_function_args
import os
import time
from datetime import datetime

# TensorFlow Logging auf ERROR setzen, um alle nicht notwendigen Ausgaben zu vermeiden
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def calculate_weights_forecast(train_dict, freq, n_splits=3, fold_length = 12, forecast_method="level", model="AutoGluon",  excludeModels=None, includeModels=None, use_best_model=True, saveResults=True, save_path = os.getcwd(), verbosity = 0, time_limit = None, dataset_name = None):
    """
    Calculates forecasts and evaluates multiple subsets of time series data. 
    Results are returned without applying any optimization to the forecasts.

    Parameters:
    - train_dict (dict): Dictionary containing the training data. Keys are tuple identifiers of subsets, and values are dataframes with time series data.
    - freq (str): The frequency of the time series data (e.g., 'D' for daily).
    - n_splits (int): Number of splits for cross-validation (default: 3).
    - forecast_method (str): "level" or "global"
    - model (str): The model to use for forecasting (default: 'AutoGluon').
    - saveResults (bool): If True, save the results as a .pkl file.

    Returns:
    - dict: A dictionary containing:
      - 'all_selected_groups': List of all selected subsets.
      - 'all_selected_group_losses': List of corresponding losses for selected subsets.
      - 'forecast_cache': Cached forecasts for each group.
      - 'hat_Y_test': Consolidated forecasts across groups.
      - 'fold_length': Test size of fold
      - 'n_splits': Number of cross-validation splits.
      - 'forecast_method': Method used for forecast level.
    """

    valid_methods = ["level", "global"]
    if forecast_method not in valid_methods:
        raise ValueError(f"Ungültige forecast_method: {forecast_method}. Zulässige Werte sind: {valid_methods}")

    start_time = time.time()

    top_level_series = train_dict[('dataset',)]
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=fold_length)

    max_variable_combinations = max(len(key) for key in train_dict.keys() if isinstance(key, tuple))
    print(f"Maximal variable combinations: {max_variable_combinations}")

    if  includeModels == None:
        include_models = ""
        best_model_alias = ""
    elif (use_best_model == False) or (isinstance(includeModels, list) and len(includeModels) == 1):
        include_models = includeModels
        best_model_alias = ""        
    else:
        include_models = find_best_model(top_level_series, freq, model=model, excludeModels=excludeModels, includeModels=includeModels, test_period=fold_length, enable_ensemble = False, eval_metric = "MAE", verbosity = verbosity, time_limit = time_limit)
        print(f"Das beste Modell ist: {include_models}")
        if isinstance(include_models, list):
            best_model_alias = include_models[0] + "_"
        else:
            best_model_alias = include_models + "_"

    include_models_alias = includeModels
    if isinstance(include_models_alias, list) and len(include_models_alias) == 1:
        include_models_alias = include_models_alias[0] + "_"
    elif isinstance(include_models_alias, str):
        include_models_alias = include_models_alias + "_"
    else:
        include_models_alias = "selection_" 

    if verbosity > 3:
        print(f"Start forecast with forecast_method: {forecast_method}")

    if forecast_method == "level":
        group_aggregated_forecast_dict = {}
        for level in range(1, max_variable_combinations + 1):
            print(f"\nProcessing level {level} / {max_variable_combinations}")

            current_level_groups = [k for k in train_dict.keys() if isinstance(k, tuple) and len(k) == level]
            print(current_level_groups)
            for group_key in current_level_groups:
                print("group_key")
                print(group_key)
                aggregated_forecast = evaluate_group(train_dict, tscv, freq, model, include_models, n_splits, group_key, verbosity = verbosity, time_limit = time_limit)
                group_aggregated_forecast_dict[group_key] = aggregated_forecast


    elif forecast_method == "global":
        group_aggregated_forecast_dict = evaluate_group(train_dict, tscv, freq, model, include_models, n_splits, group_key=None, verbosity = verbosity, time_limit = time_limit)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    os.makedirs(save_path, exist_ok=True)
    file = f"Forecast_Weights_{model}_{forecast_method}_{include_models_alias}{best_model_alias}{fold_length}_{n_splits}_{time_limit}.pkl"
    pkl_filename = os.path.join(save_path, file) 

    results = {
    "Input_Arguments": get_function_args(calculate_weights_forecast, train_dict, freq, n_splits, fold_length, forecast_method, model, excludeModels, includeModels, use_best_model),
    "meta_data": {
        "elapsed_time": elapsed_time,
        "time_limit": time_limit,
        "date": current_date,
        "include_models_alias": include_models_alias,
        "best_model_alias": best_model_alias,
        "file_name": file,
        "dataset_name": dataset_name
    },
    "aggregated_forecast": group_aggregated_forecast_dict
    }

    if saveResults:

        with open(pkl_filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved as '{pkl_filename}'.")

    return results



# Find and Use Benachmark Model  
def find_best_model(train_df, freq, model="AutoGluon", excludeModels=None, includeModels=None, 
                    test_period=None, enable_ensemble=True, eval_metric="MAE", verbosity=0, time_limit=120 * 60):
    """
    Trains a predictor and determines the best-performing model for the provided dataset.

    Parameters:
    ----------
    train_df (pd.DataFrame): 
        A DataFrame containing the training data. It should include the following columns:
        - "date": The date column for the time series.
        - "ts_id": The time series ID column for identifying different time series.
        - "total": The actual observed values.

    freq (str): 
        The frequency of the time series data. For example:
        - 'D' for daily data
        - 'W' for weekly data

    model (str, optional): 
        The name of the automated model training tool to use. Default is "AutoGluon".
        Supported values: ["AutoGluon", "AutoTS"].

    excludeModels (list of str, optional): 
        A list of model names to exclude during training. Default is None.

    includeModels (list of str, optional): 
        A list of specific model names to include during training. Default is None.

    test_period (int, optional): 
        The number of periods to use as the test set for validation. Default is None.

    enable_ensemble (bool, optional): 
        Whether to enable ensemble model training. Default is True.

    eval_metric (str, optional): 
        The evaluation metric to optimize during model selection. Default is "MAE".
        Example metrics: "RMSE", "MAPE", "MAE".

    verbosity (int, optional): 
        The verbosity level for model training outputs. Default is 0 (silent).

    time_limit (int, optional): 
        The maximum time limit (in seconds) allowed for training models. Default is 120 * 60 (2 hours).

    Returns:
    -------
    str:
        The name of the best-performing model.

    object:
        The trained predictor object.

    Notes:
    -----
    - The function currently supports AutoGluon and AutoTS for automated time series forecasting.
    - If the specified model is not found, an error message will be printed, and no model will be returned.
    """

    # Check which model to use for training
    if model == "AutoGluon":
        # Train the AutoGluon predictor and get the forecasts
        Y_hat_df, best_model = train_autogluon_and_forecast(
            train_df=train_df,
            test_period=test_period,
            freq=freq,
            date_col="date",
            id_col="ts_id",
            actuals_col="total",
            includeModels=includeModels,
            excludeModels=excludeModels,
            set_index=False,
            enable_ensemble=enable_ensemble,
            eval_metric=eval_metric,
            verbosity=verbosity,
            time_limit=time_limit
        )

    elif model == "AutoTS":
        # Train the AutoTS predictor and get the forecasts
        Y_hat_df, best_model = train_autots_and_forecast(
            train_df=train_df,
            test_period=test_period,
            freq=freq,
            date_col="date",
            id_col="ts_id",
            actuals_col="total",
            includeModels=includeModels,
            excludeModels=excludeModels,
            set_index=False,
            enable_ensemble=enable_ensemble,
            eval_metric=eval_metric,
            verbosity=verbosity,
            time_limit=time_limit
        )

    else:
        # Handle unsupported model names
        print(f"Error: The specified model '{model}' is not supported. Supported models are 'AutoGluon' and 'AutoTS'.")
        return None

    # Return the name of the best model
    return best_model




def evaluate_group(train_dict, tscv, freq, model, best_model, n_splits, group_key = None, enable_ensemble = True, eval_metric = "MAE", verbosity = 0, time_limit = 120 * 60):
    """
    Evaluates a specific group of time series data by performing cross-validation.

    Parameters:
    - group_key (tuple): Key of the subset to evaluate.
    - train_dict (dict): Dictionary containing the time series data for all subsets.
    - tscv (TimeSeriesSplit): Cross-validation object.
    - freq (str): Frequency of the time series data (e.g., 'D' for daily).
    - model (str): Forecasting model to use (e.g., 'Prophet').
    - best_model (model): Benchmark model for comparison.
    - n_splits (int): Number of cross-validation splits.

    Returns:
    - tuple: 
      - fold_losses (list): List of losses (Mean Absolute Error) for each fold.
      - group_aggregated_forecast (array): Aggregated forecast across folds.
    """
    if verbosity > 3:
        print("Start evaluate group")

    if(group_key != None):
        method = "level"
    else:
        method = "global"

    top_level_series = train_dict[('dataset',)]

    if method == "level":
        group_aggregated_forecast = {
            key: train_dict[key].groupby('date', as_index=False)['total'].sum() for key in train_dict
        }   
    elif method == "global":
        group_aggregated_forecast = {
            key: train_dict[key].groupby('date', as_index=False)['total'].sum() for key in train_dict
        }

    for fold_idx, (train_index, test_index) in enumerate(tscv.split(top_level_series)):
        print(f"Fold {fold_idx+1} / {n_splits}:")
        train_df_top_level = top_level_series.iloc[train_index]
        train_dates = train_df_top_level['date']
        test_dates = top_level_series['date'].iloc[test_index]

        if(method == "level"):
            train_df = train_dict[group_key]
            mapping = None
        else: 
            mapping, train_df = transform_dict_to_long(dataframes = train_dict, id_col = 'ts_id', date_col = 'date', actuals_col = "total", include_all_columns = True)
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df_filtered = train_df[train_df['date'] <= max(train_dates)]
        forecast = train_with_chosen_model(train_df_filtered, len(test_dates), freq=freq, best_model=best_model, model=model, method = method, mapping = mapping, enable_ensemble = enable_ensemble , eval_metric = eval_metric, verbosity = verbosity, time_limit = time_limit)
        
        if method == "global":
            for key in group_aggregated_forecast:
                if key in forecast:
                    group_aggregated_forecast[key].loc[group_aggregated_forecast[key]['date'].isin(test_dates), 'pred'] = forecast[key]
                     
        elif method == "level":
            print("level Forecast")
            group_aggregated_forecast[group_key].loc[group_aggregated_forecast[group_key]['date'].isin(test_dates), 'pred'] = forecast
    
    if verbosity > 3:
        print("End evaluate group") 
        
    if(method == "level"):
        return group_aggregated_forecast[group_key]
    elif(method == "global"):
        return group_aggregated_forecast
    else:
        print("method has to be global or level")
    
        

def train_with_chosen_model(train_df, test_period, freq, best_model=None, model="AutoGluon", method="level", 
                            mapping=None, enable_ensemble=True, eval_metric="MAE", verbosity=0, time_limit=120 * 60):
    """
    Trains a specified time series forecasting model and generates forecasts for future periods.

    Parameters:
    -----------
    train_df (pd.DataFrame): 
        A DataFrame containing the training data with the following required columns:
        - 'date': The timestamp column.
        - 'ts_id': Identifier for time series (e.g., group or level ID).
        - 'total': The actual observed values.

    test_period (int): 
        The number of future periods to forecast.

    freq (str): 
        The frequency of the data, e.g., 'D' for daily, 'W' for weekly.

    best_model (str, optional): 
        The name of a specific model to be included during training (e.g., from previous results).
        Default is None, which lets the method automatically select the best model.

    model (str, optional): 
        The time series model to be used for training and forecasting. Options include:
        - "AutoGluon"
        - "AutoTS"
        - "AutoETS"
        - "AutoARIMA"
        Default is "AutoGluon".

    method (str, optional): 
        Specifies the aggregation method:
        - "level": Forecasts are generated and aggregated at a single level.
        - "hierarchical": Forecasts are generated and aggregated hierarchically using `mapping`.
        Default is "level".

    mapping (dict, optional): 
        A dictionary used for hierarchical aggregation. Keys should represent higher-level groups,
        and values should contain lower-level identifiers.

    enable_ensemble (bool, optional): 
        Whether to enable ensemble models during forecasting.
        Default is True.

    eval_metric (str, optional): 
        The evaluation metric to be used for model training and selection (e.g., "MAE").
        Default is "MAE".

    verbosity (int, optional): 
        Verbosity level for model training logs. Higher values show more detailed output.
        Default is 0 (minimal logs).

    time_limit (int, optional): 
        Time limit in seconds for training the models.
        Default is 120 * 60 (2 hours).

    Returns:
    --------
    np.array or dict: 
        - If method="level": Returns a numpy array with forecast values aggregated by date.
        - If method="hierarchical": Returns a dictionary of numpy arrays, grouped by keys in `mapping`.
    """
    
    # AutoGluon Model Selection
    if model == "AutoGluon":
        print(f"Current Model: {model}")
        print(f"Using best_model: {best_model}")
        
        # Forecasting at the chosen level
        if method == "level":
            Y_hat_df, best_model = train_autogluon_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                includeModels=best_model, excludeModels=None, set_index=False, time_limit=time_limit
            )
            forecast_values = Y_hat_df.groupby('date')['pred'].sum().to_numpy()
        else:
            Y_hat_df, best_model = train_autogluon_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                includeModels=best_model, excludeModels=None, set_index=False, time_limit=time_limit
            )
            forecast_values = transform_long_to_dict(
                df=Y_hat_df, mapping=mapping, id_col='ts_id', date_col='date', actuals_col="pred"
            )
            # Group each DataFrame in the dictionary by 'date' and sum predictions
            for key, df in forecast_values.items():
                forecast_values[key] = df.groupby('date')['pred'].sum().to_numpy()

    # AutoTS Model Selection
    elif model == "AutoTS":
        if method == "level":
            Y_hat_df, best_model = train_autots_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                includeModels=best_model, excludeModels=None, set_index=False, time_limit=time_limit,
                enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = Y_hat_df.groupby('date')['pred'].sum().to_numpy()
        else:
            Y_hat_df, best_model = train_autots_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                includeModels=best_model, excludeModels=None, set_index=False, time_limit=time_limit,
                enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = transform_long_to_dict(
                df=Y_hat_df, mapping=mapping, id_col='ts_id', date_col='date', actuals_col="pred"
            )
            for key, df in forecast_values.items():
                forecast_values[key] = df.groupby('date')['pred'].sum().to_numpy()

    # AutoETS Model Selection
    elif model == "AutoETS":
        if method == "level":
            Y_hat_df, best_model = train_AutoETS_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                set_index=False, enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = Y_hat_df.groupby('date')['pred'].sum().to_numpy()
        else:
            Y_hat_df, best_model = train_AutoETS_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                set_index=False, enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = transform_long_to_dict(
                df=Y_hat_df, mapping=mapping, id_col='ts_id', date_col='date', actuals_col="pred"
            )
            for key, df in forecast_values.items():
                forecast_values[key] = df.groupby('date')['pred'].sum().to_numpy()

    # AutoARIMA Model Selection
    elif model == "AutoARIMA":
        if method == "level":
            Y_hat_df, best_model = train_AutoARIMA_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                set_index=False, enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = Y_hat_df.groupby('date')['pred'].sum().to_numpy()
        else:
            Y_hat_df, best_model = train_AutoARIMA_and_forecast(
                train_df, test_period, freq, date_col="date", id_col="ts_id", actuals_col="total",
                set_index=False, enable_ensemble=enable_ensemble, eval_metric=eval_metric, verbosity=verbosity
            )
            forecast_values = transform_long_to_dict(
                df=Y_hat_df, mapping=mapping, id_col='ts_id', date_col='date', actuals_col="pred"
            )
            for key, df in forecast_values.items():
                forecast_values[key] = df.groupby('date')['pred'].sum().to_numpy()
    else:
        print("The specified model was not found. Please choose a valid model.")
        return None

    return forecast_values
