# included standard models in the full default setting of autoGluon: https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/models/presets.py#L109

models = [
    "SimpleFeedForward",
    "DeepAR",
    "DLinear",
    "PatchTST",
    "TemporalFusionTransformer",
    "TiDE",
    "WaveNet",
    "RecursiveTabular",
    "DirectTabular",
    "Average",
    "SeasonalAverage",
    "Naive",
    "SeasonalNaive",
    "Zero",
    "AutoETS",
    "AutoCES",
    "AutoARIMA",
    "DynamicOptimizedTheta",
    "NPTS",
    "Theta",
    "ETS",
    "ADIDA",
    "CrostonSBA",
    "IMAPA",
    "Chronos",
]

# AutoGluon
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


def train_autogluon_and_forecast(train_df, test_period, freq, date_col, id_col, actuals_col, includeModels=None, excludeModels=None, set_index=False, enable_ensemble=True, eval_metric="MAE", verbosity=0, time_limit=60 * 60 * 24, random_seed=123):
    """
    Trains an AutoGluon model on the provided training data and generates forecasts.

    Parameters:
    train_df (pd.DataFrame): The training data containing columns for date, time series ID, and actual values.
    test_period (int): The number of future periods (forecast horizon) to predict.
    freq (str): The frequency of the time series data (e.g., 'D' for daily, 'W' for weekly).
    date_col (str): The name of the column containing the date or timestamp information.
    id_col (str): The name of the column representing the unique time series ID.
    actuals_col (str): The name of the column containing the target (actual values) to predict.
    includeModels (str or list of str, optional): A model or list of models to include in the training (e.g., "AutoARIMA", "AutoETSModel", "DeepAR", "PatchTST", etc.). Defaults to None.
    excludeModels (list of str, optional): A list of models to exclude from the training. Defaults to None.
    set_index (bool, optional): Whether to set the time series ID as the index in the resulting forecast dataframe. Defaults to False.
    enable_ensemble (bool, optional): Whether to enable ensemble learning. Defaults to True.
    eval_metric (str, optional): The evaluation metric to use for model selection (e.g., "MAE", "RMSE"). Defaults to "MAE".
    verbosity (int, optional): Level of verbosity for logging (0 = no output, higher values = more output). Defaults to 0.
    time_limit (int, optional): The maximum amount of time (in seconds) allowed for model training. Defaults to 86400 seconds (1 day).
    random_seed (int, optional): The random seed for reproducibility. Defaults to 123.

    Returns:
    pd.DataFrame: A dataframe containing the forecasted values with original column names.
    str: The name of the best model as selected by the leaderboard.
    """
    # Save the original column names for later reference
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}
    
    # Check if id_col is already set as an index and reset it if necessary
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Rename columns to AutoGluon-friendly names
    train_df = train_df.rename(columns={date_col: 'timestamp', actuals_col: 'target', id_col: 'item_id'})

    # Ensure 'target' column is of type float and 'timestamp' is in datetime format
    train_df['target'] = train_df['target'].astype(float)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

    # Identify and extract static features (columns other than 'timestamp' and 'target')
    remaining_cols = train_df.drop(columns=['timestamp', 'target']).columns
    static_features_df = train_df[remaining_cols].drop_duplicates()
    train_df = train_df[['item_id', 'timestamp', 'target']]

    # Convert the DataFrame to AutoGluon's TimeSeriesDataFrame format
    train_data_AutoGluon = TimeSeriesDataFrame.from_data_frame(
        train_df, id_column='item_id', timestamp_column='timestamp', static_features_df=static_features_df
    )

    # Adjust frequency for "M" (monthly) and "ME" (monthly end) if necessary
    if freq in ["M", "ME"]:
        if freq == "M":
            freq = "ME"

    # Initialize the AutoGluon TimeSeriesPredictor
    predictor = TimeSeriesPredictor(
        prediction_length=test_period,
        freq=freq,
        target='target',
        eval_metric=eval_metric,
        verbosity=verbosity,
        log_to_file=False,
        quantile_levels=[0.05, 0.5, 0.95]
    )

    # Print the selected models for training
    print(f"model_list: {includeModels}")

    # Ensure excludeModels is a list, even if it's provided as a string
    if excludeModels is None:
        excludeModels = []
    if isinstance(excludeModels, str):
        excludeModels = [excludeModels]
    
    # Handle the inclusion of specific models
    if includeModels is None or includeModels == "":
        print("Start fitting models with default preset.")
        predictor.fit(
            train_data_AutoGluon, 
            presets="high_quality", 
            excluded_model_types=excludeModels,
            enable_ensemble=enable_ensemble,
            time_limit=time_limit,
            random_seed=random_seed
        )
    elif includeModels in ["fast_training", "medium_quality", "good_quality", "high_quality", "best_quality"]:
        print(f"Start fitting models with {includeModels} preset.")
        predictor.fit(
            train_data_AutoGluon, 
            presets=includeModels, 
            excluded_model_types=excludeModels,
            enable_ensemble=enable_ensemble
        )
    else:
        # If specific models are provided, use best_quality preset and customize the hyperparameters
        if isinstance(includeModels, str):
            includeModels = [includeModels]
        model_dict = {model: {} for model in includeModels}

        print("Start fitting custom models with best quality preset.")
        predictor.fit(
            train_data_AutoGluon, 
            presets="best_quality",
            hyperparameters=model_dict,
            excluded_model_types=excludeModels,
            enable_ensemble=enable_ensemble
        )

    # Generate the forecast
    forecast = predictor.predict(train_data_AutoGluon)

    # Retrieve the leaderboard to identify the best performing model
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard['model'][0]

    # Reset index and rename columns to match the original names
    forecast = forecast.reset_index().rename(columns={'timestamp': original_columns['date'], 'item_id': original_columns['ts_id'], 'mean': 'pred'})

    # Set 'ts_id' as index if requested
    if set_index:
        forecast.set_index(original_columns['ts_id'], inplace=True)
        Y_hat_df = forecast[[original_columns['date'], 'pred']]
    else:
        forecast.reset_index(inplace=True)
        Y_hat_df = forecast[[original_columns['date'], original_columns['ts_id'], 'pred']]

    return Y_hat_df, best_model

