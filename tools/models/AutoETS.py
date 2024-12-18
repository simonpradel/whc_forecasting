from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import os

def train_AutoETS_and_forecast(train_df, test_periods, freq, date_col="ds", id_col="unique_id", actuals_col="y", set_index=True, enable_ensemble=True, eval_metric="MAE", verbosity=0, time_limit=60 * 60 * 24, random_seed=123):
    """
    Trains an AutoETS model and generates forecasts along with fitted values for the training period.

    Parameters:
    ----------
    train_df : pd.DataFrame
        A DataFrame containing the training data with columns:
        - 'ds': Date information (datetime format)
        - 'y': The dependent variable (numeric values)
        - A column for time series IDs (default is 'unique_id')
    
    test_periods : int
        The number of future periods for which forecasts are to be generated.

    freq : str
        The frequency of the data. Common options include:
        - 'D' for daily data
        - 'W' for weekly data
        - 'M' for monthly data
        - 'Q' for quarterly data

    date_col : str, optional, default="ds"
        The name of the column in `train_df` that contains the date information. Defaults to 'ds'.
    
    actuals_col : str, optional, default="y"
        The name of the column in `train_df` that contains the dependent variable (actual values). Defaults to 'y'.

    id_col : str, optional, default="unique_id"
        The name of the column in `train_df` that identifies the time series. Defaults to 'unique_id'.

    set_index : bool, optional, default=True
        If True, the time series ID column (`id_col`) will be set as the index of the returned forecast dataframes. If False, the index will not be set.

    enable_ensemble : bool, optional, default=True
        Whether to enable the ensemble method for model training. This argument is currently not used in the function but can be extended if needed.

    eval_metric : str, optional, default="MAE"
        The evaluation metric to be used for model performance. Common choices include "MAE" (Mean Absolute Error), "MSE" (Mean Squared Error), etc.

    verbosity : int, optional, default=0
        The level of verbosity for logging. Higher values provide more detailed logs.

    time_limit : int, optional, default=60 * 60 * 24
        The maximum time in seconds for the training process. The training will stop if it exceeds this time limit.

    random_seed : int, optional, default=123
        The seed value for random number generation to ensure reproducibility.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        1. Forecasted values for the out-of-sample periods (`Y_hat_df_AutoETS`).
        2. Fitted values for the in-sample periods (`Y_fitted_df_AutoETS`).
    """
    
    # Function that is triggered in case of a timeout during training
    def handler(signum, frame):
        raise TimeoutError("Training exceeded the time limit.")

    # Store the original column names to revert later
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}
    
    # Check if the id_col is set as an index column, and reset if necessary
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Rename columns to match the expected format for StatsForecast
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})

    # Keep only the relevant columns for forecasting
    train_df = train_df[['unique_id', 'ds', 'y']]

    # Define the seasonality based on the frequency of the data
    if freq in ["M", "ME"]:
        season_length = 12  # Monthly data, seasonality of 12 months
    elif freq == "D":
        season_length = 7  # Daily data, seasonality of 7 days (weekly cycle)
    elif freq in ["Q", "QE", "QS"]:
        season_length = 4  # Quarterly data, seasonality of 4 quarters

    # Set the environment variable to suppress warnings about ID column
    os.environ["NIXTLA_ID_AS_COL"] = "true"

    # Initialize the AutoETS model
    fcst_AutoETS = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq=freq,
        n_jobs=-1
    )

    # Fit the model on the training data
    fcst_AutoETS = fcst_AutoETS.fit(train_df)

    # Generate forecasts (both in-sample and out-of-sample) by passing the training data to the forecast method
    forecast_result = fcst_AutoETS.forecast(df=train_df, h=test_periods, fitted=True).reset_index()

    # Extract the forecasted (out-of-sample) and fitted (in-sample) values
    last_dates = forecast_result["ds"].drop_duplicates().sort_values().tail(test_periods)
    
    # Out-of-Sample Forecast
    Y_hat_df_AutoETS = forecast_result[forecast_result["ds"].isin(last_dates)].rename(columns={'AutoETS': 'pred'})
    
    # In-Sample Fitted Values
    Y_fitted_df_AutoETS = forecast_result[~forecast_result["ds"].isin(last_dates)].rename(columns={'AutoETS': 'pred'})

    # Revert the column names to their original names
    Y_hat_df_AutoETS = Y_hat_df_AutoETS.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})
    Y_fitted_df_AutoETS = Y_fitted_df_AutoETS.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})

    # Set 'ts_id' as the index if `set_index` is True
    if set_index:
        Y_hat_df_AutoETS.set_index(original_columns['ts_id'], inplace=True)
        Y_fitted_df_AutoETS.set_index(original_columns['ts_id'], inplace=True)

    # Return the forecast and fitted DataFrames
    return Y_hat_df_AutoETS, Y_fitted_df_AutoETS
