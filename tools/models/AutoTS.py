import pandas as pd
from autots import AutoTS
from autots.models.model_list import model_lists

model_list = ['AverageValueNaive',
 'GLS',
 'GLM',
 'ETS',
 'ARIMA',
 'FBProphet',
 'RollingRegression',
 'GluonTS',
 'SeasonalNaive',
 'UnobservedComponents',
 'VECM',
 'DynamicFactor',
 'MotifSimulation',
 'WindowRegression',
 'VAR',
 'DatepartRegression',
 'UnivariateRegression',
 'UnivariateMotif',
 'MultivariateMotif',
 'NVAR',
 'MultivariateRegression',
 'SectionalMotif',
 'Theta',
 'ARDL',
 'NeuralProphet',
 'DynamicFactorMQ',
 'PytorchForecasting',
 'ARCH',
 'RRVAR',
 'MAR',
 'TMF',
 'LATC',
 'KalmanStateSpace',
 'MetricMotif',
 'Cassandra',
 'SeasonalityMotif',
 'MLEnsemble',
 'PreprocessingRegression',
 'FFT',
 'BallTreeMultivariateMotif',
 'TiDE',
 'NeuralForecast',
 'DMD']

presets =  "superfast",  "fast",  "fast_parallel_no_arima", "multivariate", 'probabilistic', "best" "all"
 
def adjust_metric_weights(eval_metric):
    """
    Adjusts the metric weights based on the evaluation metric provided. The function 
    returns a dictionary of weights for different metrics, with a larger weight assigned 
    to the specified evaluation metric.

    Parameters:
    ----------
    eval_metric : str
        The evaluation metric that is used to determine which weight should be increased. 
        It can be one of the following: 'smape', 'mae', 'rmse', 'made', 'mage', 'mle', 
        'imle', 'spl', 'containment', 'contour', 'runtime'. 

    Returns:
    -------
    dict
        A dictionary with metric names as keys and their adjusted weights as values.
        The weight of the specified metric will be increased, while all other weights will 
        be set to 0.
    """
    
    # Default weighting values for various metrics
    metric_weighting = {
        'smape_weighting': 5,
        'mae_weighting': 2,
        'rmse_weighting': 2,
        'made_weighting': 0.5,
        'mage_weighting': 1,
        'mle_weighting': 0,
        'imle_weighting': 0,
        'spl_weighting': 3,
        'containment_weighting': 0,
        'contour_weighting': 1,
        'runtime_weighting': 0.05,
    }
    
    # Initialize all weights to 0
    adjusted_weights = {key: 0 for key in metric_weighting}
    
    # Construct the key for the provided eval_metric (e.g., 'mae_weighting')
    metric_key = eval_metric.lower() + '_weighting'
    
    # If the provided metric exists in the dictionary, increase its weight
    if metric_key in adjusted_weights:
        adjusted_weights[metric_key] = 15
    else:
        # If no matching metric is found, retain default weights
        adjusted_weights = metric_weighting
    
    return adjusted_weights


def train_autots_and_forecast(train_df, test_period, freq, date_col="ds", id_col="unique_id", actuals_col="y", 
                              includeModels=None, excludeModels=None, set_index=True, enable_ensemble=True, 
                              eval_metric="MAE", verbosity=0, time_limit=None, random_seed=123):
    """
    Trains an AutoTS model and generates forecasts along with fitted values for the training period.

    Parameters:
    ----------
    train_df : pd.DataFrame
        Training data with columns 'ds' (date), 'y' (value), and an additional ID column (e.g., 'unique_id').
    test_period : int
        The number of future periods to forecast.
    freq : str
        The frequency of the data, such as 'D' for daily or 'W' for weekly.
    date_col : str, optional
        The name of the column containing date information. Default is 'ds'.
    actuals_col : str, optional
        The name of the column containing the dependent variable. Default is 'y'.
    id_col : str, optional
        The name of the column containing time series IDs. Default is 'unique_id'.
    set_index : bool, optional
        Whether to set the `id_col` as the index. Default is True.
    enable_ensemble : bool, optional
        Whether to enable ensemble models. Default is True.
    eval_metric : str, optional
        The evaluation metric used to adjust the model weights. Default is "MAE".
    verbosity : int, optional
        The level of verbosity for model fitting. Default is 0.
    time_limit : int, optional
        The maximum training time in seconds. If provided, it restricts model training time.
    random_seed : int, optional
        The random seed used for reproducibility. Default is 123.

    Returns:
    -------
    Tuple[pd.DataFrame, list]
        - A DataFrame containing the forecasted values with original column names.
        - A list containing the name of the best model found during training.
    """
    
    # Define a timeout handler function
    def handler(signum, frame):
        raise TimeoutError("Training has exceeded the time limit.")
    
    # Save the original column names for later renaming
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}
    
    # If `id_col` is already an index, reset it to a column
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Prepare the dataframe for AutoTS by renaming columns
    train_df = train_df[[date_col, actuals_col, id_col]]
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})
    
    # Convert the date column to datetime format
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    
    # Adjust model list based on time limit if provided
    if time_limit is not None:
        if time_limit <= 600:  # 10 minutes
            list_full = model_lists["superfast"]
        elif time_limit <= 1800:  # 30 minutes
            list_full = model_lists["fast"]
            fast_no_arima = {
                i: list_full[i]
                for i in list_full
                if i not in [
                    'NVAR', 'UnobservedComponents', 'VECM', 'MAR', 'BallTreeMultivariateMotif', 'WindowRegression'
                ]
            }
            list_full = fast_no_arima
        else:
            list_full = model_lists["all"]
    
    # Determine which models to include based on inclusion/exclusion lists
    if includeModels is None or includeModels == "":
        if excludeModels is not None:
            model_list = [item for item in list_full if item not in excludeModels]
        else:
            model_list = list_full
    else:
        model_list = includeModels
    
    print(f"model_list: {model_list}")
    
    # If there's only one model or ensemble is disabled, don't use ensemble
    if (isinstance(model_list, list) and len(model_list) == 1) or not enable_ensemble:
        ensemble = None
    else:
        ensemble = "simple"
    
    # Adjust weights for the specified evaluation metric
    adjusted_weights = adjust_metric_weights(eval_metric)
    
    # Set generation timeout if time limit is provided
    if time_limit is not None:
        generation_timeout = time_limit / 5  # Divide by 5 to account for maximum generations
        generation_timeout = generation_timeout / 60  # Convert to minutes
    
    try:
        # Initialize and train the AutoTS model
        model = AutoTS(
            forecast_length=test_period,
            frequency=freq,
            ensemble=ensemble,
            max_generations=5,
            num_validations=2,
            model_list=model_list,
            n_jobs='auto',
            verbose=0,
            metric_weighting=adjusted_weights,
            random_seed=random_seed,
            generation_timeout=time_limit
        )
        
        if verbosity > 2:
            print(train_df.info())
        
        # Fit the model with the prepared data
        model = model.fit(train_df, date_col='ds', value_col='y', id_col="unique_id")
    
    except TimeoutError as e:
        print(e)

    # Make forecasts
    prediction = model.predict(forecast_length=test_period, verbose=0)
    forecast = prediction.long_form_results().reset_index()

    # Rename columns back to the original names
    Y_hat_df_AutoTS = forecast.rename(columns={'datetime': original_columns['date'], 
                                               'SeriesID': original_columns['ts_id'], 
                                               'Value': "pred"})
    
    # Filter forecasts for future periods
    Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS[date_col] > train_df["ds"].max()]
    
    # Select the 50% prediction interval and remove the 'PredictionInterval' column
    Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS["PredictionInterval"] == "50%"]
    Y_hat_df_AutoTS = Y_hat_df_AutoTS.drop('PredictionInterval', axis=1)

    # Retrieve the best model
    best_model = [model.best_model_name]
    if verbosity > 2:
        print("Best Model")
        print(best_model)

    # Set `id_col` as the index if `set_index` is True
    if set_index:
        Y_hat_df_AutoTS.set_index(original_columns['ts_id'], inplace=True)

    return Y_hat_df_AutoTS, best_model
