def run_model_aggregation(
    save_intermediate_results_path, 
    save_final_results_path, 
    dataset_name, 
    model=["AutoETS", "AutoArima"], 
    forecast_method=["global"], 
    use_best_model=False, 
    time_limit=60 * 10, 
    verbosity=5, 
    test_period=6, 
    includeModels=None, 
    excludeModels=None, 
    fold_length=6, 
    used_period_for_cv=0.45, 
    include_groups=["PL_line", "Segment"], 
    optim_method=["ensemble_selection", "optimize_nnls", "differential_evolution"], 
    remove_groups=[False], 
    future_periods=12,   
    use_test_data=True, 
    cutoff_lag=None,
    cutoff_date=None,
    reduceCompTime=True, 
    delete_weights_folder=True,
    delete_forecast_folder=True,
    RERUN_calculate_weights=False,
    RERUN_calculate_forecast=False,
    save_data = False, 
):
    """
    Perform model aggregation and save the results without returning a value.

    Args:
        save_intermediate_results_path (str): Path to save intermediate results like weights and forecasts.
        save_final_results_path (str): Path to save the final aggregated results.
        dataset_name (str): Name of the dataset to load and process.
        model (list, optional): List of forecasting models to use. Defaults to ["AutoETS", "AutoArima"].
        forecast_method (list, optional): Forecasting methods to apply (e.g., "global", "local"). Defaults to ["global"].
        use_best_model (bool, optional): Whether to use the best-performing model. Defaults to False.
        time_limit (int, optional): Time limit for model training in seconds. Defaults to 600.
        verbosity (int, optional): Verbosity level for logging. Defaults to 5.
        test_period (int, optional): Length of the test period for train/test split. Defaults to 6.
        includeModels (list, optional): List of models to include. Defaults to None.
        excludeModels (list, optional): List of models to exclude. Defaults to None.
        fold_length (int, optional): Length of folds for cross-validation. Defaults to 6.
        used_period_for_cv (float, optional): Proportion of the dataset used for cross-validation. Defaults to 0.45.
        include_groups (list, optional): Groups to include in optimization or weighting. Defaults to ["PL_line", "Segment"].
        optim_method (list, optional): Optimization methods to use for weight calculation. Defaults to ["ensemble_selection", "optimize_nnls", "differential_evolution"].
        remove_groups (list, optional): Whether to remove groups during optimization. Defaults to [False].
        future_periods (int, optional): Number of future periods to forecast. Defaults to 12.
        use_test_data (bool, optional): Whether to use test data in forecasting. Defaults to True.
        cutoff_lag (int, optional): Minimum number of days since the last date in the dataset. Defaults to None.
        cutoff_date (str, optional): Explicit cutoff date for data processing. Defaults to None.
        reduceCompTime (bool, optional): Whether to reduce computational time by limiting configurations. Defaults to True.
        delete_weights_folder (bool, optional): Whether to delete the weights folder before calculation. Defaults to True.
        delete_forecast_folder (bool, optional): Whether to delete the forecast folder before calculation. Defaults to True.
        RERUN_calculate_weights (bool, optional): Whether to recalculate weights. Defaults to False.
        RERUN_calculate_forecast (bool, optional): Whether to recalculate forecasts. Defaults to False.
        save_data (bool. optional): If true, it overwrites the stored data. Default to False.
    """
    
    import os    
    import itertools
    import pickle
    from tools.load_data.load_data_from_catalog import load_data_from_catalog
    from tools.transformations.prepare_data import prepare_data
    from tools.transformations.aggregate_by_level import aggregate_by_levels
    from tools.methods.split_data import split_data
    
    ###############################################################################################
    # Prepare folder paths for saving intermediate results
    ###############################################################################################
    weights_path = os.path.join(save_intermediate_results_path, "weights")
    forecasts_path = os.path.join(save_intermediate_results_path, "forecasts")
    
    ###############################################################################################
    # Prepare data by loading, transforming, and aggregating
    ###############################################################################################
    # Define the pickle filename based on dataset name
    pkl_filename = f"data_{dataset_name}.pkl"
    pkl_filename = os.path.join(save_intermediate_results_path, pkl_filename)  # Create full path
    
    # Load data from the catalog
    if verbosity >= 1:
        print("Loading data from catalog...")
    load_data = load_data_from_catalog(dataset_name, maindir=None)
    
    # Prepare the loaded data (e.g., filling missing rows)
    if verbosity >= 1:
        print("Preparing data...")
    data = prepare_data(load_data, cutoff_date=cutoff_date, fill_missing_rows=True)
    
    # Aggregate data by levels
    if verbosity >= 1:
        print("Aggregating data by levels...")
    aggregated_data_dict = aggregate_by_levels(data=data, method='dictionary', show_dict_infos=False)
    
    # Delete the original dataset to save memory
    del data['original_dataset']
    
    # Split the aggregated data into train and test datasets
    train_dic, test_dic = split_data(aggregated_data_dict, period=test_period, unique_id="ts_id", format="dictionary")
           
    ###############################################################################################
    # Save the prepared data into a pickle file
    ###############################################################################################
    # Check if the dataset folder exists, if not, create it
    pkl_directory = os.path.dirname(os.path.join(save_intermediate_results_path, dataset_name))
    
    if not os.path.exists(pkl_directory):
        os.makedirs(pkl_directory)
    
    # Save data if requested
    if save_data:
        with open(pkl_filename, 'wb') as f:
            pickle.dump({
                'data': data.copy(),
                'train_dic': train_dic,
                'test_dic': test_dic
            }, f)
        if verbosity >= 1:
            print(f"Data saved as '{pkl_filename}'.")
    
    ###############################################################################################
    # Calculate stepwise Forecasts before weighting
    ###############################################################################################
    
    def ensure_iterable(value):
        """
        Ensures that the input is returned as an iterable (list or tuple). If the input is None, a list with None is returned.
        Arguments:
        value (any): The value to be converted into an iterable. Can be any type including None, list, or tuple.
    
        Returns:
        iterable: A list or tuple containing the input value(s).
        """
        if value is None: 
            return [None]
        elif isinstance(value, (list, tuple)):  
            return value
        else:  
            return [value]
        
    # Delete the weights folder if specified
    if delete_weights_folder:
        import shutil
        if os.path.exists(weights_path):
            shutil.rmtree(weights_path)
                
    # Calculate new weights if True, otherwise use the pre-calculated ones
    if RERUN_calculate_weights:
        from tools.weights.calculate_weights_forecast import calculate_weights_forecast
        import math
    
        # Set parameters for model training and forecasting
        n_splits = math.floor(len(train_dic[('dataset',)]) * used_period_for_cv / fold_length)
    
        # Ensure the provided model, forecast method, and other parameters are iterable
        model = ensure_iterable(model)
        forecast_method = ensure_iterable(forecast_method)
        excludeModels = ensure_iterable(excludeModels)
        includeModels = ensure_iterable(includeModels)
        use_best_model = ensure_iterable(use_best_model)
        time_limit = ensure_iterable(time_limit)
    
        # Create all possible configurations by combining the model and method parameters
        configurations = list(itertools.product(
            model,
            forecast_method,
            excludeModels,
            includeModels,
            use_best_model,
            time_limit
        ))
    
        # Reduce computation time by removing unnecessary configurations
        if reduceCompTime:
            configurations = [
                (mdl, f_method, excl_models, incl_models, best_model, t_limit) 
                for mdl, f_method, excl_models, incl_models, best_model, t_limit in configurations
                if not (
                    (f_method == "level" and mdl in ["AutoETS", "AutoARIMA"]) or
                    (t_limit == 1800 and mdl in ["AutoETS", "AutoARIMA"]) or
                    (f_method == "level" and t_limit == 1800)
                )
            ]
    
        if verbosity >= 3:
            print("Forecast configurations for weights:")
            print(configurations)
    
        weights_forecast = {}
            
        # Iterate through all configurations and calculate the weights
        for mdl, f_method, excl_models, incl_models, best_model, t_limit in configurations:
            try:
                result = calculate_weights_forecast(
                    train_dict=train_dic,  # Dictionary containing training data
                    freq=data["freq"],  # Frequency of the time series data
                    n_splits=n_splits,  # Number of splits for cross-validation
                    fold_length=fold_length,  # Length of each fold for cross-validation
                    forecast_method=f_method,  # Method to use for forecasting (e.g., 'level', 'global')
                    model=mdl,  # Model type to use for forecasting (e.g., 'AutoARIMA', 'AutoETS')
                    excludeModels=excl_models,  # List of models to exclude from training
                    includeModels=incl_models,  # List of models to include for training
                    use_best_model=best_model,  # Whether to use the best model during forecasting
                    saveResults=True,  # Flag to indicate whether to save results
                    save_path=weights_path,  # Path to save the weights
                    verbosity=4,  # Verbosity level for logging
                    time_limit=t_limit  # Time limit for the forecast (in seconds)
                )
                # Store the forecast results using the file name as the key
                weights_forecast[result["meta_data"]['file_name']] = result
            except Exception as e:
                # Handle exceptions and print error details
                print(f"Error in {mdl} with {f_method}, excludeModels={excl_models}, includeModels={incl_models}, use_best_model={best_model}, time_limit={t_limit}: {e}")

    
    ###############################################################################################
    # calculate_weights
    ###############################################################################################

    from tools.weights.calculate_weights import calculate_weights
    if verbosity >= 2:
        print("Start calculating weights")
   
    weight_files = [f for f in os.listdir(weights_path) if f.startswith('Forecast')]

    configurations = list(itertools.product(optim_method, remove_groups))
    configurations = [[(method, remove) for method, remove in configurations]]
    
    dict_weighted_forecast = {}

    for file_name in weight_files:
        pkl_filename = os.path.join(weights_path, file_name)

        with open(pkl_filename, 'rb') as f:
            Forecast_Weights_data = pickle.load(f)

        try:
            for optim_method, remove_groups in configurations[0]:
                # Calcualte the optimal weights
                if verbosity >= 2:
                    print("calculate weights with optim method:")
                    print(optim_method)
                weights = calculate_weights(results=Forecast_Weights_data, optim_method=optim_method, remove_groups=remove_groups, include_groups = include_groups, hillclimbsets=1, max_iterations=200, saveResults=False, save_path=weights_path)
                filename = weights["meta_data"]["file_name"]
                dict_weighted_forecast[filename] = weights
            if verbosity >= 2:
                print("calculate weights for combinations:")    
                print(dict_weighted_forecast.keys())
        except Exception as e:
            return()
            print(f"Error loading {file_name}: {e}")

    ###############################################################################################
    # calculate Forecast
    ###############################################################################################
    from tools.forecast.create_forecasts import create_forecasts

    # Delete Forecast folder
    if delete_forecast_folder:
        if os.path.exists(forecasts_path):
            shutil.rmtree(forecasts_path)
                    
    if RERUN_calculate_forecast:

        model = ensure_iterable(model)
        forecast_method = ensure_iterable(forecast_method)
        excludeModels = ensure_iterable(excludeModels)
        includeModels = ensure_iterable(includeModels)
        use_test_data = ensure_iterable(use_test_data)  
        time_limit = ensure_iterable(time_limit) 

        
        configurations = list(itertools.product(
            model,
            forecast_method,
            excludeModels,
            includeModels,
            use_test_data,
            time_limit
        ))

        if reduceCompTime:
            configurations = [
                (mdl, f_method, excl_models, incl_models, use_test_d, t_limit) 
                for mdl, f_method, excl_models, incl_models, use_test_d, t_limit in configurations
                if not (
                    (f_method == "level" and mdl in ["AutoETS", "AutoARIMA"]) or
                    (t_limit == 1800 and mdl in ["AutoETS", "AutoARIMA"]) or
                    (f_method == "level" and t_limit == 1800)
                )
            ]

        if verbosity >= 3: 
            print("configurations Forecasts")
            print(configurations)

        forecast_dic = {}

        # go thtough all configurations
        for mdl, f_method, excl_models, incl_models, use_test, t_limit in configurations:
            try:
                if use_test:
                    forecast = create_forecasts(
                        train_dic, 
                        test_dic, 
                        future_periods=future_periods, 
                        freq=data["freq"], 
                        model=mdl, 
                        includeModels=incl_models, 
                        excludeModels=excl_models, 
                        saveResults=True, 
                        save_path=forecasts_path, 
                        forecast_method=f_method, 
                        time_limit=t_limit
                    )
                else:
                    forecast = create_forecasts(
                        train_dic,  
                        future_periods=future_periods, 
                        freq=data["freq"], 
                        model=mdl, 
                        includeModels=incl_models, 
                        excludeModels=excl_models, 
                        saveResults=True, 
                        save_path=forecasts_path, 
                        forecast_method=f_method, 
                        time_limit=t_limit
                    )
                file_name = forecast["meta_data"]["file_name"]
                forecast_dic[file_name] = forecast
            except Exception as e:
                print(f"Error in model {mdl} with forecast_method {f_method}, use_test_data={use_test}, excludeModels={excl_models}, includeModels={incl_models}, time_limit={t_limit}: {e}")


    ###############################################################################################
    # Combine Weights + Forecasts
    ###############################################################################################
    from tools.weights.compare_weighted_forecasts import compare_weighted_forecasts

    if verbosity >= 2:
        print("Start Combine Weights + Forecasts")

    # load forecasts
    forecast_files = [f for f in os.listdir(forecasts_path) if f.startswith('Forecast')]
    forecast_dic = {}
    for file_name in forecast_files:
        pkl_filename = os.path.join(forecasts_path, file_name)
        try:
            with open(pkl_filename, 'rb') as f:
                forecasts = pickle.load(f)
                forecast_dic[file_name] = forecasts
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    from tools.combine_results.combine_weights_and_forecasts import create_weights_forecast_dict

    if verbosity >= 4:
        print("compare_weighted_forecasts")
        print(dict_weighted_forecast.keys())

    weighted_losses, weighted_losses_results = compare_weighted_forecasts(dict_weighted_forecast)

    # Combine Train Forecast with Test Forecast according to criteria
    results_grouping_parameter = ["model", "forecast_method", "time_limit"]
    results_additional_grouping_parameter = ["optim_method"] 

    if verbosity >= 2:
        print("create_weights_forecast_dict")
        print(forecast_dic.keys())

    weights_forecast_dict = create_weights_forecast_dict(dict_weighted_forecast, forecast_dic, weighted_losses, results_grouping_parameter, results_additional_grouping_parameter, verbosity = verbosity)

    ###############################################################################################
    # Reconciliation methods
    ###############################################################################################
    if verbosity > 3:
        print("Reconciliation methods")

    from tools.transformations.transform_multiple_dict_to_long import transform_multiple_dict_to_long

    # aggregated data into long format (needed for reconciliation method)
    aggregated_data = aggregate_by_levels(data = data, exclude_variables = None, method='long', show_dict_infos=False)

    pkl_filename = [entry['meta_data']['file_name'] for entry in forecast_dic.values() if 'meta_data' in entry and 'file_name' in entry['meta_data']]

    new_forecast_dic = {}

    for key, value in forecast_dic.items():
        if "predicted_dic" in value:
            new_forecast_dic[key] = value["predicted_dic"]


    if verbosity >= 2:
        print("transform_multiple_dict_to_long")

    Y_hat_df = transform_multiple_dict_to_long(new_forecast_dic, id_col = "ts_id", numeric_col = "pred")

    if verbosity >= 4:
        print("transform_multiple_dict_to_long End")
    from hierarchicalforecast.methods import BottomUp, MinTrace
    from hierarchicalforecast.core import HierarchicalReconciliation

    reconcilers = [
        BottomUp(),
        MinTrace(method='wls_struct'),
        MinTrace(method='ols'),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    Y_rec_df = {}
  
    for label, Y_hat_df_label in Y_hat_df.items():

        print(label)
        s_only = set(aggregated_data["S_df"].index) - set(Y_hat_df_label.index)
        yhat_only = set(Y_hat_df_label.index) - set(aggregated_data["S_df"].index)


        print(f"Einträge in Y_hat_df, aber nicht in S_df: {yhat_only}")
        print(f"Einträge in S_df, aber nicht in Y_hat_df: {s_only}")
        
        
        Y_rec_df[label] = hrec.reconcile(
            Y_hat_df=Y_hat_df_label,  # Das transformierte DataFrame
            S=aggregated_data["S_df"],  # Aggregierte Daten, angenommen du hast diese schon geladen
            tags=aggregated_data["tags"]  # Tags, ebenfalls schon geladen
        )

    ###############################################################################################
    # Combine actuals and reconciliations
    ###############################################################################################
    if verbosity > 3:
        print("Combining actuals and reconciliations...")
    
    from tools.combine_results.combine_actuals_and_reconciliations import combine_actuals_and_reconciliations
    
    # Combine actual values with reconciliations into a dictionary
    reconciliation_dict = combine_actuals_and_reconciliations(aggregated_data, Y_rec_df)
    
    # Iterate through weighted forecast dictionary to validate reconciliation entries
    for model_key, model_data in weights_forecast_dict.items():
        # Extract the forecast key for the current model
        forecast_key = model_data.get('forecast_key')
    
        # Check if the forecast key exists in the reconciliation dictionary
        if forecast_key and forecast_key in reconciliation_dict:
            print(model_data.get('forecast_equal_weights'))  # Debugging/logging output
    
    ###############################################################################################
    # Combine reconciliation results, optimization method results, and mean forecasts
    ###############################################################################################
    if verbosity > 3:
        print("Merging reconciliations, optimization results, and mean method results...")
    
    from tools.combine_results.combine_weightedForecast_and_reconciliation import merge_forecast_with_reconciliation
    
    # Merge the weighted forecasts with reconciliation results
    final_dict = merge_forecast_with_reconciliation(weights_forecast_dict, reconciliation_dict)
    
    ###############################################################################################
    # Calculate evaluation metrics for the forecasts
    ###############################################################################################
    if verbosity > 3:
        print("Calculating evaluation metrics...")
    
    from tools.evaluation.calculate_metrics import calculate_metrics
    
    # List of forecasting methods to evaluate
    method_list = ["base", "equal_weights_pred", "BottomUp", "MinTrace_method-wls_struct",
                   "MinTrace_method-ols", "weighted_pred"]
    
    # List of metrics to calculate
    metric = ["MASE", "WAPE", "RMSSE", "SMAPE"]
    
    # Copy the final results dictionary for evaluation
    metrics_result_dict = final_dict.copy()
    
    # Extract dataset name for logging and evaluation
    dataset = data['dataframe_name']
    
    # Call the calculate_metrics function to compute evaluation metrics
    metrics_table = calculate_metrics(final_result=metrics_result_dict,
                                      method_list=method_list,
                                      metrics=metric,
                                      dataset_name=dataset,
                                      date_column='date',
                                      actual_column='y',
                                      verbosity=verbosity,
                                      test_period=test_period,
                                      future_periods=future_periods)
    
    # Add evaluation metrics to the final dictionary
    final_dict["evaluation_metrics"] = metrics_table
    
    if verbosity >= 2:
        print("Metrics calculation completed.")
    
    ###############################################################################################
    # Save results in the intermediate and final paths
    ###############################################################################################
    def save_pickle(final_dict, save_intermediate_results_path, dataset_name, delte_input_data):
        """
        Save the final dictionary as a pickle file. Optionally, remove input data keys to reduce file size.
    
        Args:
            final_dict (dict): Dictionary containing the forecast results, metrics, and reconciliation data.
            save_intermediate_results_path (str): Directory path to save the intermediate pickle file.
            dataset_name (str): Name of the dataset used for naming the pickle file.
            delte_input_data (bool): If True, remove input data keys to reduce the pickle file size.
    
        Returns:
            dict: The final dictionary (possibly with reduced content if delte_input_data=True).
        """
        # Generate the filename for the pickle file
        pkl_filename = os.path.join(save_intermediate_results_path, dataset_name + ".pkl")
        os.makedirs(save_intermediate_results_path, exist_ok=True)
    
        # If input data should be removed, clean the final dictionary
        if delte_input_data:
            keys_to_remove = ['weights_key', 'forecast_key', 'weights_dict', 'forecast_dict', 'reconciliation_dct']
    
            for outer_key, inner_dict in final_dict.items():
                # Check if the value is a dictionary
                if isinstance(inner_dict, dict):
                    # Remove specified keys if they exist
                    for key in keys_to_remove:
                        inner_dict.pop(key, None)  # Safe removal without errors
    
        # Save the cleaned dictionary as a pickle file
        output_dir = os.path.dirname(pkl_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        with open(pkl_filename, 'wb') as f:
            pickle.dump(final_dict, f)
            print(f"Results saved in '{pkl_filename}' as a pickle file.")
    
        return final_dict
    
    # Save the final dictionary as a pickle file in the intermediate path
    final_dict = save_pickle(final_dict=final_dict,
                             save_intermediate_results_path=save_intermediate_results_path,
                             dataset_name=dataset_name,
                             delte_input_data=True)
    
    # Optionally, save the results in the final path if provided
    if save_final_results_path is not None:
        save_path = save_final_results_path
        pkl_filename = os.path.join(save_path, dataset_name + ".pkl")
        os.makedirs(save_path, exist_ok=True)
    
        with open(pkl_filename, 'wb') as f:
            pickle.dump(final_dict, f)
        print(f"Results saved in '{pkl_filename}' as a pickle file.")
    
    return None
