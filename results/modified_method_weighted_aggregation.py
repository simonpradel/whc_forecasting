import os
import sys

main_dir = "/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction"
sys.path.append(os.path.abspath(main_dir))
os.chdir(main_dir)

import random
random.seed(123)

def run_model_aggregation(
    save_intermediate_results_path, 
    save_final_results_path, 
    dataset_name, 
    model = ["AutoETS", "AutoArima"], 
    forecast_method = ["global"], 
    use_best_model = False, 
    time_limit = 60 * 10, 
    verbosity = 5, 
    test_period = 6,  # TEST PERIOD (TRAIN/TEST SPLIT) can be shorter than future period
    includeModels = None, 
    excludeModels = None, 
    fold_length = 6, 
    used_period_for_cv = 0.45, 
    include_groups = ["PL_line", "Segment"], 
    optim_method = ["ensemble_selection", "optimize_nnls", "differential_evolution"], 
    remove_groups = [False], 
    future_periods = 12, # prediction length 
    use_test_data = True, 
    cutoff_lag = None,
    cutoff_date = None,
    reduceCompTime = True, 
    delete_weights_folder = True,
    delete_forecast_folder = True,
    RERUN_calculate_weights = False,
    RERUN_calculate_forecast = False,
):
    """
    Diese Funktion führt die Modellaggregation durch und speichert die Ergebnisse ohne Rückgabewert.
    Die Pfade für Zwischenergebnisse und Endergebnisse werden übergeben, ebenso wie alle weiteren Parameter.
    """
    
    import os
    import sys
    import pandas as pd      
    import itertools

    ###############################################################################################
    # Prepare folder
    ###############################################################################################
    weights_path = os.path.join(save_intermediate_results_path, "weights")
    forecasts_path = os.path.join(save_intermediate_results_path, "forecasts")

    ###############################################################################################
    # Cutoff Date
    ###############################################################################################

    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    if (cutoff_date == None):
        # Heutiges Datum
        today = datetime.today()

        # Berechne den letzten Tag des letzten Monats
        last_day_last_month = today.replace(day=1) - relativedelta(days=1)

        # Überprüfen, ob der letzte Tag des letzten Monats mindestens x Tage zurückliegt
        if cutoff_lag == None:
            cutoff_lag = 5

        if (today - last_day_last_month).days < cutoff_lag:
            # Wenn weniger als x Tage zurück, nehme den letzten Tag des vorletzten Monats
            last_day_last_month = last_day_last_month.replace(day=1) - relativedelta(days=1)

        # Formatierung in 'YYYY-MM-DD'
        cutoff_date = last_day_last_month.strftime('%Y-%m-%d')
    
    ###############################################################################################
    # prepare data
    ###############################################################################################

    import pickle
    from tools.load_data.load_data_from_catalog import load_data_from_catalog
    from tools.transformations.prepare_data import prepare_data
    from tools.transformations.aggregate_by_level import aggregate_by_levels
    from tools.methods.split_data import split_data

    # Dateiname für das Speichern der Daten festlegen
    pkl_filename = f"data_{dataset_name}.pkl"
    pkl_filename = os.path.join(save_intermediate_results_path, pkl_filename)  # Vollständigen Pfad erstellen

    load_data_from_catalog = False
    save_data = False #'DONT CHANGE TO TRUE' UNLESS YOU WANT TO OVERWRITE THE DATA

    # Datei laden
    if load_data_from_catalog == False:
        with open(pkl_filename, 'rb') as f:
            loaded_data = pickle.load(f)

        # Zugriff auf die einzelnen Elemente
        data = loaded_data['data']
        train_dic = loaded_data['train_dic']
        test_dic = loaded_data['test_dic']
    else:
        # Laden der Daten aus dem Katalog
        load_data = load_data_from_catalog(dataset_name, maindir = None)

        # Vorbereitung der Daten (Schließen von Lücken in den Daten)
        data = prepare_data(load_data, cutoff_date=cutoff_date, fill_missing_rows=True)

        # Kopieren der vorbereiteten Daten
        prepared_data = data["pandas_df"].copy()
        
        # Aggregieren der Daten auf verschiedenen Ebenen und Umwandlung in Dictionary
        aggregated_data_dict = aggregate_by_levels(data=data, method='dictionary', show_dict_infos=False)


        # Entfernen des 'original_dataset' Schlüssels, um redundante Daten zu vermeiden
        del data['original_dataset']

        # Aufteilen der aggregierten Daten in Trainings- und Testdatensätze (Dictionary-Format)
        train_dic, test_dic = split_data(aggregated_data_dict, period=test_period, unique_id="ts_id", format="dictionary")
       
    ###############################################################################################
    # Save data
    ###############################################################################################
    # Überprüfe, ob der Verzeichnispfad existiert, und erstelle ihn bei Bedarf
    pkl_directory = os.path.dirname(os.path.join(save_intermediate_results_path, dataset_name))

    if not os.path.exists(pkl_directory):
        os.makedirs(pkl_directory)

    # Speichern von 'data', 'train_dic' und 'test_dic' in der pickle-Datei
    if(save_data):
        with open(pkl_filename, 'wb') as f:
            pickle.dump({
                'data': data.copy(),
                'train_dic': train_dic,
                'test_dic': test_dic
            }, f)
        if verbosity >= 1:
            print(f"Data saved as '{pkl_filename}'.")

    ###############################################################################################
    # Calculate stepwise Forecasts before Weighting
    ###############################################################################################
    # Hilfsfunktion, um Nicht-iterierbare oder Strings in Listen zu konvertieren
    def ensure_iterable(value):
        if value is None:  # Falls None, gebe eine Liste mit None zurück
            return [None]
        elif isinstance(value, (list, tuple)):  # Falls es bereits eine Liste oder ein Tupel ist, ändere nichts
            return value
        else:  # Andernfalls umwandle es in eine Liste
            return [value]
    
    if RERUN_calculate_weights:
        from tools.weights.calculate_weights_forecast import calculate_weights_forecast
        import math

        # Delete Weight folder
        if delete_weights_folder:
            import shutil
            if os.path.exists(weights_path):
                shutil.rmtree(weights_path)

        # Set parameters
        n_splits = math.floor(len(train_dic[('dataset',)])*used_period_for_cv/fold_length)

            
        # Fallback zu None, falls eine der Variablen None ist
        model = ensure_iterable(model)
        forecast_method = ensure_iterable(forecast_method)
        excludeModels = ensure_iterable(excludeModels)
        includeModels = ensure_iterable(includeModels)
        use_best_model = ensure_iterable(use_best_model)
        time_limit = ensure_iterable(time_limit)

        configurations = list(itertools.product(
            model,
            forecast_method,
            excludeModels,
            includeModels,
            use_best_model,
            time_limit
        ))

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
            print("configurations forecast weights")
            print(configurations)

        weights_forecast = {}
        # Durchlaufen aller Konfigurationen
        for mdl, f_method, excl_models, incl_models, best_model, t_limit in configurations:
            try:
                result = calculate_weights_forecast(
                    train_dict=train_dic,
                    freq=data["freq"],
                    n_splits=n_splits,
                    fold_length=fold_length,
                    forecast_method=f_method,
                    model=mdl,
                    excludeModels=excl_models,
                    includeModels=incl_models,
                    use_best_model=best_model,
                    saveResults=True,
                    save_path=weights_path,
                    verbosity=4,
                    time_limit=t_limit
                )
                weights_forecast[result["meta_data"]['file_name']] = result
            except Exception as e:
                 print(f"Error in {model} with {forecast_method}, excludeModels={excludeModels}, includeModels={includeModels}, use_best_model={use_best_model}, time_limit={time_limit}: {e}")

    
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
                # Berechne die Gewichte und optimiere das Modell
                if verbosity >= 2:
                    print("calculate weights with optim method:")
                    print(optim_method)
                weights = calculate_weights(results=Forecast_Weights_data, optim_method=optim_method, remove_groups=remove_groups, include_groups = include_groups, hillclimbsets=1, max_iterations=200, saveResults=False, save_path=current_dir)
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
  
    if RERUN_calculate_forecast:

        # Delete Forecast folder
        if delete_forecast_folder:
            if os.path.exists(forecasts_path):
                shutil.rmtree(forecasts_path)

        # Fallback zu einer Liste mit None oder der Variablen als Liste, falls sie nicht iterierbar ist
        model = ensure_iterable(model)
        forecast_method = ensure_iterable(forecast_method)
        excludeModels = ensure_iterable(excludeModels)
        includeModels = ensure_iterable(includeModels)
        use_test_data = ensure_iterable(use_test_data)  # Sicherstellen, dass use_test_data iterierbar ist
        time_limit = ensure_iterable(time_limit)  # Sicherstellen, dass time_limit iterierbar ist

        # Erstellen des Kreuzprodukts für alle Variablenkombinationen
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

        # Durchlaufen aller Konfigurationen
        forecast_dic = {}

        for mdl, f_method, excl_models, incl_models, use_test, t_limit in configurations:
            try:
                if use_test:
                    forecast = create_forecasts(
                        train_dic, 
                        test_dic,  # Testdaten werden verwendet
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
                        train_dic,  # Keine Testdaten
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

    # input_filename = os.path.join(forecasts_path, 'forecast_results.pkl')
    # with open(input_filename, 'rb') as f:
    #     forecast_dic = pickle.load(f)

    from tools.combine_results.combine_weights_and_forecasts import combine_weights_and_forecasts
    from tools.combine_results.combine_weights_and_forecasts import create_weights_forecast_dict

    if verbosity >= 4:
        print("compare_weighted_forecasts")
        print(dict_weighted_forecast.keys())

    # Legacy: This part can be written better
    weighted_losses, weighted_losses_results = compare_weighted_forecasts(dict_weighted_forecast)

    # SCHWACHSTELLE: Prüfe genau ob die Weights und Fct dict nur anhand der folgenden Variablen kombineirt werden knnen 
    # Combine Train Forecast with Test Forecast according to criteria
    # for each weights forecast a test forecast can be combined through the parameter model and forecast method (each weights forecast with eg level and AutoETS should be combined with the same level and autoETS forecast in the testperiod.)
    # for each weights forecast results 3 forecasts because there are 3 optim-methods 
    results_grouping_parameter = ["model", "forecast_method", "time_limit"]
    results_additional_grouping_parameter = ["optim_method"] # Gibt mehrere Predictions aus
    #results_additional_grouping_parameter = ["optim_method", "hillclimbsets"]
    if verbosity >= 2:
        print("create_weights_forecast_dict")
        print(forecast_dic.keys())

    weights_forecast_dict = create_weights_forecast_dict(dict_weighted_forecast, forecast_dic, weighted_losses, results_grouping_parameter, results_additional_grouping_parameter, verbosity = verbosity)
    print("DEBUGGING")
    print(weights_forecast_dict.keys())

    ###############################################################################################
    # Reconciliation methods
    ###############################################################################################
    if verbosity > 3:
        print("Reconciliation methods")

    from tools.transformations.transform_multiple_dict_to_long import transform_multiple_dict_to_long

    # aggregated data into long format (needed for reconciliation method)
    aggregated_data = aggregate_by_levels(data = data, exclude_variables = None, method='long', show_dict_infos=False)

    pkl_filename = [entry['meta_data']['file_name'] for entry in forecast_dic.values() if 'meta_data' in entry and 'file_name' in entry['meta_data']]
    #pkl_filename = [entry['file_name'] for entry in forecast_dic.values() if 'file_name' in entry]
    #print(forecast_dic['Forecast_AutoETS_global_12_M.pkl']["predicted_dic"].keys())

    # Neues Dictionary initialisieren
    new_forecast_dic = {}
    # Iteration über die erste Ebene des forecast_dic
    for key, value in forecast_dic.items():
        # Überprüfen, ob "predicted_dic" im aktuellen Wert enthalten ist
        if "predicted_dic" in value:
            # Den entsprechenden "predicted_dic" in das neue Dictionary einfügen
            new_forecast_dic[key] = value["predicted_dic"]


    if verbosity >= 2:
        print("transform_multiple_dict_to_long")

    #print(new_forecast_dic)
    # this is needed to use reconciliation methods
    Y_hat_df = transform_multiple_dict_to_long(new_forecast_dic, id_col = "ts_id", numeric_col = "pred")

    if verbosity >= 4:
        print("transform_multiple_dict_to_long End")
    from hierarchicalforecast.methods import BottomUp, MinTrace, MinTraceSparse, ERM
    from hierarchicalforecast.core import HierarchicalReconciliation

    reconcilers = [
        BottomUp(),
        MinTrace(method='wls_struct'),
        MinTrace(method='ols'),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    # Dictionary für die rekonsilierten Daten initialisieren
    Y_rec_df = {}
  
    # Schleife, die durch das transformierte Dictionary iteriert und die Funktion hrec.reconcile() anwendet
    for label, Y_hat_df_label in Y_hat_df.items():

        print(label)
        # Wende die reconcile-Funktion auf jeden Eintrag an
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
    # combine actuals and reconciliations
    ###############################################################################################
    if verbosity > 3:
        print("combine_actuals_and_reconciliations")

    from tools.transformations.transform_aggregated_data import transform_long_to_dict
    from tools.combine_results.combine_actuals_and_reconciliations import combine_actuals_and_reconciliations

    reconciliation_dict = combine_actuals_and_reconciliations(aggregated_data, Y_rec_df)

    ###############################################################################################
    # Combine reconciliation + opt method + mean method
    ###############################################################################################
    if verbosity > 3:
        print("reconciliation + opt method + mean method")

    from tools.combine_results.combine_weightedForecast_and_reconciliation import combine_weightedForecast_reconciliation
    from tools.combine_results.combine_weightedForecast_and_reconciliation import merge_forecast_with_reconciliation

    #Funktion zum Verschneiden von forecast_weighted und reconciliation_dct basierend auf den Schlüssel-Gruppierungsvariablen
    final_dict = merge_forecast_with_reconciliation(weights_forecast_dict, reconciliation_dict)

    ###############################################################################################
    # Calculate Metrics
    ###############################################################################################
    if verbosity > 3:
        print("Calculate Metrics")

    from tools.evaluation.calculate_metrics import calculate_metrics
    method_list = ["base", "equal_weights_pred", "BottomUp", "MinTrace_method-wls_struct", "MinTrace_method-ols", "weighted_pred"]
    metric = ["MAE", "MSE", "MAPE", "MASE", "WAPE", "RMSE", "RMSSE", "SMAPE"]  
    metrics_result_dict = final_dict.copy()
    dataset = data['dataframe_name']
    #metrics_result_dict.display()
    metrics_table = calculate_metrics(final_result = metrics_result_dict, method_list = method_list, metrics =  metric, dataset_name = dataset, date_column = 'date', actual_column = 'y', verbosity = verbosity, test_period = test_period, future_periods = future_periods)
    # keys_to_remove = ['a', 'c', 'd']  # 'd' ist nicht im Dictionary
    # for key in keys_to_remove:
    #     final_dict.pop(key, None)
    final_dict["evaluation_metrics"] = metrics_table

    if verbosity >= 2:
        print("Calculate Metrics done")
        # print(metrics_table)
    ###############################################################################################
    # Save everything in save_intermediate_results_path/final path
    ###############################################################################################
    def save_pickle(final_dict, save_intermediate_results_path, dataset_name, delte_input_data):
        # Dateipfad erstellen
        pkl_filename = os.path.join(save_intermediate_results_path, dataset_name + ".pkl")
        os.makedirs(save_intermediate_results_path, exist_ok=True)


        if (delte_input_data):
            # Wenn File zu groß wird, lösche paar Input Argumente
            keys_to_remove = ['weights_key', 'forecast_key', 'weights_dict', 'forecast_dict', 'reconciliation_dct']

            for outer_key, inner_dict in final_dict.items():
                # Prüfe, ob der Wert ein Dictionary ist
                if isinstance(inner_dict, dict):
                    # Lösche die angegebenen Schlüssel, falls sie existieren
                    for key in keys_to_remove:
                        inner_dict.pop(key, None)  # None verhindert Fehler, falls der Schlüssel nicht vorhanden ist

        # else:
        cobined_dict = final_dict
        output_dir = os.path.dirname(pkl_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Datei speichern
        with open(pkl_filename, 'wb') as f:
            pickle.dump(cobined_dict, f)
            print(f"Results saved in '{pkl_filename}' as a pkl file.")

        return cobined_dict

    # Beispielaufruf
    final_dict = save_pickle(final_dict = final_dict, save_intermediate_results_path = save_intermediate_results_path, dataset_name = dataset_name, delte_input_data = True)

    # optional: save in final_path
    if(save_final_results_path != None):
        save_path = save_final_results_path
        pkl_filename = os.path.join(save_path, dataset_name + ".pkl")
        os.makedirs(save_path, exist_ok=True)

        # save in save_intermediate_results_path
        with open(pkl_filename, 'wb') as f:
            pickle.dump(final_dict, f)
        print(f"Results saved in '{pkl_filename}' as a pkl file.")

    return None


################################################################################################################
# SETTINGS
################################################################################################################

current_dir = "/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/results/"

# main Setting 
forecast_methods = ["global", "level"]
time_limit = [60 * 10, 60 * 30] # pro fold = time_limit in sec, für jedes Kombination!! # 60 * 10 = 10 Min, 60 * 30 = 30 Min
models = ["AutoARIMA", "AutoETS", "AutoTS", "AutoGluon"]
optim_method = ["ensemble_selection", "optimize_nnls", "differential_evolution"]
used_period_for_cv = 0.45

# Further Settings
reduceCompTime = True # if True, the combination "level" + "AutoETS" or "AutoARIMA" is NOT calculated
use_best_model = False
verbosity = 3 # 0 = close to nothing, 1 = only important messages, 2 = all important section in MAIN File, 3 = all important section in FUNCTION File, 4 = debugging Main, 5 = Debugging Function
includeModels = None
excludeModels = None
remove_groups = [False] #True, 
use_test_data = True
delete_weights_folder = False # DONT CHANGE TO TRUE, IF YOU DONT WANT DO DISCARD LONG RUN CALCULATIONS
delete_forecast_folder = False # DONT CHANGE TO TRUE, IF YOU DONT WANT DO DISCARD LONG RUN CALCULATIONS
RERUN_calculate_weights = True # False only useful if a weights folder exists 
RERUN_calculate_forecast = True # False only useful if a forecasts folder exists
RERUN_calculate_weights = False # False only useful if a weights folder exists 
RERUN_calculate_forecast = False # False only useful if a forecasts folder exists

### Telefonica Settings ###
test_period = 1
fold_length = 6
future_periods = 12
include_groups = None # ["PL_line", "Segment"]
### Telefonica Settings ###


# Individual settings
datasets = [
    {"Telefoncia_data": False, "name": "website_traffic", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "store_item_demand", "test_period": 6, "fold_length": 6, "future_periods": 6, "include_groups": None},
    {"Telefoncia_data": False, "name": "retail_prices", "test_period": 12, "fold_length": 12, "future_periods": 12, "include_groups": None},
    {"Telefoncia_data": False, "name": "prison_population", "test_period": 4, "fold_length": 4, "future_periods": 4, "include_groups": None},
    {"Telefoncia_data": False, "name": "natural_gas_usage", "test_period": 12, "fold_length": 12, "future_periods": 12, "include_groups": None},
    {"Telefoncia_data": False, "name": "italian_grocery_store", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "global_electricity_production", "test_period": 15, "fold_length": 15, "future_periods": 15, "include_groups": None},
    {"Telefoncia_data": False, "name": "australian_labour_market", "test_period": 14, "fold_length": 14, "future_periods": 14, "include_groups": None},
    {"Telefoncia_data": False, "name": "M5", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "tourism", "test_period": 7, "fold_length": 7, "future_periods": 7, "include_groups": None},
    {"Telefoncia_data": False, "name": "superstore", "test_period": 4, "fold_length": 4, "future_periods": 4, "include_groups": None},
    {"Telefoncia_data": True, "name": "Telefonica - bad_debt", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - commercial_costs", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - cos", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - fbb_fixed_other_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - hardware_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - mobile_service_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - non_commercial_costs", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - non_recurrent_income_cost", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups}
]


# Filter datasets
run_Telefonica_data = True
run_other_data = True

# run one or multiple specific dataset (einschließlich, was nicht laufen soll)
run_dataset = None # no filter
run_dataset = ['Telefonica - bad_debt', 'Telefonica - hardware_revenue', 'Telefonica - mobile_service_revenue', 'Telefonica - non_commercial_costs', 'Telefonica - non_recurrent_income_cost']
# run_dataset = ["M5"]
# run_dataset = ["website_traffic", "store_item_demand", "retail_prices", "prison_population", "natural_gas_usage", "italian_grocery_store", "global_electricity_production", "australian_labour_market", "M5", "tourism", "superstore"]  

run_dataset = ['global_electricity_production']
run_dataset = ['australian_labour_market', 'global_electricity_production', 'italian_grocery_store', 'M5', 'natural_gas_usage', 'prison_population', 'retail_prices', 'store_item_demand', 'superstore', 'tourism', 'Telefonica - bad_debt', 'Telefonica - commercial_costs', 'Telefonica - cos', 'Telefonica - fbb_fixed_other_revenue', 'Telefonica - hardware_revenue', 'Telefonica - mobile_service_revenue', 'Telefonica - non_commercial_costs', 'Telefonica - non_recurrent_income_cost']
run_dataset = ["tourism", "global_electricity_production", "superstore", "M5"]
run_dataset = ['Telefonica - bad_debt', 'Telefonica - commercial_costs', 'Telefonica - cos', 'Telefonica - fbb_fixed_other_revenue', 'Telefonica - hardware_revenue', 'Telefonica - mobile_service_revenue', 'Telefonica - non_commercial_costs', 'Telefonica - non_recurrent_income_cost']
#run_dataset = ['website_traffic']
run_dataset = None # no filter

#run_dataset = None # no filter
exclude_datasets = None #["website_traffic", "australian_labour_market", "superstore", "M5"]
#exclude_datasets = ['Telefonica - bad_debt']  # z.B. ["website_traffic"] oder None

# Iteriere über jeden Datensatz und führe die Funktion mit Fehlerbehandlung aus
def filter_datasets(datasets, run_Telefonica_data, run_other_data, run_dataset=None, exclude_datasets=None):
    # Filtere nach spezifischen Datensätzen, falls angegeben
    if run_dataset:
        datasets = [dataset for dataset in datasets if dataset["name"] in run_dataset]
    
    # Filtere nach ausgeschlossenen Datensätzen, falls angegeben
    if exclude_datasets:
        datasets = [dataset for dataset in datasets if dataset["name"] not in exclude_datasets]
    
    # Filtere nach Telefonica-Daten oder anderen Daten, je nach Flags
    if not run_Telefonica_data:
        datasets = [dataset for dataset in datasets if not dataset["Telefoncia_data"]]
    if not run_other_data:
        datasets = [dataset for dataset in datasets if dataset["Telefoncia_data"]]
    
    return datasets

# Aufruf der Filterfunktion
datasets = filter_datasets(datasets, run_Telefonica_data, run_other_data, run_dataset, exclude_datasets)

# Ausgabe der gefilterten Datensätze
print("Filtered datasets:")
for dataset in datasets:
    print(dataset)

################################################################################################################
# RUN
################################################################################################################    
for dataset in datasets:
    try:
        dataset_name = dataset["name"]
        print(f"Running model aggregation for dataset: {dataset_name}")
        
        if dataset["Telefoncia_data"]:
            save_final_results_path = "/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/results/Telefonica - overall/forecasts"
        else:
            save_final_results_path = None
            #save_final_results_path = "/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/results/results_paper"
            
        # Ausführung der Funktion mit spezifischen Parametern
        run_model_aggregation(
            save_intermediate_results_path = os.path.join(current_dir, dataset_name), 
            save_final_results_path = save_final_results_path, 
            dataset_name = dataset_name, 
            model = models, 
            forecast_method = forecast_methods, 
            use_best_model = use_best_model, 
            time_limit = time_limit, 
            verbosity = verbosity, 
            test_period = dataset["test_period"], 
            includeModels = includeModels, 
            excludeModels = excludeModels, 
            fold_length = dataset["fold_length"], 
            used_period_for_cv = used_period_for_cv, 
            include_groups = dataset["include_groups"], # grouping variables which should be included in any case
            optim_method = optim_method, 
            remove_groups = remove_groups, 
            future_periods = dataset["future_periods"], 
            use_test_data = use_test_data, 
            reduceCompTime = reduceCompTime,
            delete_weights_folder = delete_weights_folder,
            delete_forecast_folder = delete_forecast_folder,
            RERUN_calculate_weights = RERUN_calculate_weights,
            RERUN_calculate_forecast = RERUN_calculate_forecast
        )
        
        print(f"Finished model aggregation for dataset: {dataset_name}")
        
    except Exception as e:
        print(f"Error in model aggregation for dataset: {dataset_name} - {e}")

