import sys
import os
import pandas as pd
import numpy as np
import pickle
from tools.models.AutoGluon import train_autogluon_and_forecast 
from tools.models.AutoTS import train_autots_and_forecast 
from tools.models.AutoARIMA import train_AutoARIMA_and_forecast 
from tools.models.AutoETS import train_AutoETS_and_forecast 
from tools.transformations.transform_aggregated_data import *
from tools.methods.get_function_args import get_function_args
from datetime import datetime
import time



def create_forecasts(train_dic, test_dic = None, future_periods = 12, freq = "D", model="AutoGluon", train_end_date=None, includeModels=None, excludeModels=None, forecast_method="global", saveResults=True, save_path=os.getcwd(), time_limit = None, verbosity = 0):
    """
    Creates forecasts for selected groups and returns a DataFrame with actual and forecasted values.
    Additionally, the `test_dic` is enriched with forecasted values (via a left join on 'ts_id' and 'date').

    Parameters:
    train_dic (dict): Dictionary with time series data at the lower level.
    test_dic (dict): Dictionary with DataFrames at the top level with columns 'date' and 'total'.
    future_periods (int): Number of future periods for the forecast.
    freq (str): Frequency of the time series (e.g., 'D' for daily, 'M' for monthly).
    model (str): Model type for forecasting (e.g., 'AutoGluon', 'Naive', 'NBEATS').
    train_end_date (str): End date for the training dataset.
    includeModels (list): List of models to include (relevant only for AutoGluon).
    excludeModels (list): List of models to exclude (relevant only for AutoGluon).
    forecast_method (str): Forecasting mode, defaulting to "global" for a global forecast.
    saveResults (bool): If True, the result is saved as a `.pkl` file.

    Returns:
    dict: The updated `test_dic`, which is augmented with the forecasted values.
    """
    start_time = time.time()

    if test_dic is not None:
        predicted_dic = test_dic.copy()
    else:
        predicted_dic = {}

    best_model = None
    
    # Handle global mode
    if forecast_method == "global":
        mapping, df_long = transform_dict_to_long(train_dic, id_col='ts_id', date_col='date', actuals_col="total")
        print("Starting global forecast...")
        
        if model == "AutoTS":
            print(df_long.info())
            predicted_values, best_model = train_autots_and_forecast(
                df_long, future_periods, freq=freq, includeModels=includeModels,
                excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit = time_limit)
        elif model == "AutoGluon":
            predicted_values, best_model = train_autogluon_and_forecast(
                df_long, future_periods, freq=freq, includeModels=includeModels,
                excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit = time_limit)
        elif model == "AutoETS":
                predicted_values, fitted_values = train_AutoETS_and_forecast(
                df_long, future_periods, freq=freq, date_col = "date", id_col = "ts_id", actuals_col = "total", set_index = False, verbosity = verbosity)
        elif model == "AutoARIMA":
                predicted_values, fitted_values = train_AutoARIMA_and_forecast(
                df_long, future_periods, freq=freq, date_col = "date", id_col = "ts_id", actuals_col = "total", set_index = False, verbosity = verbosity)
        
        print("Global forecast completed.")
        predicted_dic = transform_long_to_dict(predicted_values, mapping=mapping, id_col='ts_id', date_col='date', actuals_col="pred")
        
        if(test_dic != None):
            for key, df in predicted_dic.items():

                # Extrahiere die Spaltennamen aus dem Tupel (key), um diese gezielt zu füllen
                columns_to_fill = list(key)  # Falls key ein Tupel ist
                columns_to_fill.append('date')
                predicted_dic[key] = test_dic[key].merge(df, on=columns_to_fill, how='outer')
                
                # Auffüllen der ts_id Spalte gemäß der Spalten in "list(key)", die immer den gleichen Wert haben  mit Forward-Fill 
                if 'ts_id' in predicted_dic[key].columns:
                    predicted_dic[key]['ts_id'] = predicted_dic[key].groupby(list(key))['ts_id'].ffill()

        else:   
            for group_key, df in predicted_dic.items():              
                #  Merken der relevanten dates aus predicted_values
                relevant_dates = df['date'].unique()

                # Extrahiere die Spaltennamen aus dem Tupel (key), um diese gezielt zu füllen
                columns_to_fill = list(group_key)  # Falls key ein Tupel ist
                columns_to_fill.append('date')
                predicted_dic[group_key] = train_dic[group_key].merge(df, on=columns_to_fill, how='outer')
                
                # Auffüllen der ts_id Spalte gemäß der Spalten in "list(key)", die immer den gleichen Wert haben  mit Forward-Fill 
                if 'ts_id' in predicted_dic[group_key].columns:
                    predicted_dic[group_key]['ts_id'] = predicted_dic[group_key].groupby(list(group_key))['ts_id'].ffill()
                
                # Filtere den DataFrame, um nur relevante dates zu behalten
                predicted_dic[group_key] = predicted_dic[group_key][predicted_dic[group_key]['date'].isin(relevant_dates)]
                
        #print(predicted_dic[('dataset',)])
                


    # Handle level mode
    elif forecast_method == "level":
        print("Starting level-specific forecast for each group...")
        total_iterations = len(train_dic.keys())
        current_iteration = 0
        for key, df in train_dic.items():
            current_iteration += 1
            print(f"combinations {current_iteration}/{total_iterations}: {key}")
            df = df.copy()

            if train_end_date is not None:
                df = df[df['date'] <= train_end_date]

            if model == "AutoTS":
                #print(df.head())
                #print(df.info())
                predicted_values, fitted_values = train_autots_and_forecast(
                    df, future_periods, freq=freq, includeModels=includeModels,
                    excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit = time_limit)
                #print(predicted_values.head())
            elif model == "AutoGluon":
                predicted_values, fitted_values = train_autogluon_and_forecast(
                    df, future_periods, freq=freq, includeModels=includeModels,
                    excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False, time_limit = time_limit)
            elif model == "AutoETS":
                predicted_values, fitted_values = train_AutoETS_and_forecast(df, future_periods, freq=freq, date_col = "date", id_col = "ts_id", actuals_col = "total", set_index = False, verbosity = verbosity)
            elif model == "AutoARIMA":
                predicted_values, fitted_values = train_AutoARIMA_and_forecast(df, future_periods, freq=freq, date_col = "date", id_col = "ts_id", actuals_col = "total", set_index = False, verbosity = verbosity)

            # Speichern der Vorhersagen für jede Gruppe
            predicted_dic[key] = predicted_values

            if(test_dic != None):
 
                predicted_dic[key] = predicted_dic[key].merge(test_dic[key], on=['ts_id', 'date'], how='outer')

                # Extrahiere die Spaltennamen aus dem Tupel (key), um diese gezielt zu füllen
                  # Falls key ein Tupel ist
                
                # Fülle nur die relevanten Spalten und behalte den Index bei
                #predicted_dic[key] = predicted_dic[key].merge(test_dic[key], on=columns_to_fill, how='outer')

                # Auffüllen der Spalten in "keys" gemäß ihres letzten Wertes mit Forward-Fill
                for col in key:
                    if col in predicted_dic[key].columns:
                        predicted_dic[key][col] = predicted_dic[key].groupby('ts_id')[col].ffill()

            else:
                relevant_dates = predicted_dic[key]['date'].unique()    
                predicted_dic[key] = predicted_dic[key].merge(train_dic[key], on=['ts_id', 'date'], how='outer')

                # Extrahiere die Spaltennamen aus dem Tupel (key), um diese gezielt zu füllen
                  # Falls key ein Tupel ist
                
                # Fülle nur die relevanten Spalten und behalte den Index bei
                #predicted_dic[key] = predicted_dic[key].merge(test_dic[key], on=columns_to_fill, how='outer')

                # Auffüllen der Spalten in "keys" gemäß ihres letzten Wertes mit Forward-Fill
                for col in key:
                    if col in predicted_dic[key].columns:
                        predicted_dic[key][col] = predicted_dic[key].groupby('ts_id')[col].ffill()
                predicted_dic[key] = predicted_dic[key][predicted_dic[key]['date'].isin(relevant_dates)]
                
            #print(predicted_dic[key])


                             

    # Processing results
    if train_end_date is None:
        train_end_date = max([df['date'].max() for df in train_dic.values()])
    else:
        train_end_date = train_end_date

    if test_dic is not None:
        test_end_date = max([df['date'].max() for df in test_dic.values()])

    out_of_sample = ""
    if test_dic is None:
        test_end_date = None
        out_of_sample = "OOS_"

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Zeit in Stunden und Minuten umrechnen
    hours, rem = divmod(elapsed_time, 3600)
    minutes = rem // 60
    
    # Zeit im Format "Stunden:Minuten" ausgeben
    elapsed_time_str = f"{int(hours)}:{int(minutes):02d} h"
    print(f"Total execution time: {elapsed_time_str}")

    # Speichere das aktuelle Datum
    current_date = datetime.now().strftime('%Y-%m-%d')
 
    # Dateiname anpassen
    if isinstance(includeModels, str):
        includeModels = [includeModels]

    if includeModels and len(includeModels) == 1:
        model_suffix = includeModels[0]  # Falls nur ein Modell in der Liste ist, füge es zum Dateinamen hinzu
        file_name = f"{out_of_sample}Forecast_{model}_{model_suffix}_{forecast_method}_{future_periods}_{freq}_{time_limit}.pkl"
    else:
        file_name = f"{out_of_sample}Forecast_{model}_{forecast_method}_{future_periods}_{freq}_{time_limit}.pkl"

    # combined_result um das aktuelle Datum erweitern
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
            train_end_date=train_end_date, includeModels=includeModels, excludeModels=excludeModels, forecast_method=forecast_method, saveResults=saveResults, save_path=save_path)
    }


    if saveResults:
        # Überprüfe, ob das Verzeichnis "weights" innerhalb des save_path existiert, und erstelle es gegebenenfalls
        os.makedirs(save_path, exist_ok=True)

        # Erstelle den vollständigen Pfad für die zu speichernde Datei innerhalb des Ordners "forecasts"
        pkl_filename = os.path.join(save_path, file_name)

        with open(pkl_filename, 'wb') as f:
            pickle.dump(combined_result, f)
        print(f"Results saved in '{pkl_filename}' as a pkl file.")

    return combined_result