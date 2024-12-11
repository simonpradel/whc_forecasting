import numpy as np
import pandas as pd 

def calculate_metric(df, column, metric, verbosity, test_period, future_periods):
    """Berechnet die angegebene Metrik."""
    
    df = df.copy()  # Sicherstellen, dass an einer Kopie gearbeitet wird, um Warnungen zu vermeiden

    df_extracted = df[['y', column]]
    df_actuals = df[['y', column]]
    # Entferne Nullwerte in der Spalte 'method', die als Forecast-Horizont dienen
    #df_actuals = df_extracted.dropna(subset=["y"]) # includes actuals + pred
    #df_pred = df_extracted.dropna(how='any', inplace=False) # only pred
    #print("df_no_nan_method")
    #print(df_actuals)
    forecast_horizon_length = test_period

    # y_train sind alle vorherigen Werte in 'y' (außer den letzten 'forecast_horizon_length' Werten)
    y_train = df_actuals['y'].iloc[:-future_periods]

    # Teile y in y_train und y_test auf
    # y_test ist der letzte Teil der Spalte 'y' mit Länge des Forecast-Horizonts
    out_of_sample_predictions = future_periods - test_period


    # Auswahl der letzten Zeilen für die Vorhersagezeiträume
    df_pred_inc_oos = df_actuals.iloc[-future_periods:]
    print(df_pred_inc_oos)
    # Definiere y_test und y_pred
    if out_of_sample_predictions:  # Prüft, ob out_of_sample_predictions einen Wert hat (nicht leer, nicht 0)
        y_test = df_pred_inc_oos['y'][:-out_of_sample_predictions]
        y_pred = df_pred_inc_oos[column][:-out_of_sample_predictions]
    else:
        y_test = df_pred_inc_oos['y']
        y_pred = df_pred_inc_oos[column]



    # Entferne Zeilen, bei denen y_pred NaN-Werte enthält
    non_nan_indices = y_pred.dropna().index
    y_test = y_test.loc[non_nan_indices]
    y_pred = y_pred.loc[non_nan_indices]

    # Entferne Zeilen, bei denen y_test NaN-Werte enthält
    non_nan_indices = y_test.dropna().index
    y_test = y_test.loc[non_nan_indices]
    y_pred = y_pred.loc[non_nan_indices]

    # Überprüfe, ob y_pred und y_test gleich lang sind
    if ( verbosity >= 3):
        if len(y_pred) == len(y_test):
            # Überprüfe, ob die Indizes von y_pred und y_test übereinstimmen
            if all(y_pred.index != y_test.index):
                raise ValueError("y_pred und y_test sind gleich lang, aber die Indizes stimmen nicht überein.")
        else:
            raise ValueError("y_pred und y_test sind nicht gleich lang.")

        # Ausgabe: y_train, y_test, y_pred
        if ( verbosity >= 5):
            print("metric:\n", metric)
            print("y_train:\n", y_train)
            print("y_test:\n", y_test)
            print("y_pred:\n", y_pred)
    
    # MAE, MAPE, MSE, RMSE, MASE, RMSSE, SMAPE, WAPE
    if metric == 'MAE':
        true_deviation = y_test - y_pred
        return abs(true_deviation).mean()

    elif metric == 'MAPE':
        #from sklearn.metrics import mean_absolute_percentage_error
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        mape = MeanAbsolutePercentageError(y_test, y_pred)
        return mape

    elif metric == 'MSE':
        #from sklearn.metrics import mean_squared_error
        from sktime.performance_metrics.forecasting import MeanSquaredError
        mse = MeanSquaredError(y_test, y_pred)
        return(mse)

    elif metric == 'RMSE':
        #from sklearn.metrics import root_mean_squared_error 
        from sktime.performance_metrics.forecasting import MeanSquaredError
        rmse = MeanSquaredError(y_test, y_pred, square_root=True)
        return(rmse)

    elif metric == 'MASE':
        #mase = np.mean(np.abs(y_pred - y_test)) / np.mean(np.abs(y_train[:-forecast_horizon_length] - y_train[forecast_horizon_length:]))
        from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
        results = MeanAbsoluteScaledError()
        mase = results(y_test, y_pred, y_train=y_train)
        return(mase)

    elif metric == 'RMSSE':
        from sktime.performance_metrics.forecasting import MeanSquaredScaledError
        results = MeanSquaredScaledError(square_root=True)
        rmsse = results(y_test, y_pred, y_train=y_train)
        return(rmsse)

    elif metric == 'SMAPE':
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        results = MeanAbsolutePercentageError(symmetric=True)
        smape = results(y_test, y_pred)
        return(smape)

    elif metric == 'WAPE':
        absolute_deviation = abs(y_test - y_pred)
        wape = absolute_deviation.sum() / abs(y_test).sum()
        return wape

    else:
        raise ValueError(f"Unbekannte Metrik: {metric}")


def calculate_metrics(final_result, method_list, metrics, dataset_name, date_column = 'date', actual_column = 'y', verbosity = 0, test_period = None, future_periods = None):
    # Leere Liste, um die Ergebnisse zu speichern
    results = []
    
    models = list(final_result.keys())
    if (verbosity>3):
        print("/fct: calculate_metrics: ")
        print(models)
    # Iteriere über jedes Modell im final_result
    for model in models:
        # Extrahiere die Optimierungsmethode und Vorhersagemethode aus den Input_Arguments
        optim_method = final_result[model]["weights_dict"]["Input_Arguments"].get('optim_method', None)
        forecast_method = final_result[model]["weights_dict"]["Input_Arguments"].get('forecast_method', None)
        forecast_model = final_result[model]["weights_dict"]["Input_Arguments"].get('model', None)
        elapsed_time = final_result[model]['weights_dict']["meta_data"].get('elapsed_time', None) + final_result[model]["forecast_dict"]["meta_data"].get('elapsed_time', None)
        time_limit = final_result[model]['weights_dict']["meta_data"]['time_limit'] if 'time_limit' in final_result[model]['weights_dict']["meta_data"] else None


        model_data = final_result[model]["combined_results"][('dataset',)]
        
        # Filtere den DataFrame, um nur Zeilen zu behalten, bei denen sowohl Actuals als auch Vorhersagen vorhanden sind
        #model_data = model_data.dropna(subset=[actual_column] + method_list)
        
        # Berechne die angegebenen Metriken für jede Methode
        for method in method_list:
            for metric in metrics:
                # Berechne den Metrikwert für die jeweilige Methode und Metrik
                if (verbosity>=5):
                    print("calculate_metric: ", metric)
                metric_value = calculate_metric(model_data, method, metric, verbosity, test_period, future_periods)
                if (verbosity>=5):
                    print("calculate_metric End: ", metric, metric_value)
                if dataset_name in ("hardware_revenue", "mobile_service_revenue", "fbb_fixed_other_revenue", "cos", "commercial_costs", "bad_debt", "non_commercial_costs", "non_recurrent_income_cost"):
                    dataset_type = "private"
                else:
                    dataset_type = "public"

                # Speichere das Ergebnis als neue Zeile
                results.append({
                    #"dataset": 
                    'Model': model,
                    'optim_method': optim_method,
                    'forecast_method': forecast_method,
                    'forecast_model': forecast_model,
                    'elapsed_time' : elapsed_time,
                    'method': method,
                    'metric': metric,
                    'dataset': dataset_name,
                    'dataset_type': dataset_type,
                    'time_limit': time_limit,
                    'value': metric_value
                })
    
    # Erstelle einen DataFrame mit den Ergebnissen
    results_df = pd.DataFrame(results)
    
    return results_df

