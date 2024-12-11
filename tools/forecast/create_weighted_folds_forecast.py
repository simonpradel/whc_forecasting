import pandas as pd
import pickle
from tools.methods.create_weighted_forecast import create_weighted_forecast
from tools.models.Naive import train_Naive_and_forecast
from tools.methods.get_function_args import get_function_args
import time
import os

def rolling_weighted_forecast(train_dic, test_dic, weights_dict, folds, freq, model="Naive", includeModels=None, excludeModels=None, selection_mode="selected_only", saveResults=True, save_path = os.getcwd()):
    """
    Erstellt ein Rolling Forecast mit mehreren Folds, wobei der Trainingsdatensatz in jedem Fold erweitert wird.

    Parameters:
    ----------
    train_dic : dict
        Dictionary mit den Trainingsdaten.
    test_dic : dict
        Dictionary mit den Testdaten.
    selected_groups : list
        Liste der ausgewählten Gruppen.
    weights : list
        Liste der Gewichte für die ausgewählten Gruppen.
    folds : int
        Anzahl der Folds.
    freq : str
        Frequenz der Zeitreihen (z.B. 'D' für täglich, 'M' für monatlich).
    model : str, optional
        Zu verwendendes Modell. Standardmäßig "Naive".
    includeModels : list, optional
        Modelle, die in create_weighted_forecast verwendet werden sollen.
    excludeModels : list, optional
        Modelle, die in create_weighted_forecast ausgeschlossen werden sollen.

    Returns:
    -------
    pd.DataFrame, dict
        DataFrame mit den Forecasts für jeden Fold und eine Spalte 'fold'.
        Dictionary, das die zusammengeführten DataFrames aus predicted_dic enthält, jeweils auch mit einer Spalte 'fold'.
    """

    start_time = time.time()

    # Länge des Testdatensatzes basierend auf der Anzahl der Zeilen im Test-Dictionary
    test_length = min([len(df) for df in test_dic.values()])
    
    # Bestimme die Länge eines Folds, indem der Testdatensatz gleichmäßig auf die Folds aufgeteilt wird
    fold_length = test_length // folds  

    # Hol das letzte Datum aus den Trainingsdaten
    last_train_date = train_dic[('dataset',)]['date'].max() 

    # Initialisiere einen leeren DataFrame und ein leeres Dictionary für die Ergebnisse
    final_forecast_df = train_dic[('dataset',)].copy()
    combined_predicted_dic = train_dic.copy()

    final_forecast_df = final_forecast_df[final_forecast_df['date'] <= last_train_date]

    for group_key, pred_df in combined_predicted_dic.items():
    # Filtere die Predicted-Daten nach dem letzten Trainingsdatum
        combined_predicted_dic[group_key] = combined_predicted_dic[group_key][combined_predicted_dic[group_key]['date'] <= last_train_date]

    fold_train_end_date = last_train_date

    # Iteriere über die Folds
    for fold in range(folds):

        print(f"Fold {fold+1}/{folds} - Training bis: {fold_train_end_date}")

        # Kopiere das Training-Dictionary und erweitere es mit den Testdaten bis zur Länge des Folds
        extended_train_dic = {}

        for group_key, train_df in train_dic.items():
            # Kombiniere die Training-Daten mit einem Teil der Testdaten bis zur aktuellen Fold-Länge
            train_df = train_dic[group_key].copy()
            test_df = test_dic[group_key].copy()

            # Wähle Testdaten bis zum aktuellen Fold-Ende basierend auf dem Datum
            fold_test_data = test_df[test_df['date'] <= fold_train_end_date]

            # Füge die ausgewählten Testdaten den Trainingsdaten hinzu
            extended_train_df = pd.concat([train_df, fold_test_data], axis=0).reset_index(drop=True)
            extended_train_dic[group_key] = extended_train_df

        # Bestimme die Anzahl der Zukunftsperioden (entspricht einem Fold)
        future_periods = fold_length

        # Erzeuge die Forecasts und predicted_dic
        top_level_forecast_df, predicted_dic = create_weighted_forecast(
            extended_train_dic,
            test_dic,
            weights_dict,
            future_periods=future_periods,
            freq=freq,
            model=model,
            includeModels=includeModels,
            excludeModels=excludeModels,
            selection_mode=selection_mode,
            saveResults=False
        )

        # Filtere die Forecast-Daten nach dem letzten Trainingsdatum
        filtered_forecast_df = top_level_forecast_df[top_level_forecast_df['date'] > fold_train_end_date]

        # Wähle die ersten k Zeilen gemäß der future_periods
        filtered_forecast_df = filtered_forecast_df.head(future_periods)
        filtered_forecast_df['fold'] = fold + 1

        # Füge den gefilterten Forecast zu final_forecast_df hinzu
        final_forecast_df = pd.concat([final_forecast_df, filtered_forecast_df], axis=0)

        # Verarbeite auch predicted_dic
        for group_key, pred_df in predicted_dic.items():
            # Filtere die Predicted-Daten nach dem letzten Trainingsdatum
            filtered_pred_df = pred_df[pred_df['date'] > fold_train_end_date]

            # Jetzt gruppiere nach 'ts_id' und wähle für jede Gruppe die ersten k Zeilen aus
            filtered_pred_df = filtered_pred_df.groupby('ts_id').head(future_periods).reset_index(drop=True)
  
            filtered_pred_df['fold'] = fold + 1

            # Füge den gefilterten DataFrame zum jeweiligen DataFrame in combined_predicted_dic hinzu
            combined_predicted_dic[group_key] = pd.concat([combined_predicted_dic[group_key], filtered_pred_df], axis=0)

        # Definiere das Enddatum des Trainingsdatensatzes für den aktuellen Fold
        fold_train_end_date = fold_train_end_date + pd.DateOffset(days=fold_length)



    # Setze die Indizes zurück
    final_forecast_df = final_forecast_df.reset_index(drop=True)
    for group_key in combined_predicted_dic:
        combined_predicted_dic[group_key] = combined_predicted_dic[group_key].reset_index(drop=True)

    # Zeit in Stunden und Minuten umrechnen
    hours, rem = divmod(elapsed_time, 3600)
    minutes = rem // 60
    
    # Zeit im Format "Stunden:Minuten" ausgeben
    elapsed_time_str = f"{int(hours)}:{int(minutes):02d} h"
    print(f"Total execution time: {elapsed_time_str}")


    # Speichere das Ergebnis als CSV, falls saveResults aktiviert ist
    if saveResults:
        combined_result = {
            "final_forecast_df": final_forecast_df,
            "combined_predicted_dic": combined_predicted_dic,
            "Input_Arguments": get_function_args(rolling_weighted_forecast, train_dic, test_dic, weights_dict, folds, freq, model, includeModels, excludeModels, selection_mode, saveResults, save_path),
            "elapsed_time": elapsed_time_str
        }

        file_name = f"Fold_Forecast_{model}_{test_length}_{freq}_{folds}.pkl"

        # Überprüfe, ob das Verzeichnis existiert, und erstelle es gegebenenfalls
        os.makedirs(save_path, exist_ok=True)

        # Erstelle den vollständigen Pfad für die zu speichernde Datei
        pkl_filename = os.path.join(save_path, file_name)

        with open(pkl_filename, 'wb') as f:
            pickle.dump(combined_result, f)
        print(f"Results saved in '{pkl_filename}' as a pkl file.")


    return final_forecast_df, combined_predicted_dic

