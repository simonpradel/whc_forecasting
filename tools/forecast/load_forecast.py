import pickle
import os

def load_forecast(file_names, exclude_variables=None):
    """
    Lädt Forecast-Daten aus Pickle-Dateien und entfernt bestimmte Variablen (Keys und Spalten).
    
    :param file_names: Liste der Dateipfade oder ein einzelner Dateipfad (als String) zu den Pickle-Dateien.
    :param exclude_variables: Liste der zu entfernenden Variablen/Spalten.
    :return: Dictionary mit gefilterten Forecasts, wobei der Dateiname (ohne Pfad) als Schlüssel verwendet wird.
    """
    loaded_data = {}

    # Wenn file_names ein String ist, mache es zu einer Liste
    if isinstance(file_names, str):
        file_names = [file_names]

    # Iteriere über alle Dateinamen
    for file_name in file_names:
        # Lade die Datei
        with open(file_name, 'rb') as f:
            forecast_data = pickle.load(f)["predicted_dic"]

            # Kopiere das Dictionary, um während der Iteration Keys entfernen zu können
            filtered_forecast = forecast_data.copy()

            # Wenn `exclude_variables` angegeben ist
            if exclude_variables:
                # Iteriere über die Keys im Forecast-Dictionary
                for key in list(filtered_forecast.keys()):
                    # Prüfe, ob eine der Variablen in den Tupel-Keys enthalten ist
                    if any(var in key for var in exclude_variables):
                        # Entferne diesen Key aus dem Dictionary
                        del filtered_forecast[key]
            
            # Speichere das gefilterte Dictionary mit dem Dateinamen (ohne Pfad) als Key
            loaded_data[os.path.basename(file_name)] = filtered_forecast



    return loaded_data