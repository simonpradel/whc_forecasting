import pickle
import os

def load_weights(file_names):
    """
    Lädt Gewichtsdaten aus Pickle-Dateien.
    
    :param file_names: Liste der Dateipfade oder ein einzelner Dateipfad (als String) zu den Pickle-Dateien.
    :return: Dictionary mit geladenen Gewichten, wobei der Dateiname (ohne Pfad) als Schlüssel verwendet wird.
    """
    loaded_weights = {}

    # Wenn file_names ein String ist, mache es zu einer Liste
    if isinstance(file_names, str):
        file_names = [file_names]

    # Iteriere über alle Dateinamen
    for file_name in file_names:
        # Lade die Datei
        with open(file_name, 'rb') as f:
            weights_data = pickle.load(f)

        # Speichere die Gewichtsdaten mit dem Dateinamen (ohne Pfad) als Key
        loaded_weights[os.path.basename(file_name)] = weights_data

    return loaded_weights