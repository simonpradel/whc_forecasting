import pandas as pd

def split_data(data, period, unique_id, format="dictionary", set_index=False):
    """
    Teilt die Daten in Trainings- und Testdaten basierend auf der gegebenen Periodendauer und dem gewählten Format.

    Parameters:
    data (dict): Ein Dictionary, das einen oder mehrere DataFrames enthält.
                 Erwartet wird ein Schlüssel 'top_level' für den top-level DataFrame
                 und weitere Schlüssel für die lower-level DataFrames.
    period (int): Die Anzahl der letzten Zeilen pro 'unique_id', die zum Testen verwendet werden.
    unique_id (str): Der Spaltenname, der als 'unique_id' verwendet wird, um Gruppen zu identifizieren.
    format (str): Das Format, in dem die Daten gesplittet werden. Kann 'dictionary' oder 'long' sein.
    set_index (bool): Wenn True, wird die Spalte 'unique_id' als Index gesetzt.

    Returns:
    dict oder tuple: Trainings- und Testdaten für die lower-level DataFrames und den top-level DataFrame,
                     entweder als Dictionary (bei format='dictionary') oder als DataFrames (bei format='long').
    """

    if format == "dictionary":
        # Initialisiere Dictionaries für Trainings- und Testdaten
        train_data = {}
        test_data = {}

        # Teile die Daten basierend auf der Anzahl der Perioden auf
        for key, df in data.items():
            # Für jede unique_id die letzten 'period' Einträge als Testdaten verwenden
            test_data[key] = df.groupby(unique_id).tail(period)
            train_data[key] = df.drop(test_data[key].index)
            
            # Setze den Index auf 'unique_id', falls gewünscht
            if set_index:
                train_data[key] = train_data[key].set_index(unique_id)
                test_data[key] = test_data[key].set_index(unique_id)
        
        return train_data, test_data
    
    elif format == "long":
        # Initialisiere DataFrames für Trainings- und Testdaten
        data = data['Y_df']

        # Splitte die Daten basierend auf der Anzahl der Perioden pro 'unique_id'
        test_df = data.groupby(unique_id).tail(period)
        train_df = data.drop(test_df.index)

        # Setze den Index auf 'unique_id', falls gewünscht
        if set_index:
            test_df = test_df.set_index(unique_id)
            train_df = train_df.set_index(unique_id)

        return train_df, test_df

    else:
        raise ValueError("Das Format muss entweder 'dictionary' oder 'long' sein.")

