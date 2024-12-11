import pandas as pd
from functools import reduce

def combine_aggregation_methods(forecast_dicts, date_col='ds'):
    """
    Führt eine Liste von Dictionaries mit DataFrames zusammen, indem die DataFrames anhand ihrer Datumsspalte
    und gleicher Keys zusammengeführt werden. Alle numerischen Spalten (außer der Datumsspalte) werden als 
    Ganzzahlen behandelt.

    :param forecast_dicts: Liste von Dictionaries, die DataFrames enthalten.
    :param date_col: Der Name der Datumsspalte, anhand derer die DataFrames zusammengeführt werden sollen (Standard: 'ds').
    :return: Ein Dictionary mit DataFrames, zusammengeführt anhand der gemeinsamen Keys und Datumsspalte.
    """
    # Setze ein leeres Dictionary für die zusammengeführten Ergebnisse
    merged_forecasts = {}

    # Filtere Dictionaries, um sicherzustellen, dass alle Werte DataFrames sind
    forecast_dicts = [d for d in forecast_dicts if all(isinstance(v, pd.DataFrame) for v in d.values())]

    # Finde alle gemeinsamen Keys über alle Dictionaries hinweg
    common_keys = set.intersection(*[set(d.keys()) for d in forecast_dicts])

    # Verarbeite jeden gemeinsamen Key
    for key in common_keys:
        # Extrahiere die DataFrames für den aktuellen Key aus allen Dictionaries
        dfs_to_merge = [d[key] for d in forecast_dicts]
        
        # Setze sicher, dass die Datumsspalte 'ds' als Index verwendet wird
        dfs_to_merge = [df.set_index(date_col) for df in dfs_to_merge]

        # Führe die DataFrames anhand des Indexes (Datumsspalte) zusammen
        merged_df = reduce(lambda left, right: left.join(right, how='left'), dfs_to_merge)
        
        # Setze den Index zurück und verwende wieder die Datumsspalte 'ds'
        merged_df.reset_index(inplace=True)

        # Stelle sicher, dass alle numerischen Spalten (außer der Datumsspalte) Floats ohne Dezimalstellen sind
        for col in merged_df.columns:
            if col != date_col:
                merged_df[col] = merged_df[col].astype('float').astype('int')

        # Speichere das zusammengeführte DataFrame im Ergebnis-Dictionary
        merged_forecasts[key] = merged_df

    return merged_forecasts
