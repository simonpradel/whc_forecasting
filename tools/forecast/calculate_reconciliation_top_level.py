def calculate_reconciliation_top_level(loaded_forecasts):
    """
    Verarbeitet die DataFrames in einem Dictionary, indem die Spalte ohne "/" als 'top-level' umbenannt wird
    und bei den anderen Spalten das Präfix entfernt wird.
    
    :param loaded_forecasts: Dictionary, das DataFrames enthält, auf die die Verarbeitung angewendet wird.
    :return: Dictionary mit den gleichen Keys wie im Input, wobei die DataFrames angepasst wurden.
    """

    def filter_rows(df):
        # Filtere nur die Zeilen, deren 'unique_id' keinen "/" enthält
        filtered_df = df[~df.index.str.contains('/')].reset_index(drop=True)
        return filtered_df
    
    def process_df(df):
        # Schritt 1: Identifizieren der Spalte ohne "/"
        target_col = None
        for col in df.columns:
            if "/" not in col and col not in ["ds", "y"]:
                target_col = col
                break

        # Schritt 2: Umbenennen der Spalte in 'top-level'
        if target_col:
            df.rename(columns={target_col: 'top-level'}, inplace=True)

        # Schritt 3: Entfernen des Präfixes bei allen anderen Spalten
        def remove_prefix(col):
            return col.split('/', 1)[-1] if "/" in col else col
        
        df.columns = [remove_prefix(col) for col in df.columns]

        df = filter_rows(df)

        # Rückgabe des neuen DataFrames
        return df, target_col

    # Dictionary zur Speicherung der Ergebnisse
    result_dict = {}

    # Verarbeitung der DataFrames in jedem Key des Dictionaries
    for key, df in loaded_forecasts.items():
        processed_df, target_col = process_df(df)
        result_dict[key] = processed_df

    # Rückgabe des neuen Dictionarys mit den angepassten DataFrames
    return result_dict
