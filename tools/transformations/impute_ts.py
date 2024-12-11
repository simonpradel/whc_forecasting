import pandas as pd

def add_missing_rows(dataframes_dict):
    updated_dataframes = {}
    
    # Sammle alle Daten aus allen DataFrames in einem einzigen DataFrame
    all_data = pd.concat(dataframes_dict.values(), ignore_index=True)
    
    # Bestimme das Mindest- und Höchstdatum über alle Daten
    min_date = all_data['date'].min()
    max_date = all_data['date'].max()
    date_range = pd.date_range(min_date, max_date, freq='M') 
    
    for df_name, df in dataframes_dict.items():
        # Sortiere nach ts_id und date
        df_sorted = df.sort_values(by=['ts_id', 'date']).reset_index(drop=True)
        
        # Erstelle ein leeres DataFrame zum Sammeln der aktualisierten Daten
        updated_rows = []
        
        # Schleife über jede ts_id und ergänze fehlende Monate
        for ts_id, df_ts in df_sorted.groupby('ts_id'):
            # Finde fehlende Monate für diese ts_id
            existing_dates = df_ts['date'].unique()
            existing_dates = pd.to_datetime(existing_dates)  # Konvertiere zu DateTime-Objekten
            missing_dates = date_range.difference(existing_dates)
            
            for date in missing_dates:
                # Erstelle eine neue Zeile für diesen Monat
                new_row = df_ts.iloc[-1].copy()  # Kopiere die letzte vorhandene Zeile für diese ts_id
                new_row['date'] = date  # Setze das Datum auf das Monatsende
                new_row['total'] = 0  # Setze den total-Wert auf 0
                
                # Füge die neue Zeile zur Sammlung hinzu
                updated_rows.append(new_row)
        
        # Füge die aktualisierten Zeilen zum ursprünglichen DataFrame hinzu
        updated_df = pd.concat([df_sorted, pd.DataFrame(updated_rows)], ignore_index=True)
        
        # Füge das aktualisierte DataFrame zur Liste hinzu
        updated_dataframes[df_name] = updated_df
    
    return updated_dataframes


    