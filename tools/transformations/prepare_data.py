import pandas as pd

def prepare_data(data, cutoff_date=None, fill_missing_rows=False):
    """
    Process the data based on the given conditions.

    Parameters:
    - data (dict): Dictionary containing the data from function "load_data_from_catalog".
    - cutoff_date (str or None): String in date format to apply a cutoff or None if no cutoff should be applied.
    - fill_missing_rows (bool): Whether to fill missing rows with a specific frequency.

    Returns:
    - dict: Dictionary with processed pandas DataFrame under key "pandas_df".
    """

    data = data.copy()

    # Convert Spark DataFrame to Pandas DataFrame
    if "Telefonica" in data["datasetNameGeneral"]:
        spark_df = data['original_dataset']
        pandas_df = prepare_dataframe(spark_df, data["target_column_name"]).copy()  # Copy to avoid modifying the original
    elif "test" in data["datasetNameGeneral"]:
        spark_df = data['original_dataset']
        pandas_df = prepare_dataframe(spark_df, data["target_column_name"]).copy()  # Copy to avoid modifying the original
    else:
        spark_df = data['original_dataset']
        pandas_df = spark_df.toPandas().copy()  # Ensure a deep copy is made

    # Apply cutoff if provided
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        pandas_df['date'] = pd.to_datetime(pandas_df['date'])  # Ensure 'date' is a datetime column
        pandas_df = pandas_df[pandas_df['date'] <= cutoff_date].copy()  # Copy to ensure the filtered DataFrame is new

    # Fill NA's with zeros
    if fill_missing_rows:
        len_before = len(pandas_df)
        pandas_df = add_missing_rows(pandas_df, freq=data["freq"]).copy()  # Copy to ensure a new DataFrame is created
        #print(f"{len(pandas_df) - len_before} rows were added to the dataset")
        pandas_df['total'] = pandas_df['total'].fillna(0)

    # Create 'ts_id' based on grouping variables with integer values
    grouping_vars = data.get("grouping_variables", [])

    if grouping_vars:
        # Generate a unique integer ts_id for each combination of grouping variables
        pandas_df['ts_id'] = pandas_df.groupby(grouping_vars, observed =False ).ngroup() + 1
    else:
        raise ValueError("No grouping variables found in 'data' dictionary to create 'ts_id'.")

    # Store the processed DataFrame back in the data dictionary
    data["pandas_df"] = pandas_df  

    return data





def add_missing_rows(df, freq):
    """
    Ergänzt fehlende Daten in einem DataFrame basierend auf der angegebenen Frequenz
    und setzt die Werte der 'total'-Spalte für diese Zeitperioden auf 0.

    Parameters:
    - df (pd.DataFrame): Das zu ergänzende DataFrame, welches die Spalten 'ts_id', 'date' und 'total' enthält.
    - freq (str): Die Frequenz der Zeitreihe. Beispiele: 'D' für täglich, 'M' für monatlich, 'Q' für vierteljährlich.

    Returns:
    - pd.DataFrame: Das ergänzte DataFrame mit fehlenden Zeitperioden.
    """
    
    # Sicherstellen, dass 'date' eine Datetime-Spalte ist
    df['date'] = pd.to_datetime(df['date'])

    # Sortiere nach ts_id und date
    df_sorted = df.sort_values(by=['ts_id', 'date']).reset_index(drop=True)

    # Bestimme das Mindest- und Höchstdatum im DataFrame
    min_date = df_sorted['date'].min()
    max_date = df_sorted['date'].max()

    date_range = pd.date_range(min_date, max_date, freq=freq)

    # Erstelle ein leeres DataFrame zum Sammeln der aktualisierten Daten
    updated_rows = []

    # Schleife über jede ts_id und ergänze fehlende Zeitperioden
    for ts_id, df_ts in df_sorted.groupby('ts_id'):
        # Finde fehlende Zeitperioden für diese ts_id
        existing_dates = df_ts['date'].unique()
        existing_dates = pd.to_datetime(existing_dates)  # Konvertiere zu DateTime-Objekten
        missing_dates = date_range.difference(existing_dates)

        for date in missing_dates:
            # Erstelle eine neue Zeile für diese Zeitperiode
            new_row = df_ts.iloc[-1].copy()  # Kopiere die letzte vorhandene Zeile für diese ts_id
            new_row['date'] = date  # Setze das Datum auf den fehlenden Zeitraum
            new_row['total'] = 0  # Setze den total-Wert auf 0

            # Füge die neue Zeile zur Sammlung hinzu
            updated_rows.append(new_row)

    # Füge die aktualisierten Zeilen zum ursprünglichen DataFrame hinzu
    updated_df = pd.concat([df_sorted, pd.DataFrame(updated_rows)], ignore_index=True)
    
    return updated_df



def prepare_dataframe(spark_df, target_column):
    # Spark DataFrame in Pandas DataFrame umwandeln
    pandas_df = spark_df.toPandas()
    
    # Schrägstriche in allen Feldern durch Unterstriche ersetzen
    pandas_df = pandas_df.replace(to_replace=r'/', value='_', regex=True)

    # 'year' und 'month' Spalten bereinigen und konvertieren
    pandas_df['Year'] = pandas_df['Year'].str.extract(r'(\d{4})')
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
        "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
        "NOV": "11", "DEC": "12"
    }
    pandas_df['Period'] = pandas_df['Period'].str.strip().map(month_map)
    
    # Erstellen des letzten Tags eines Monats
    pandas_df['date'] = pd.to_datetime(pandas_df['Year'] + '-' + pandas_df['Period'])
    pandas_df['date'] = pandas_df['date'] + pd.offsets.MonthEnd(0)
    
    # Zielspalte in float umwandeln
    pandas_df.rename(columns={target_column: 'total'}, inplace=True)
    
    # Unnötige Spalten droppen
    pandas_df = pandas_df.drop(columns=['Year', 'Period'])
    
    # Kategorische Spalten konvertieren
    columns_to_convert = pandas_df.columns[~pandas_df.columns.str.contains('date|total')]
    pandas_df[columns_to_convert] = pandas_df[columns_to_convert].astype('category')
    
    # Gruppenvariablen bestimmen und 'ts_id' erstellen
    ts_vars = pandas_df.columns[~pandas_df.columns.str.contains('date|total|id')].tolist()
    pandas_df['ts_id'] = pandas_df.groupby(ts_vars, observed =False ).ngroup()
    
    return pandas_df



