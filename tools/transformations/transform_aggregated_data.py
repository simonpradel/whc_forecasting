import pandas as pd
import numpy as np


def transform_long_to_dict(df: pd.DataFrame, mapping: dict, id_col: str = 'unique_id', date_col: str = 'ds', actuals_col: str = 'y') -> dict:
    """
    Transformiert einen DataFrame vom Long-Format in ein Dictionary von DataFrames, basierend auf einem Mapping.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame im Long-Format, der mindestens die Spalte enthält, die durch `id_col` definiert ist.
    mapping : dict
        Dictionary, das Spaltennamen und ihre zugehörigen `id_col`-Werte beschreibt.
    id_col : str
        Der Name der Spalte, die die IDs enthält. Standardmäßig 'unique_id'.
    date_col : str
        Der Name der Spalte, die Datumsinformationen enthält. Standardmäßig 'ds'.
    actuals_col : str
        Der Name der Spalte, die die abhängige Variable enthält. Standardmäßig 'y'.

    Returns:
    -------
    dict
        Dictionary, das gefilterte und aufgeteilte DataFrames enthält. Der Key des Dictionaries
        ist ein Tupel, das die Namen der neuen Spalten repräsentiert (z.B. ('region', 'subregion')).
    """
    dataframes = {}

    for key, values in mapping.items():
        # Zerlege den Key in Spaltennamen
        columns = key.split('/')
        # Filtere die Daten für jede unique_id in der Mapping-Tabelle
        filtered_df = df[df[id_col].isin(values)].copy()

        # Gebe die gefilterten unique_id-Werte aus, die in das DataFrame gepackt werden
        #print(f"Für Key: {key}, die folgenden unique_id-Werte werden verwendet: {filtered_df[id_col].unique()}")

        if not filtered_df.empty:
            # Erstelle ein DataFrame mit den entsprechenden Spalten
            df_split = filtered_df.copy()
            for col in columns:
                df_split[col] = df_split[id_col].apply(lambda x: x.split('/')[columns.index(col)] if len(x.split('/')) > columns.index(col) else None)

            # Entferne die originale 'id_col'-Spalte
            df_split.drop(columns=id_col, inplace=True)

            # Erstelle den Schlüssel als Tupel
            key_tuple = tuple(columns)

            # Setze den neuen DataFrame in das Dictionary
            dataframes[key_tuple] = df_split

    return dataframes




def transform_dict_to_long(dataframes: dict, id_col: str = 'unique_id', date_col: str = 'ds', actuals_col: str = 'y', set_index=False, include_all_columns=False) -> tuple:
    """
    Transformiert ein Dictionary von DataFrames zurück in ein DataFrame im Long-Format.

    Parameters:
    ----------
    dataframes : dict
        Ein Dictionary, bei dem der Key ein Tupel von Spaltennamen ist (z.B. ('region', 'subregion')) 
        und der Value ein DataFrame ist, der mindestens die folgenden Spalten enthält:
        - Die Spalten, die den Key repräsentieren (z.B. 'region', 'subregion').
        - `date_col`: Spalte, die Datumsinformationen enthält.
        - `actuals_col`: Spalte, die die abhängige Variable enthält.
    id_col : str
        Der Name der Spalte, die die IDs enthält. Standardmäßig 'unique_id'.
    date_col : str
        Der Name der Spalte, die Datumsinformationen enthält. Standardmäßig 'ds'.
    actuals_col : str
        Der Name der Spalte, die die abhängige Variable enthält. Standardmäßig 'y'.
    set_index : bool
        Wenn True, wird die `id_col`-Spalte als Index gesetzt. Standardmäßig False.
    include_all_columns : bool
        Wenn True, werden alle Spalten des DataFrames beibehalten. Standardmäßig False.

    Returns:
    -------
    tuple
        - mapping : dict
          Mapping, das die Kombination von Spaltennamen (als Key) mit den kombinierten `id_col`-Werten (als Value) beschreibt.
        - combined_long_format_df : pd.DataFrame
          Ein DataFrame im Long-Format, der die Spalten enthält:
          - `id_col`: Eine kombinierte ID, die durch das Zusammenfügen der Spaltenwerte erstellt wurde.
          - `date_col`: Die Datumsinformationen.
          - `actuals_col`: Die abhängige Variable (z.B. Metriken oder Zielvariable).
          - Alle anderen Spalten, falls `include_all_columns` True ist.
    """
    # Mapping-Tabelle initialisieren
    mapping = {}

    # Liste für die gesammelten DataFrames im Longformat
    long_format_list = []

    # Durch alle DataFrames im Eingabedictionary iterieren
    for key, df in dataframes.items():

        df = pd.DataFrame(df)
        
        # Extrahiere die Spaltennamen für den aktuellen DataFrame
        columns = key if isinstance(key, tuple) else (key,)

        # Füge die Spaltennamen als Key für die Mapping-Tabelle hinzu
        mapping_key = '/'.join(columns)

        # Kombiniere die Werte der beteiligten Spalten und speichere sie als ein Array von Strings
        if all(col in df.columns for col in columns):
            combined_values = df[list(columns)].astype(str).agg('/'.join, axis=1).unique()
            mapping[mapping_key] = np.array(combined_values)

        # Erzeuge den Longformat DataFrame mit den Spalten: id_col, date_col, actuals_col
        df_long = pd.DataFrame()
        if all(col in df.columns for col in [date_col, actuals_col]):
            # Generiere die unique_id basierend auf den Kombinationen aus der Mapping-Tabelle
            df_long[id_col] = df[list(columns)].astype(str).agg('/'.join, axis=1)
            df_long[date_col] = df[date_col]
            df_long[actuals_col] = df[actuals_col]

            if include_all_columns:
                # Füge alle anderen Spalten hinzu, außer die bereits vorhandenen (id_col, date_col, actuals_col)
                other_columns = df.columns.difference([id_col, date_col, actuals_col])
                # Füge jede dieser Spalten einzeln zu df_long hinzu, um die richtige Ausrichtung sicherzustellen
                for col in other_columns:
                    df_long[col] = df[col].values

   
        # Füge den Longformat DataFrame zur Liste hinzu
        long_format_list.append(df_long)
    
    # Kombiniere alle DataFrames in der Liste zu einem einzigen DataFrame
    combined_long_format_df = pd.concat(long_format_list, ignore_index=True)
    
    if set_index:
        # Setze den Index und entferne die Spalte
        combined_long_format_df.set_index(id_col, inplace=True, drop=True)

    # Rückgabe der Mapping-Tabelle und des kombinierten DataFrames
    return mapping, combined_long_format_df

