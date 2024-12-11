import pandas as pd

def count_ts_in_dataframe(df: pd.DataFrame) -> None:
    """
    Funktion zur Anzeige der Anzahl von einzigartigen Zeitreihen in einem spezifischen DataFrame.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Der DataFrame, der mindestens die Spalte 'ts_id' enthalten muss, 
        um die Anzahl der einzigartigen Zeitreihen zu zählen.
        - 'ts_id': Die ID der Zeitreihe.
        
    Returns:
    -------
    None
        Die Funktion gibt die Anzahl der einzigartigen Zeitreihen im DataFrame aus.
    """
    # Überprüfen, ob die erforderliche Spalte 'ts_id' vorhanden ist
    if 'ts_id' not in df.columns:
        raise ValueError("Der DataFrame muss die Spalte 'ts_id' enthalten.")
    
    # Anzahl der einzigartigen Zeitreihen berechnen
    num_ts = df['ts_id'].nunique()
    
    # Ausgabe der Anzahl der Zeitreihen
    print(f"Der DataFrame hat {num_ts} einzigartige Zeitreihen.")



def get_ts_info(df: pd.DataFrame, sort_by_total_sum: bool = False) -> pd.DataFrame:
    """
    Funktion zur Rückgabe der Zeitreiheninformationen für einen spezifischen DataFrame.
    Enthält die Anzahl der Ausprägungen, Anfangs- und Enddaten sowie den Gesamtwert der Target-Spalte für jede Zeitreihe.
    Optional kann nach 'total_sum' sortiert werden.

    Parameters:
    ----------
    df : pd.DataFrame
        Der DataFrame, der mindestens die folgenden Spalten enthalten muss:
        - 'ts_id': Die ID der Zeitreihe.
        - 'date': Das Datum der Zeitreihe.
        - 'total': Der Wert der Zielvariable.

    sort_by_total_sum : bool, optional
        Ob die Ergebnisse nach 'total_sum' sortiert werden sollen. Standard ist False.

    Returns:
    -------
    pd.DataFrame
        DataFrame mit den Informationen zu jeder Zeitreihe.
        Enthält die Spalten:
        - 'ts_id': ID der Zeitreihe.
        - 'count': Anzahl der Datenpunkte in der Zeitreihe.
        - 'start_date': Das früheste Datum in der Zeitreihe.
        - 'end_date': Das späteste Datum in der Zeitreihe.
        - 'total_sum': Gesamtwert der Zielvariable für die Zeitreihe.
        - 'percentage_of_total': Prozentsatz der Zeitreihe am Gesamtwert der Zielvariable, wenn `sort_by_total_sum` True ist.
    """
    # Identifizieren der Gruppenvariablen (alles außer 'date', 'total' und 'ts_id')
    group_vars = df.columns[~df.columns.isin(['date', 'total', 'ts_id'])].tolist()

    # Einzigartige Zeitreihen
    unique_ts = df[group_vars + ['ts_id']].drop_duplicates().reset_index(drop=True)
    
    # Anzahl der Ausprägungen pro Zeitreihe
    count_per_ts = df.groupby('ts_id').size().reset_index(name='count')
    
    # Anfangs- und Enddaten pro Zeitreihe
    start_end_dates = df.groupby('ts_id')['date'].agg(['min', 'max']).reset_index()
    start_end_dates.rename(columns={'min': 'start_date', 'max': 'end_date'}, inplace=True)
    
    # Gesamtwert der Zielvariable pro Zeitreihe
    total_sum_per_ts = df.groupby('ts_id')['total'].sum().reset_index(name='total_sum')

    # Alle Informationen zusammenführen
    ts_info = unique_ts.merge(count_per_ts, on='ts_id')
    ts_info = ts_info.merge(start_end_dates, on='ts_id')
    ts_info = ts_info.merge(total_sum_per_ts, on='ts_id')

    # Gesamtwert der Spalte 'total_sum'
    total_sum = ts_info['total_sum'].sum()

    # Ensure ts_info['total_sum'] is a float type
    # Prozentsatz jeder Zeitreihe am Gesamtwert
    # Convert ts_info['total_sum'] to float if decimals aren't critical
    # Ensure ts_info['total_sum'] is a float type
    ts_info['total_sum'] = ts_info['total_sum'].astype(float)
    total_sum = float(total_sum)

    ts_info['percentage_of_total'] = (ts_info['total_sum'] / total_sum * 100).round(2)

    # Sortieren nach total_sum, falls erforderlich
    if sort_by_total_sum:
        ts_info = ts_info.sort_values(by='total_sum', ascending=False).reset_index(drop=True)

    return ts_info

