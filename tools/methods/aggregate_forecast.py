import pandas as pd
import numpy as np

# def identify_lowest_level_dataframe(train_dict):
#     """
#     Identifies the DataFrame with the lowest aggregation level based on the number of categories.

#     Parameters:
#     train_dict (dict): Dictionary of DataFrames with time series data.

#     Returns:
#     pd.DataFrame: DataFrame with the lowest aggregation level.
#     """
#     lowest_level_df = None
#     max_categories = -1

#     for key, df in train_dict.items():
#         # Convert the key to a string if it's not already a string
#         key_str = str(key)
#         # Count the number of categories based on commas in the key
#         num_categories = key_str.count(',') + 1
#         if num_categories > max_categories:
#             max_categories = num_categories
#             lowest_level_df = df

#     return lowest_level_df


def identify_lowest_level_dataframe(train_dict):
    """
    Identifies the DataFrame with the lowest aggregation level based on the number of categories.

    Parameters:
    train_dict (dict): Dictionary of arrays with time series data, where keys are tuples representing the hierarchy.

    Returns:
    tuple: Key with the lowest aggregation level.
    """
    lowest_level_key = None
    max_categories = -1

    for key in train_dict.keys():
        # Convert the key to a string if it's not already a string
        key_str = str(key)
        # Count the number of categories based on commas in the key
        num_categories = key_str.count(',') + 1
        if num_categories > max_categories:
            max_categories = num_categories
            lowest_level_key = key

    return lowest_level_key


def aggregate_forecasts(forecast_dict, aggregate_by=None):
    """
    Aggregiert alle Zeitreihen pro DataFrame in dem Dictionary, sodass jeder DataFrame nur noch eine Zeitreihe enthält.
    Wenn ein `aggregate_by`-Argument angegeben wird, wird nicht nach `date` aggregiert, sondern nach den angegebenen Spalten.
    
    Parameters:
    forecast_dict (dict): Dictionary mit DataFrames, die aggregiert werden sollen.
    aggregate_by (list, optional): Liste von Spaltennamen, nach denen aggregiert werden soll, anstelle von 'date'.
    
    Returns:
    dict: Dictionary mit aggregierten DataFrames.
    """
    aggregated_dict = {}

    for group_key, df in forecast_dict.items():
        if aggregate_by is None:
            # Aggregation nach 'date', um alle Zeitreihen auf eine zu reduzieren
            aggregated_df = df.groupby(['date']).agg({
                'total': lambda x: np.nan if x.isna().all() else x.sum(),
                'forecast': lambda x: np.nan if x.isna().all() else x.sum()
            }).reset_index()
            # Füge ts_id mit dem Wert 0 hinzu
            aggregated_df['ts_id'] = 0
        else:
            # Überprüfen, ob alle angegebenen Spalten im DataFrame vorhanden sind
            if not all(col in df.columns for col in aggregate_by + ['date']):
                # DataFrame überspringen, wenn die angegebenen Spalten fehlen
                print(f"DataFrame für {group_key} übersprungen, da erforderliche Spalten fehlen.")
                continue
            
            group_columns = ['date'] + aggregate_by
            aggregation_columns = ['total', 'forecast']

            # Aggregation nach den angegebenen Spalten
            aggregated_df = df.groupby(group_columns).agg({
                'total': lambda x: np.nan if x.isna().all() else x.sum(),
                'forecast': lambda x: np.nan if x.isna().all() else x.sum()
            }).reset_index()
            
            # Summiere die aggregierten Werte, um die Größe der Zeitreihen zu bestimmen
            size_df = aggregated_df.groupby(aggregate_by).agg({
                'total': 'sum',
                'forecast': 'sum'
            }).reset_index()

            # Bestimme die Rangfolge basierend auf der Summe (z.B. 'total' + 'forecast')
            size_df['size'] = size_df['total'] + size_df['forecast']
            size_df = size_df.sort_values(by='size', ascending=False).reset_index(drop=True)

            # Füge ts_id basierend auf der Größe hinzu
            size_df['ts_id'] = range(1, len(size_df) + 1)

            # Mergen der ts_id in den aggregierten DataFrame
            aggregated_df = aggregated_df.merge(size_df[aggregate_by + ['ts_id']], on=aggregate_by, how='left')

        # Füge das aggregierte DataFrame zum Ergebnis-Dictionary hinzu
        aggregated_dict[group_key] = aggregated_df

    return aggregated_dict


# def aggregate_forecasts(train_dict, aggregate_cols=None, aggregation_method='mean'):
#     """
#     Aggregates forecast values in the train_dict to the specified level.

#     Parameters:
#     train_dict (dict): Dictionary of DataFrames with time series data.
#     aggregate_cols (list or tuple or None, optional): List of columns to aggregate on. If None, aggregate all to the top level.
#     aggregation_method (str): Aggregation method to use. Possible values: 'mean' (default), 'bottom-up'.

#     Returns:
#     pd.DataFrame: Aggregated DataFrame with actual 'total' values and aggregated 'forecast' values.
#     """
#     if aggregation_method not in ['mean', 'bottom-up']:
#         raise ValueError("Invalid aggregation method. Choose from 'mean' or 'bottom-up'.")

#     # Identify the DataFrame with the lowest aggregation level
#     if aggregation_method == 'bottom-up':
#         lowest_level_df = identify_lowest_level_dataframe(train_dict)
#     else:
#         lowest_level_df = None

#     # Find the lowest date with forecast values across all DataFrames in train_dict
#     lowest_forecast_date = pd.Timestamp.max
#     # Find the highest date with actual total values in the DataFrame
#     highest_actual_date = pd.Timestamp.min

#     for key, df in train_dict.items():
#         if 'forecast' in df.columns:
#             min_forecast_date = df.loc[df['forecast'].notna(), 'date'].min()
#             if pd.notna(min_forecast_date) and min_forecast_date < lowest_forecast_date:
#                 lowest_forecast_date = min_forecast_date
        
#         max_actual_date = df.loc[df['total'].notna(), 'date'].max()
#         if pd.notna(max_actual_date) and max_actual_date > highest_actual_date:
#             highest_actual_date = max_actual_date

#     # If using bottom-up method and lowest_level_df is found, aggregate only that DataFrame
#     if aggregation_method == 'bottom-up' and lowest_level_df is not None:
#         # Aggregate to the specified level (aggregate_cols)
#         if aggregate_cols is None:
#             df_agg = lowest_level_df.groupby('date').agg({'total': 'sum', 'forecast': 'sum'}).reset_index()
#         else:
#             df_agg = lowest_level_df.groupby(list(aggregate_cols) + ['date']).agg({'total': 'sum', 'forecast': 'sum'}).reset_index()
        
#         # Initialize forecast values before lowest_forecast_date to NA
#         if pd.notna(lowest_forecast_date):
#             df_agg.loc[df_agg['date'] < lowest_forecast_date, 'forecast'] = pd.NA
#         if pd.notna(highest_actual_date):
#             df_agg.loc[df_agg['date'] > highest_actual_date, 'total'] = pd.NA
        
#         # Generate ts_id based on aggregate_cols
#         if aggregate_cols is None:
#             df_agg['ts_id'] = 1     
#         else:    
#             ts_id_map = {}
#             current_ts_id = 0
#             df_grouped = df_agg.groupby(aggregate_cols).size().reset_index()
#             for i, row in df_grouped.iterrows():
#                 combination_tuple = tuple(row[col] for col in aggregate_cols)
#                 if combination_tuple not in ts_id_map:
#                     ts_id_map[combination_tuple] = current_ts_id
#                     current_ts_id += 1

#             # Add ts_id to the aggregated DataFrame
#             df_agg['ts_id'] = df_agg.apply(lambda row: ts_id_map[tuple(row[col] for col in aggregate_cols)], axis=1)

#         return df_agg

#     # Otherwise, aggregate all DataFrames to the specified level
#     elif aggregation_method == 'mean':
#         # Initialize a list to store aggregated DataFrames
#         aggregated_dfs = []

#         # Initialize a dictionary to store ts_id mappings
#         ts_id_map = {}
#         current_ts_id = 0

#         # Iterate over each DataFrame in the dictionary
#         for key, df in train_dict.items():
#             # Aggregate to the specified level (aggregate_cols)
#             if aggregate_cols is None:
#                 df_agg = df.groupby('date').agg({'total': 'sum', 'forecast': 'sum'}).reset_index()
#             else:
#                 df_agg = df.groupby(list(aggregate_cols) + ['date']).agg({'total': 'sum', 'forecast': 'sum'}).reset_index()
            
#             # Initialize forecast values before lowest_forecast_date to NA
#             if pd.notna(lowest_forecast_date):
#                 df_agg.loc[df_agg['date'] < lowest_forecast_date, 'forecast'] = pd.NA
#             if pd.notna(highest_actual_date):
#                 df_agg.loc[df_agg['date'] > highest_actual_date, 'total'] = pd.NA

#             # Generate ts_id based on aggregate_cols
#             if aggregate_cols is None:
#                 df_agg['ts_id'] = 1     
#             else: 
#                 df_grouped = df_agg.groupby(aggregate_cols).size().reset_index()
#                 for i, row in df_grouped.iterrows():
#                     combination_tuple = tuple(row[col] for col in aggregate_cols)
#                     if combination_tuple not in ts_id_map:
#                         ts_id_map[combination_tuple] = current_ts_id
#                         current_ts_id += 1

#                 # Add ts_id to the aggregated DataFrame
#                 df_agg['ts_id'] = df_agg.apply(lambda row: ts_id_map[tuple(row[col] for col in aggregate_cols)], axis=1)

#             # Append aggregated DataFrame to the list
#             aggregated_dfs.append(df_agg)

#         # Concatenate all aggregated DataFrames
#         all_aggregated_df = pd.concat(aggregated_dfs, ignore_index=True)

#         # Group by the specified columns, date, and ts_id to get the final aggregated DataFrame
#         if aggregate_cols is None:
#             final_aggregated_df = all_aggregated_df.groupby('date', as_index=False).agg({'total': 'first', 'forecast': 'mean'})
#             final_aggregated_df['ts_id'] = 1 
#         else:
#             final_aggregated_df = all_aggregated_df.groupby(list(aggregate_cols) + ['date'], as_index=False).agg({'total': 'first', 'forecast': 'mean'})

#         return final_aggregated_df



def process_forecasts(forecast_dict, aggregate_by=None, aggregation_method='top'):
    """
    Verarbeitet Forecast-Daten basierend auf der Aggregationsmethode und den optionalen Aggregationsspalten.
    
    Parameters:
    forecast_dict (dict): Dictionary mit DataFrames, die aggregiert werden sollen.
    aggregate_by (list, optional): Liste von Spaltennamen, nach denen aggregiert werden soll, anstelle von 'date'.
    aggregation_method (str, optional): Methode zur Aggregation, Optionen sind 'top', 'mean', 'bottom'.
    
    Returns:
    pd.DataFrame: Aggregierter DataFrame basierend auf der gewählten Methode.
    """
    
    # Aggregiere die Forecast-Daten basierend auf der angegebenen Methode
    aggregated_dict = aggregate_forecasts(forecast_dict, aggregate_by = aggregate_by)


    # Wandel aggregate_by eine Liste ist, auch wenn sie None ist
    if aggregate_by is None:
        aggregate_by = []

    if aggregation_method == 'top':
            if aggregate_by:
                top_level_key = tuple(aggregate_by)
                if top_level_key in forecast_dict:
                    return forecast_dict[top_level_key]
                else:
                    raise ValueError(f"Kein DataFrame mit dem Schlüssel '{top_level_key}' im Dictionary gefunden.")
            else:
                # Gebe den DataFrame mit dem Schlüssel "top_level_series" zurück
                top_level_key = 'top_level_series'
                if top_level_key in forecast_dict:
                    return forecast_dict[top_level_key]
                else:
                    raise ValueError(f"Kein DataFrame mit dem Schlüssel '{top_level_key}' im Dictionary gefunden.")
    
    elif aggregation_method == 'bottom':
        # Bestimme den DataFrame mit dem niedrigsten Aggregationslevel
        lowest_level_key = identify_lowest_level_dataframe(forecast_dict)
        if lowest_level_key in aggregated_dict:
            df = aggregated_dict[lowest_level_key]
            if aggregate_by:
                # ts_id gemäß den Gruppen der "aggregate_by" setzen
                df['ts_id'] = df.groupby(aggregate_by).ngroup()
            else:
                df['ts_id'] = 0
            return df
        else:
            raise ValueError(f"Kein DataFrame mit dem Schlüssel '{lowest_level_key}' im Dictionary gefunden.")
    
    elif aggregation_method == 'mean':
        # Berechne den Mittelwert über alle DataFrames
        all_dfs = [df for df in aggregated_dict.values()]
        
        if not all_dfs:
            raise ValueError("Kein DataFrame zur Berechnung des Mittelwerts vorhanden.")
        
        # Kombiniere alle DataFrames in einem
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Berechne den Mittelwert
        mean_df = combined_df.groupby(['date'] + aggregate_by).agg({'total': 'mean', 'forecast': 'mean'}).reset_index()
        
        # Füge die ts_id-Spalte hinzu
        if aggregate_by:
            mean_df['ts_id'] = mean_df.groupby(aggregate_by).ngroup()
        else:
            mean_df['ts_id'] = 0
        
        return mean_df
    
    else:
        raise ValueError("Ungültige Aggregationsmethode. Verfügbare Optionen: 'top', 'mean', 'bottom'.")

