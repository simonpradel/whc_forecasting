import pandas as pd

def calculate_weighted_forecast(Weights_dic, Forecast_dic):
    # Extrahiere die (Columns, Weights) Tupel aus Weights_dic
    selected_groups_with_weights = Weights_dic["selected_groups_with_weights"]

    # Initialisiere eine Liste, um die gewichteten DataFrames zu speichern
    weighted_dfs = []

    # Durchlaufe jedes Tupel (Columns, Weights)
    for columns, weight in selected_groups_with_weights:
        # Wandle die Liste der Spalten in ein Tupel um, um die Keys von Forecast_dic abzugleichen
        key_tuple = tuple(columns)
        
        # Überprüfen, ob der key_tuple im Forecast_dic["predicted_dic"] existiert
        if key_tuple in Forecast_dic["predicted_dic"]:
            # Extrahiere den DataFrame
            df = Forecast_dic["predicted_dic"][key_tuple]

            # Gruppiere nach "date" und summiere die "pred" Spalte
            df_grouped = df.groupby("date", as_index=False)['pred'].sum()

            # Multipliziere die summierte "pred" Spalte mit dem Gewicht
            df_grouped['pred_weighted'] = df_grouped['pred'] * weight

            # Füge den gewichteten DataFrame zur Liste hinzu
            weighted_dfs.append(df_grouped[['date', 'pred_weighted']])

    # Summiere die gewichteten "pred" Spalten über alle gewichteten DataFrames hinweg
    if weighted_dfs:
        # Initialisiere einen DataFrame mit der Summierung der gewichteten "pred" Spalten
        weighted_sum_df = weighted_dfs[0]

        # Falls es mehr als einen DataFrame gibt, iteriere und summiere die 'pred_weighted' Spalten
        for wdf in weighted_dfs[1:]:
            weighted_sum_df = weighted_sum_df.merge(wdf, on='date', how='outer').fillna(0)
            weighted_sum_df['pred_weighted'] = weighted_sum_df['pred_weighted_x'] + weighted_sum_df['pred_weighted_y']
            weighted_sum_df = weighted_sum_df[['date', 'pred_weighted']]

        # Füge die Summen-Spalte dem DataFrame mit dem Key ("dataset",) hinzu
        if ("dataset",) in Forecast_dic["predicted_dic"]:
            # Mergen des gewichteten Summe-DF mit dem entsprechenden DataFrame in Forecast_dic
            weighted_df = Forecast_dic["predicted_dic"][("dataset",)].merge(weighted_sum_df, on='date', how='left')
            
            # Entferne die alte 'pred' Spalte falls vorhanden
            if 'pred' in weighted_df.columns:
                weighted_df.drop(columns=['pred'], inplace=True)
                
            # Benenne die neue 'pred_weighted' Spalte in 'pred' um
            weighted_df.rename(columns={'pred_weighted': 'opt_method'}, inplace=True)
        else:
            print('Der Key ("dataset",) ist nicht in Forecast_dic["predicted_dic"] vorhanden.')
            return None
        weighted_df = pd.DataFrame(weighted_df)
        return weighted_df


def wrapper_calculate_weighted_forecast(weights_dict=None, forecast_dict=None):
    """
    Wrapper Funktion, die durch ein Dictionary von Dictionaries iteriert und calculate_weighted_forecast aufruft.
    
    :param weights_dict: Dictionary oder verschachteltes Dictionary mit den Gewichtungsinformationen.
    :param forecast_dict: Dictionary oder verschachteltes Dictionary mit den Forecast-Daten.
    :return: Dictionary mit gewichteten Forecasts, basierend auf den Keys der Eingabedaten.
    """
    result = {}
    Forecast_dic = {}
    # Überprüfen, ob einer der beiden Dictionaries verschachtelt ist, aber nicht beide
    if isinstance(forecast_dict, dict) and all(isinstance(v, dict) for v in forecast_dict.values()):
        # Iteriere durch das verschachtelte Forecast_dict
        for key, nested_forecast in forecast_dict.items():
            # Wende calculate_weighted_forecast auf jeden Unter-Dict an
            Forecast_dic["predicted_dic"] = nested_forecast
            forecast = calculate_weighted_forecast(weights_dict, Forecast_dic)
            # Spalte "DS" in Date-Format umwandeln
            forecast['ds'] = forecast["date"]
            forecast = forecast[['ds', 'opt_method']]
            result[key] = forecast
    
    else:
        raise ValueError("Einer der beiden Input-Dictionaries muss verschachtelt sein, aber nicht beide gleichzeitig.")
    
    return result





