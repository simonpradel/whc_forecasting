# import pandas as pd

# def calculate_BU(forecast_dict, includeVariables):
#     """
#     Filtert ein verschachteltes Dictionary basierend auf den Tupel-Keys, sodass nur Tupel enthalten sind,
#     die ausschließlich Variablen aus includeVariables oder Teilmengen davon beinhalten.
    
#     :param forecast_dict: Verschachteltes Dictionary, dessen Keys Tupel sind.
#     :param includeVariables: Liste der Variablen, die im Tupel vorhanden sein dürfen.
#     :return: Gefiltertes verschachteltes Dictionary.
#     """
#     result = {}
    
#     # Iteriere durch das äußere Dictionary
#     for outer_key, inner_dict in forecast_dict.items():
#         # Überprüfe, ob der Wert ein Dictionary ist
#         if isinstance(inner_dict, dict):
#             # Filtere das innere Dictionary
#             filtered_inner_dict = {
#                 key: value for key, value in inner_dict.items()
#                 if all(var in includeVariables for var in key)  # Alle Variablen des Tupels müssen in includeVariables sein
#                 and set(key).issubset(includeVariables)         # Das Tupel muss eine Teilmenge von includeVariables sein
#             }
#             # Wenn das gefilterte innere Dictionary nicht leer ist, füge es dem Ergebnis hinzu
#             if filtered_inner_dict:
#                 result[outer_key] = filtered_inner_dict
#         else:
#             raise ValueError("Jeder Wert in forecast_dict sollte ein Dictionary sein.")
    
#     group_by_date_and_sum_pred(result)

#     return result


# def group_by_date_and_sum_pred(forecast_dict):
#     """
#     Iteriert durch ein verschachteltes Dictionary, gruppiert für jeden Eintrag die Daten nach "date" und summiert die Spalte "pred".
    
#     :param forecast_dict: Verschachteltes Dictionary, dessen Keys Tupel sind.
#     :return: DataFrame, gruppiert nach "date" mit der Summe der "pred"-Werte für alle Dictionary-Einträge.
#     """
#     results = {}
    
#     # Iteriere durch das äußere Dictionary
#     for key, inner_dict in forecast_dict.items():
#         print(f"Verarbeite Key: {key}")
        
#         longest_key = max(inner_dict.keys(), key=lambda k: len(k))
#         df = pd.DataFrame.from_dict(inner_dict[longest_key])

#         # Setze sicher, dass 'date' eine Spalte ist, und dass 'pred' summiert werden soll
#         if 'date' not in df.columns or 'pred' not in df.columns:
#             raise ValueError("'date' oder 'pred' Spalte nicht im DataFrame gefunden.")
        
#         # Gruppiere nach 'date' und summiere die Spalte 'pred'
#         result_df = df.groupby('date')['pred'].sum().reset_index()
#         result_df["ds"] = result_df["date"]
#         result_df["custom_BottomUp"] = result_df["pred"]
#         result_df.drop(columns=['date', 'pred'], inplace=True)
        
#         results[key] = result_df
    
#     return results
