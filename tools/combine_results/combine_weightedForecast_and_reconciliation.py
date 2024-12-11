import pandas as pd

#Funktion zum Verschneiden von forecast_weighted und reconciliation_dct basierend auf den Schlüssel-Gruppierungsvariablen
def combine_weightedForecast_reconciliation(forecast_weighted, forecast_equal_weights, reconciliation_dct):
    """
    Verschmilzt die Datenframes aus forecast_weighted, forecast_equal_weights und reconciliation_dct basierend auf den gemeinsamen Keys.
    
    :param forecast_weighted: Dictionary mit den gewichteten Vorhersagen.
    :param forecast_equal_weights: Dictionary mit den gleichgewichteten Vorhersagen.
    :param reconciliation_dct: Dictionary mit den tatsächlichen Daten (Actuals).
    
    :return: Aktualisiertes forecast_weighted Dictionary mit verschmolzenen Daten.
    """
    for key in forecast_weighted.keys():
        if key in reconciliation_dct:
            # Forecast- und Actual-Daten extrahieren
            forecast_df = forecast_weighted[key]
            forecast_df_equal = forecast_equal_weights.get(key)  # Sicherstellen, dass ein Key existiert
            actuals_df = reconciliation_dct[key]

            # Die Gruppierungsvariablen sind der Schlüssel (key) plus die Spalte 'date'
            group_by_cols = list(key) + ['date']

            # Falls forecast_equal_weights für diesen Key existiert, wird gemerged
            if forecast_df_equal is not None:
                forecast_df = pd.merge(
                    forecast_df,
                    forecast_df_equal[['equal_weights_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )
                # Verknüpfen der DataFrames auf Basis der Gruppierungsvariablen mit einem outer join
                merged_df = pd.merge(
                    actuals_df, 
                    forecast_df[['weighted_pred', 'equal_weights_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )
            else:
                # Verknüpfen der DataFrames auf Basis der Gruppierungsvariablen mit einem outer join
                merged_df = pd.merge(
                    actuals_df, 
                    forecast_df[['weighted_pred'] + group_by_cols],
                    on=group_by_cols,
                    how='outer'
                )
            print(merged_df[['equal_weights_pred', 'MinTrace_method-wls_struct']]) 
            # Das verschmolzene DataFrame wieder in das forecast_weighted Dictionary einfügen
            forecast_weighted[key] = merged_df

    return forecast_weighted



def merge_forecast_with_reconciliation(weights_forecast_dict, reconciliation_dict):
    """
    Verschmilzt iterativ die 'forecast_weighted' Einträge aus 'weights_forecast_dict' mit den entsprechenden
    'reconciliation_dict' Einträgen basierend auf den Schlüssel-Gruppierungsvariablen.
    
    :param weights_forecast_dict: Dictionary mit verschiedenen Modellen und ihren gewichteten Vorhersagen.
    :param reconciliation_dict: Dictionary mit den tatsächlichen Daten (Actuals).
    
    :return: Das aktualisierte 'weights_forecast_dict' Dictionary mit den verschmolzenen 'combined_results'.
    """
    final_dict = weights_forecast_dict.copy()
    # Iteriere durch alle Einträge in weights_forecast_dict
    for model_key, model_data in final_dict.items():
        # Extrahiere den forecast_key, um den entsprechenden Eintrag im reconciliation_dict zu finden
        forecast_key = model_data.get('forecast_key')

        # Prüfe, ob der forecast_key im reconciliation_dict existiert
        if forecast_key and forecast_key in reconciliation_dict:
            # Extrahiere die Daten aus forecast_weighted und reconciliation_dct
            forecast_weighted = model_data.get('forecast_weighted')
            forecast_equal_weights = model_data.get('forecast_equal_weights')
            reconciliation_dct = reconciliation_dict[forecast_key]

            # Überprüfe, ob forecast_weighted tatsächlich vorhanden ist
            if forecast_weighted:
                # Verwende die Funktion combine_weightedForecast_reconciliation zum Verschneiden
                combined_results = combine_weightedForecast_reconciliation(forecast_weighted, forecast_equal_weights, reconciliation_dct)
                # Speichere die verschmolzenen Ergebnisse unter 'combined_results'
                model_data['combined_results'] = combined_results
                model_data["reconciliation_dct"] = reconciliation_dct

    return final_dict