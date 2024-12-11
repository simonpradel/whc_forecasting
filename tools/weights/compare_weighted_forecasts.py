import pickle
import os

def compare_weighted_forecasts(files_dict):
    """
    Verarbeitet Dateien, extrahiert Vorhersagen und berechnet den gewichteten Verlust.
    
    Args:
        files_dict (dict): Dictionary mit Dateinamen als Schlüsseln und deren Inhalten als Werten.
    
    Returns:
        tuple: Zwei sortierte Dictionaries:
               1. weighted_losses (nach Verlust sortiert)
               2. detailed_results (enthält Gruppen, Gewichte und Loss, ebenfalls sortiert)
    """
    weighted_losses = {}
    detailed_results = {}

    for file_name, file_content in files_dict.items():
        try:
            selected_groups_with_weights = file_content.get('selected_groups_with_weights', [])
            aggregated_forecast = file_content.get('aggregated_forecast', {})

            # Filtere die Keys aus aggregated_forecast basierend auf selected_groups_with_weights
            filtered_forecasts = {
                key: value for key, value in aggregated_forecast.items()
                if any(group[0] == key for group in selected_groups_with_weights)
            }

            combined_preds = 0
            combined_totals = 0
            groups_weights = []  # Speichert Gruppen und ihre Gewichte
            
            # Iteriere über die gefilterten Forecasts
            for key, df in filtered_forecasts.items():
                # Filtere den DataFrame auf Zeilen, wo die Spalte 'pred' nicht 'na' ist
                df_filtered = df[df['pred'].notna()]
                
                # Extrahiere 'pred' und 'total'
                preds = df_filtered['pred'].to_numpy()
                totals = df_filtered['total'].to_numpy()

                # Hol das Gewicht für den aktuellen Key (erstes Element ist der Key, zweites das Gewicht)
                weight = next(group[1] for group in selected_groups_with_weights if group[0] == key)
                
                # Kombiniere Vorhersagen und Totals gemäß des Gewichts
                combined_preds += preds * weight
                combined_totals += totals * weight

                # Speichere die Gruppe und das Gewicht
                groups_weights.append((key, weight))

            # Berechne den Verlust (MAE)
            loss = (abs(combined_preds - combined_totals)).mean()

            weighted_losses[file_name] = loss

            # Speichere die detaillierten Ergebnisse (Gruppen, Gewichte, Loss)
            detailed_results[file_name] = {
                "groups_weights": groups_weights,
                "loss": loss
            }

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Sortiere weighted_losses und detailed_results nach dem Verlust (Loss)
    sorted_weighted_losses = dict(sorted(weighted_losses.items(), key=lambda x: x[1]))
    sorted_detailed_results = dict(sorted(detailed_results.items(), key=lambda x: x[1]['loss']))

    return sorted_weighted_losses, sorted_detailed_results

