import pandas as pd

def combine_weights_and_forecasts(chosenWeights, dict_forecast, verbosity=0, target_col = 'weighted_pred'):
    """
    Diese Funktion aggregiert Vorhersagen basierend auf einer Auswahl an Gewichtungen und
    wendet diese an, um gewichtete Vorhersagen zu erstellen.
    
    :param chosenWeights: Liste von Tupeln, die die ausgewählten Kombinationen und die entsprechenden Gewichte enthält.
    :param dict_forecast: Dictionary mit den Vorhersagen.
    :param verbosity: Gibt an, ob Debugging-Informationen gedruckt werden sollen. Druckt nur, wenn verbosity > 3.
    
    :return: Neues Dictionary mit den aggregierten und gewichteten Vorhersagen.
    """

    # Schritt 0: Extrahiere die Schlüssel aus den Tupeln in chosenWeights und finde die gemeinsamen Elemente
    selected_combinations = [set(tup[0]) for tup in chosenWeights]
    common_elements = set.intersection(*selected_combinations)  # Gemeinsame Elemente in allen Tupeln

    # Erstelle eine sortierte Liste der gemeinsamen Elemente, um sie später konsistent zu verwenden
    common_elements = sorted(common_elements)

    if verbosity > 5:
        print("Schritt 0: Gemeinsame Elemente")
        print(common_elements)
        print()

    # Schritt 1: Bereite einen neuen Dictionary vor, behalte die Keys aus dict_forecast, setze die Values auf None
    new_forecast_dict = {key: None for key in dict_forecast.keys()}

    if verbosity > 5:
        print("Schritt 1: Neuer Dictionary mit leeren Werten")
        print(new_forecast_dict)
        print()

    # Schritt 2: Aggregiere alle Dictionaries auf die gemeinsame Ebene (common_elements)
    def aggregate_predictions(df, group_by_cols):
        return df.groupby(group_by_cols)['pred'].sum().reset_index()

    # Schritt 2: Aggregiere nur die DataFrames, deren Schlüssel die gemeinsamen Elemente enthalten
    aggregated_forecasts = {}
    for key, df in dict_forecast.items():
        if set(common_elements).issubset(set(key)):
            group_by_cols = list(common_elements) + ['date']
            aggregated_df = aggregate_predictions(df, group_by_cols)
            aggregated_forecasts[key] = aggregated_df

    if verbosity > 5:
        print("Schritt 2: Aggregierte Vorhersagen")
        print(aggregated_forecasts)
        print()

    # Schritt 3: Anwenden der Gewichte auf die aggregierten Vorhersagen
    weighted_forecasts = {}
    for key, df in aggregated_forecasts.items():
        weight = dict(chosenWeights).get(key, 0)
        df[target_col] = df['pred'] * weight
        weighted_forecasts[key] = df

    if verbosity > 5:
        print("Schritt 3: Gewichtete Vorhersagen")
        print(weighted_forecasts)
        print()

    # Schritt 4: Summe der gewichteten Vorhersagen für identische Kombinationen berechnen
    combined_weighted_forecasts = pd.concat(weighted_forecasts.values(), ignore_index=True)
    summed_weighted_forecasts = combined_weighted_forecasts.groupby(common_elements + ['date'])[target_col].sum().reset_index()

    if verbosity > 5:
        print("Schritt 4: Summierte gewichtete Vorhersagen")
        print(summed_weighted_forecasts)
        print()

    # Schritt 5: Berechne für jede mögliche Teilmenge von `common_elements` die Summe der gewichteten Vorhersagen
    from itertools import chain, combinations
    
    def get_all_subsets(s):
        """Hilfsfunktion, um alle Teilmengen eines Sets zu berechnen."""
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    
    subsets = list(get_all_subsets(common_elements))
    
    if verbosity > 5:
        print("Schritt 5: Alle Teilmengen der gemeinsamen Elemente")
        print(subsets)
        print()

    for subset in subsets:
        subset = list(subset)
        grouped_forecast = summed_weighted_forecasts.groupby(subset + ['date'])[target_col].sum().reset_index()
        
        for key in new_forecast_dict.keys():
            if set(key) == set(subset):
                new_forecast_dict[key] = grouped_forecast

    if verbosity > 5:
        print("Schritt 5: Aktualisierter Dictionary mit gewichteten Vorhersagen")
        print(new_forecast_dict)
        print()

    # Schritt 6: Lösche alle leeren Einträge aus dem neuen Dictionary
    new_forecast_dict = {key: value for key, value in new_forecast_dict.items() if value is not None}

    if verbosity > 5:
        print("Schritt 6: Bereinigter Dictionary")
        print(new_forecast_dict)
        print()

    return new_forecast_dict



# Funktion zum Finden übereinstimmender Dictionaries und Hinzufügen von forecast_weighted 
def create_weights_forecast_dict(weights_result, forecast_dic, weighted_losses, keys_to_check, additional_keys_weights=None, additional_keys_forecast=None, verbosity = 0):
    matches = {}

    if additional_keys_weights is None:
        additional_keys_weights = []
    if additional_keys_forecast is None:
        additional_keys_forecast = []

    for weight_key, weight_val in weights_result.items():
        weight_args = weight_val['Input_Arguments']

        time_limit = weight_val.get('meta_data', {}).get('time_limit')
        if time_limit is not None:
            weight_args['time_limit'] = time_limit

        if verbosity >= 5:
            print("Weighted Input Arguments Keys")
            print(weight_args.keys())

        for forecast_key, forecast_val in forecast_dic.items():
            forecast_args = forecast_val['Input_Arguments']
                      
            time_limit = forecast_val.get('meta_data', {}).get('time_limit')
            if time_limit is not None:
                forecast_args['time_limit'] = time_limit

            if verbosity >= 5:
                print("forecast_args Input Arguments Keys")
                print(forecast_args.keys())

            all_match = True
            matched_values = {}
            additional_weights_values = {}
            additional_forecast_values = {}

            for key in keys_to_check:
                if weight_args.get(key) == forecast_args.get(key):
                    matched_values[key] = weight_args[key]
                else:
                    all_match = False
                    break

            if all_match:
                for add_key in additional_keys_weights:
                    if add_key in weight_args:
                        additional_weights_values[add_key] = weight_args[add_key]

                for add_key in additional_keys_forecast:
                    if add_key in forecast_args:
                        additional_forecast_values[add_key] = forecast_args[add_key]

                match_values = [weight_args.get(key) for key in keys_to_check]
                match_values += [weight_args.get(add_key) for add_key in additional_keys_weights if add_key in weight_args]
                match_values += [forecast_args.get(add_key) for add_key in additional_keys_forecast if add_key in forecast_args]
                base_match_string = "_".join(map(str, match_values))

                # Berechnung von forecast_weighted
                chosenWeights = weight_val["selected_groups_with_weights"]
                dict_forecast = forecast_val["predicted_dic"]
                if verbosity >= 5:
                    print("start combine_weights_and_forecasts with weighted pred")        
                forecast_weighted = combine_weights_and_forecasts(chosenWeights, dict_forecast, verbosity, "weighted_pred")

                # Berechnung von forecast equal weights
                equalWeights = create_weighted_tuples(dict_forecast)
                if verbosity >= 5:
                    print("start combine_weights_and_forecasts with equal weights")
                forecast_equal_weights = combine_weights_and_forecasts(equalWeights, dict_forecast, verbosity, "equal_weights_pred")

                # Prüfe, ob der base_match_string bereits existiert
                if base_match_string in matches:
                    current_weight_key = matches[base_match_string]['weights_key']
                    current_loss = weighted_losses.get(current_weight_key, float('inf'))
                    new_loss = weighted_losses.get(weight_key, float('inf'))

                    if new_loss < current_loss:
                        matches[base_match_string] = {
                            'weights_key': weight_key,
                            'forecast_key': forecast_key,
                            'weights_dict': weight_val,
                            'forecast_dict': forecast_val,
                            'forecast_weighted': forecast_weighted,  # Hinzufügen des berechneten Werts
                            'forecast_equal_weights': forecast_equal_weights,  # Hinzufügen des berechneten Werts
                            'matched_values': matched_values,
                            'additional_weights_values': additional_weights_values,
                            'additional_forecast_values': additional_forecast_values
                        }
                else:
                    matches[base_match_string] = {
                        'weights_key': weight_key,
                        'forecast_key': forecast_key,
                        'weights_dict': weight_val,
                        'forecast_dict': forecast_val,
                        'forecast_weighted': forecast_weighted,  # Hinzufügen des berechneten Werts
                        'forecast_equal_weights': forecast_equal_weights,  # Hinzufügen des berechneten Werts
                        'matched_values': matched_values,
                        'additional_weights_values': additional_weights_values,
                        'additional_forecast_values': additional_forecast_values
                    }

    return matches

def create_weighted_tuples(dict_forecast):
        # Anzahl der Keys im Dictionary berechnen
        num_keys = len(dict_forecast)
        
        # Gleichgewichteter Wert berechnen
        equal_weight = 1 / num_keys if num_keys > 0 else 0
        
        # Liste von Tupeln erstellen
        weighted_tuples = [(key, equal_weight) for key in dict_forecast.keys()]
        
        return weighted_tuples