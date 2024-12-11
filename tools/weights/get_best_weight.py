def scale_weights(weights_dict, exclude_variables=None):
    """
    Skaliert die Gewichte in 'selected_groups_with_weights' so, dass die Summe 1 beträgt,
    wobei bestimmte Variablen ausgeschlossen werden können.
    Falls keine Gewichte vorhanden sind, wird ('dataset',), 1 zurückgegeben.
    
    Args:
        weights_dict (dict): Dictionary mit 'selected_groups_with_weights' als Schlüssel.
        exclude_variables (list): Liste von Variablen, die aus den Gruppen ausgeschlossen werden sollen.
    
    Returns:
        dict: Ein Dictionary mit skalierten Gewichten.
    """
    for key, weights in weights_dict.items():
        selected_groups_with_weights = weights_dict.get("selected_groups_with_weights", [])
        
        if exclude_variables:
            # Filtere die Gruppen, um solche mit den ausgeschlossenen Variablen zu entfernen
            selected_groups_with_weights = [
                (group, weight) for group, weight in selected_groups_with_weights
                if not any(exclude in group for exclude in exclude_variables)
            ]

        if not selected_groups_with_weights:
            # Falls keine Gewichte übrig sind, standardmäßig ('dataset',), 1 zurückgeben
            weights_dict["selected_groups_with_weights"] = [(('dataset',), 1)]
        else:
            # Berechne die Summe aller verbliebenen Gewichte
            total_weight = sum(weight for group, weight in selected_groups_with_weights)
            
            if total_weight == 0:
                # Wenn die Summe 0 ist, setze das Gewicht ebenfalls auf ('dataset',), 1
                weights_dict["selected_groups_with_weights"] = [(('dataset',), 1)]
            else:
                # Skaliere die Gewichte so, dass die Summe 1 beträgt
                weights_dict["selected_groups_with_weights"] = [(group, weight / total_weight) for group, weight in selected_groups_with_weights]
    
    return weights_dict

def get_best_weight(weighted_losses, dict_weighted_forecast, include=None, exclude=None):
    """
    Returns the first key of the dictionary that matches the inclusion and exclusion criteria based on the Input_Arguments.
    
    :param weighted_losses: dict, the input dictionary with weighted losses.
    :param dict_weighted_forecast: dict, the dictionary containing forecast details.
    :param include: dict, the key-value pairs that must be present in Input_Arguments. 
                    If a value is an empty string, only the presence of the key is checked.
    :param exclude: str or list, the string(s) that must not be in the key.
    :return: the first matching key, or None if no match is found.
    """
    # Convert exclude to a list if it's not already
    if exclude and isinstance(exclude, str):
        exclude = [exclude]

    # Iterate through the keys in weighted_losses
    for key in weighted_losses.keys():
        # Check if the key contains any exclude terms (if specified)
        if exclude and any(term in key for term in exclude):
            continue
        
        # Extract the corresponding Input_Arguments from dict_weighted_forecast
        input_args = dict_weighted_forecast[key]["Input_Arguments"]
        
        # Check if all key-value pairs in include match those in input_args
        match = True

        for k, v in include.items():
            if k in input_args:
                # If the value in include is empty, only check if the key exists in input_args
                if v == "":
                    continue

                # If the value is not empty, check if the values match
                elif input_args[k] != v:
                    match = False
                    break
            else:
                match = False
                break
        #print(match)
        # If all conditions are satisfied, return the corresponding dictionary
        if match:
            weights_dict = dict_weighted_forecast[key]
            print("The chosen model is: ", key)
            # You can apply further processing to the weights_dict here, such as scaling weights
            weights_dict = scale_weights(weights_dict, exclude_variables=exclude)
            
            return weights_dict

    # If no match is found, return None
    return None
