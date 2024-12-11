from tools.transformations.transform_aggregated_data import transform_dict_to_long

def transform_multiple_dict_to_long(loaded_forecasts, id_col='unique_id', date_col='date', numeric_col=None):
    """
    Transformiert die Forecast-Daten in ein Long-Format.
    
    :param loaded_forecasts: Dictionary mit geladenen Forecast-Daten.
    :param id_col: Spalte, die als ID verwendet wird (Standard: 'unique_id').
    :param date_col: Spalte, die als Datumsspalte verwendet wird (Standard: 'date').
    :param numeric_col: Die numerische Spalte, die transformiert werden soll.
    :return: Dictionary mit den transformierten DataFrames im Long-Format.
    :raises ValueError: Falls keine numerische Spalte angegeben wurde.
    """
    if numeric_col is None:
        raise ValueError("Die numerische Spalte muss explizit angegeben werden.")

    transformed_data = {}

    # Iteriere über die geladenen Forecasts und transformiere sie
    for label, forecast_dict in loaded_forecasts.items():
        # Überprüfe, ob die angegebene numerische Spalte existiert
        if numeric_col not in forecast_dict[('dataset',)].columns:
            raise ValueError(f"Die Spalte {numeric_col} existiert nicht in den Daten für {label}.")

        # Wende die Transformation an
        mapping, transformed_df = transform_dict_to_long(
            dataframes=forecast_dict, 
            id_col=id_col, 
            date_col=date_col, 
            actuals_col=numeric_col,
            set_index=True
        )
        
        # Transformiere die Spalte 'date' zu 'ds'
        transformed_df.rename(columns={date_col: 'ds'}, inplace=True)
        transformed_df.index.names = ['unique_id']
  
        # Speichere das transformierte DataFrame im Dictionary
        transformed_data[label] = transformed_df

    return transformed_data