import pandas as pd

def calculate_average_forecasts(loaded_forecasts):
    """
    Berechnet den Durchschnitt der Forecasts über alle Keys im Dictionary und benennt die Datumsspalte in 'ds' um.
    
    :param loaded_forecasts: Dictionary, das die geladenen Forecasts enthält. Kann entweder geschachtelt oder einfach sein.
    :return: Dictionary mit DataFrames, die den Durchschnitt für jeden Key enthalten.
    """
    mean_method_df = {}

    # Überprüfe, ob `loaded_forecasts` ein einfaches Dictionary ist oder geschachtelt
    if all(isinstance(v, dict) for v in loaded_forecasts.values()):
        # Geschachteltes Dictionary
        for label, forecast_dict in loaded_forecasts.items():
            aggregated_dfs = []

            # Iteriere über die DataFrames im inneren Dictionary
            for key, df in forecast_dict.items():
                # Benenne die Datumsspalte in 'ds' um
                df = df.rename(columns={df.columns[df.columns.str.contains('date|time', case=False)][0]: 'ds'})
                
                # Berechne die Summe der 'pred'-Werte pro Datum
                aggregated_df = df.groupby('ds')['pred'].sum().reset_index()
                aggregated_dfs.append(aggregated_df)

            # Kombiniere alle aggregierten DataFrames aus dem inneren Dictionary
            combined_df = pd.concat(aggregated_dfs)
            
            # Berechne den Durchschnitt der 'pred'-Werte pro Datum
            mean_df = combined_df.groupby('ds')['pred'].mean().reset_index(name='mean_method')

            # Speichere das Ergebnis für das aktuelle Label (erster Dict-Schlüssel)
            mean_method_df[label] = mean_df

    else:
        # Einfaches Dictionary
        for label, df in loaded_forecasts.items():
            # Benenne die Datumsspalte in 'ds' um
            df = df.rename(columns={df.columns[df.columns.str.contains('date|time', case=False)][0]: 'ds'})
            
            # Berechne die Summe der 'pred'-Werte pro Datum
            aggregated_df = df.groupby('ds')['pred'].sum().reset_index()
            
            # Berechne den Durchschnitt der 'pred'-Werte pro Datum
            mean_df = aggregated_df.groupby('ds')['pred'].mean().reset_index(name='mean_method')
            
            # Speichere das Ergebnis für das aktuelle Label
            mean_method_df[label] = mean_df

    # Rückgabe eines Dictionaries mit DataFrames, für jedes Label ein DataFrame
    return mean_method_df

