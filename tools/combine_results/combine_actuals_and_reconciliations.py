import pandas as pd
from tools.transformations.transform_aggregated_data import transform_long_to_dict

def combine_actuals_and_reconciliations(aggregated_data, Y_rec_df):
        """
        Diese Funktion durchläuft alle Keys in Y_rec_df und führt die Join-Operation für jedes Dictionary-Paar durch.
        Danach wird die Transformation mit transform_long_to_dict durchgeführt.
        
        :param aggregated_data: Dictionary mit den aggregierten Daten
        :param Y_rec_df: Dictionary mit den zu verschneidenden DataFrames
        :return: Dictionary mit den Ergebnissen der Transformationen für jeden Key
        """
        result_dict = {}

        # Durchlaufe alle Keys in Y_rec_df
        for forecast_key, forecast_df in Y_rec_df.items():
            # Benenne die Spalte 'ds' in 'date' um
            forecast_df = forecast_df.rename(columns={'ds': 'date'})
            
            # Benenne auch die entsprechende Spalte in aggregated_data['Y_df'] um, falls notwendig
            Y_df_renamed = aggregated_data['Y_df'].rename(columns={'ds': 'date'})
            
            # Anpassung der Spaltennamen: Entferne "pred/" am Anfang und benenne "pred" in "base" um
            forecast_df.columns = [
                col.replace('pred/', '') if col.startswith('pred/') else ('base' if col == 'pred' else col)
                for col in forecast_df.columns
            ]
            
            # Führe die Join-Operation durch
            merged_df = pd.merge(
                Y_df_renamed, 
                forecast_df, 
                on=['unique_id', 'date'], 
                how='outer'
            )
            
            # Transformiere die gemergten Daten in ein Dictionary
            dict_df = transform_long_to_dict(
                df=merged_df, 
                mapping=aggregated_data['tags'], 
                id_col='unique_id', 
                date_col='date', 
                actuals_col='y'
            )
            
            # Speichere das Ergebnis für diesen Key
            result_dict[forecast_key] = dict_df

        return result_dict