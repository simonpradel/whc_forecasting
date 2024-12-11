import pickle
from tools.methods.transform_aggregated_data import *

def load_and_transform_forecasts(file_paths, id_col='unique_id', date_col='date'):
    """
    Lädt Vorhersage-Daten aus Pickle-Dateien, transformiert diese und gibt das transformierte Dictionary zurück.
    
    :param file_paths: Dictionary mit Dateipfaden zu den Pickle-Dateien
    :param id_col: Spalte, die als ID verwendet wird (Standard: 'unique_id')
    :param date_col: Spalte, die als Datumsspalte verwendet wird (Standard: 'date')
    :return: Dictionary mit den transformierten DataFrames
    """
    # Dictionary für die geladenen Daten initialisieren
    loaded_data = {}
    top_level_forecast = {}
    mean_method_df = {}
    
    # Dateien in einer Schleife öffnen und laden
    for label, path in file_paths.items():
        with open(path, 'rb') as f:
            loaded_data[label] = pickle.load(f)["predicted_dic"]
        with open(path, 'rb') as f:
            top_level_forecast[label] = pickle.load(f)["forecast_df"]
            
    
    # Dictionary für die transformierten Daten initialisieren
    transformed_data = {}
    
    # Iteriere über alle geladenen DataFrames
    for label, df in loaded_data.items():
        # Finde die numerische Spalte, die nicht 'unique_id' ist
        numeric_cols = df[('dataset',)].select_dtypes(include='number').columns
        actuals_col = [col for col in numeric_cols if col != id_col][0]  # Nimm die erste numerische Spalte außer der ID-Spalte
        

        # Liste, um die aggregierten DataFrames zu speichern
        aggregated_dfs = []
        dict_df = df.copy()
        # Schritt 1: Iteriere über jedes DataFrame im Dictionary
        for key, df2 in dict_df.items():

            # Gruppiere nach den nicht-numerischen Spalten und 'ds', summiere 'pred'
            aggregated_df = df2.groupby(date_col)['pred'].sum().reset_index()
            
            # Füge den aggregierten DataFrame zur Liste hinzu
            aggregated_dfs.append(aggregated_df)

        # Schritt 2: Kombiniere alle aggregierten DataFrames
        combined_df = pd.concat(aggregated_dfs)

        # Schritt 3: Gruppiere den kombinierten DataFrame nach 'ds' und berechne den Durchschnitt der 'pred'-Werte
        mean_method_df[label] = combined_df.groupby(date_col)['pred'].mean().reset_index(name='mean_method')


   
        # Wende die Transformation an
        mapping, transformed_df = transform_dict_to_long(
            dataframes=df, 
            id_col=id_col, 
            date_col=date_col, 
            actuals_col=actuals_col,  # Verwende die gefundene numerische Spalte
            set_index=True
        )
        
        # Transformiere die Spalte 'date' zu 'ds'
        transformed_df.rename(columns={date_col: 'ds'}, inplace=True)
        
        # Speichere das transformierte DataFrame im neuen Dictionary
        transformed_data[label] = transformed_df
    
    # Rückgabe der transformierten DataFrames
    return top_level_forecast, transformed_data, mean_method_df
