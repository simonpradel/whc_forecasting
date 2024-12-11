import pandas as pd
from tools.methods.split_data import split_data

def process_dataframe(df):
    """
    Verarbeitet den DataFrame, indem er die Spalte ohne "/" als 'top-level' umbenennt
    und bei den anderen Spalten das Präfix entfernt.
    """
    # Schritt 1: Identifizieren der Spalte ohne "/"
    target_col = None
    for col in df.columns:
        if "/" not in col and col not in ["ds", "y"]:
            target_col = col
            break

    # Schritt 2: Umbenennen der Spalte in 'top-level'
    if target_col:
        df.rename(columns={target_col: 'top-level'}, inplace=True)

    # Schritt 3: Entfernen des Präfixes bei allen anderen Spalten
    def remove_prefix(col):
        return col.split('/', 1)[-1] if "/" in col else col
    
    df.columns = [remove_prefix(col) for col in df.columns]

    # Rückgabe des neuen DataFrames und des extrahierten Spaltennamens
    return df, target_col




    
def combine_forecast_and_actuals(Y_rec_df, datasetNameGeneral, opt_top_level_forecast, mean_method_df):
    """
    Führt einen Merge von Y_rec_df, Y_test_df und Y_train_df basierend auf 'ds' durch,
    kombiniert die Spalte 'y' und verbindet die Ergebnisse mit opt_top_level_forecast.
    
    Parameters:
    Y_rec_df (pd.DataFrame): DataFrame mit rekalibrierten Vorhersagen.
    datasetNameGeneral (str): Der Name des Datensatzes, um die entsprechenden Zeilen zu filtern.
    opt_top_level_forecast (pd.DataFrame): DataFrame mit Top-Level-Forecast und 'opt_method'.

    Returns:
    pd.DataFrame: Der zusammengeführte DataFrame mit einer kombinierten 'y'-Spalte und opt_method.
    """


    # Schritt 2: Filtere die DataFrames nach datasetNameGeneral
    merged_df = Y_rec_df[Y_rec_df.index == datasetNameGeneral]

    # Schritt 6: Mergen mit opt_top_level_forecast basierend auf 'date' in opt_top_level_forecast und 'ds' in merged_df
    df_opt = opt_top_level_forecast[['date', 'total', 'opt_method']]  # Nur 'date' und 'opt_method' aus opt_top_level_forecast
    df_opt.rename(columns={'date': 'ds'}, inplace=True)

    final_merged_df = pd.merge(merged_df, df_opt, left_on='ds', right_on='ds', how='outer')

    mean_method_df.rename(columns={'date': 'ds'}, inplace=True)
    final_merged_df = pd.merge(final_merged_df, mean_method_df, left_on='ds', right_on='ds', how='outer')

    final_merged_df.rename(columns={'total': 'y'}, inplace=True)
    final_merged_df['y'] = pd.to_numeric(final_merged_df['y'], errors='coerce')


    # Entferne Spalten, benenne sie um
    final_merged_df, extracted_column = process_dataframe(final_merged_df)

    return final_merged_df

# Anwendung der Funktion auf alle Labels im Dictionary

def combine_all_forecasts(Y_rec_df, opt_top_level_forecast, mean_method_df, datasetNameGeneral):
    """
    Iteriert durch alle Labels im rekonsilierten DataFrame und führt die Funktion combine_forecast_and_actuals aus.

    Parameters:
    Y_rec_df (dict): Dictionary mit rekonsilierten DataFrames.
    opt_top_level_forecast (dict): Dictionary mit opt_top_level_forecast für jedes Label.
    datasetNameGeneral (str): Allgemeiner Name des Datensatzes.

    Returns:
    dict: Dictionary mit den kombinierten DataFrames für jedes Label.
    """
    # Dictionary für die kombinierten Daten initialisieren
    combined_results = {}

    # Schleife durch jedes Label im Y_rec_df
    for label, Y_rec_df_label in Y_rec_df.items():
        # Wende die Funktion combine_forecast_and_actuals auf jeden Eintrag an
        combined_results[label] = combine_forecast_and_actuals(
            Y_rec_df_label,      # Das rekonsilierte DataFrame
            datasetNameGeneral,  # Allgemeiner Datensatzname
            opt_top_level_forecast[label],  # opt_top_level_forecast für das aktuelle Label
            mean_method_df[label] #same weights
        )

    return combined_results



