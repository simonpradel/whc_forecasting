import re
import pandas as pd

import re
import pandas as pd

def calculate_metric(df, column, metric):
    """Berechnet die angegebene Metrik."""
    if metric == 'MAPE':
        df['true_deviation'] = df['y'] - df[column]
        df['absolute_deviation'] = abs(df['true_deviation'])
        df['percentage_deviation'] = (df['absolute_deviation'] / df['y']) * 100
        return df['percentage_deviation'].median()
    
    elif metric == 'MAE':
        df['true_deviation'] = df['y'] - df[column]
        df['absolute_deviation'] = abs(df['true_deviation'])
        return df['absolute_deviation'].median()
    
    elif metric == 'MSE':
        df['true_deviation'] = df['y'] - df[column]
        df['squared_deviation'] = df['true_deviation'] ** 2
        return df['squared_deviation'].mean()

    else:
        raise ValueError(f"Unbekannte Metrik: {metric}")

    
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


import pandas as pd
import os

def evaluate_forecast_performance_all_methods(dfs_dict, metric='MAPE', save_path=None, file_name=None):
    """
    Bewertet die Prognoseleistung verschiedener Modelle mithilfe der angegebenen Metrik.
    
    Parameters:
    dfs_dict (dict): Ein Dictionary von DataFrames, in denen die Vorhersagen gespeichert sind.
    metric (str): Die zu berechnende Metrik. Standardmäßig 'MAPE'.
    save_path (str, optional): Pfad, um die Ergebnisse zu speichern. Wenn None, wird nichts gespeichert.
    file_name (str, optional): Name der Datei, unter dem die Ergebnisse gespeichert werden sollen. Muss mit save_path angegeben werden.
    
    Returns:
    pd.DataFrame: Ein DataFrame mit den berechneten Metriken für jedes Modell und jede Methode.
    """
    # Initialisiere ein leeres Dictionary, um die Metrik-Ergebnisse zu speichern
    metric_dic = {}

    # Iteriere über das Dictionary von DataFrames
    for model_name, df in dfs_dict.items():
        # Schritt 1: Verarbeite den DataFrame (Prefix-Entfernung, Umbenennung der Spalte ohne "/")
        processed_df, extracted_column = process_dataframe(df)

        # Schritt 2: Entferne alle Zeilen mit NA im 'top-level'
        df_clean = processed_df.dropna(subset=['top-level']).copy()

        # Initialisiere ein Dictionary, um die Metrik-Werte für dieses Modell zu speichern
        metric_dict = {}

        # Schritt 3: Iteriere über alle Spalten außer 'ds' und 'y'
        for column in df_clean.columns:
            if column not in ['ds', 'y']:
                # Berechne die Metrik für die aktuelle Spalte
                metric_dict[column] = calculate_metric(df_clean, column, metric).round(2)

        # Speichere die berechneten Metriken in metric_dic mit dem Modellnamen als Schlüssel
        metric_dic[model_name] = metric_dict

    # Schritt 4: Erstelle einen DataFrame mit den Metrik-Werten
    metric_df = pd.DataFrame(metric_dic).T  # Transponiere den DataFrame, damit jedes Modell eine Zeile ist

    # Sortiere den DataFrame nach der "opt_method" Spalte
    if 'opt_method' in metric_df.columns:
        metric_df = metric_df.sort_values(by='opt_method', ascending=True)

    # Speichern des DataFrames, falls ein Speicherpfad und Dateiname angegeben sind
    if save_path and file_name:
        full_path = os.path.join(save_path, file_name)
        try:
            metric_df.to_csv(full_path, index=True)
            print(f"Ergebnisse wurden erfolgreich gespeichert in: {full_path}")
        except Exception as e:
            print(f"Fehler beim Speichern der Datei: {e}")
    elif save_path and not file_name:
        print("Fehler: Ein Dateiname muss angegeben werden, wenn ein Speicherpfad festgelegt ist.")
    elif file_name and not save_path:
        print("Fehler: Ein Speicherpfad muss angegeben werden, wenn ein Dateiname festgelegt ist.")

    return metric_df


