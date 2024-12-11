import numpy as np
import pandas as pd
from itertools import combinations
from hierarchicalforecast.utils import aggregate
from databricks.sdk.runtime import *

def prepare_conciliation_forecast(dataframes_dict, datasetNameGeneral, grouping_variables, 
                                  max_combination_length=None, groupby_cols=None):
    processed_dfs = []

    for df_name, Y_df in dataframes_dict.items():
        # Überprüfe und entferne die 'ts_id'-Spalte, falls vorhanden
        if 'ts_id' in Y_df.columns:
            Y_df = Y_df.drop(columns=['ts_id'])

        # Spalten umbenennen
        Y_df = Y_df.rename({'total': 'y', 'date': 'ds'}, axis=1)

        # Füge die 'Dataset'-Spalte hinzu
        Y_df.insert(0, 'dataset', datasetNameGeneral)

        # Wähle die Spalten in der richtigen Reihenfolge aus
        selected_columns = ['dataset', 'ds', 'y'] + grouping_variables
        Y_df = Y_df[selected_columns]

        # Konvertiere die 'ds'-Spalte in das Datetime-Format
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])

        # Extrahiere alle Spaltennamen außer 'ds' und 'y', wenn keine angegeben sind
        if groupby_cols is None:
            groupby_cols = Y_df.columns[~Y_df.columns.isin(['ds', 'y'])].tolist()
            first_element = [groupby_cols[0]]
            groupby_cols = groupby_cols[1:]

        # Generiere alle Kombinationen der groupby-Spalten
        groupby_combinations = []
        max_comb_len = max_combination_length if max_combination_length is not None else len(groupby_cols)
        for r in range(1, max_comb_len + 1):
            groupby_combinations.extend(combinations(groupby_cols, r))

        groupby_combinations = [first_element] + [first_element + list(comb) for comb in groupby_combinations]

        # Aggregiere die Daten (die Funktion 'aggregate' wird hier als gegeben angenommen)
        Y_df, S_df, tags = aggregate(Y_df, groupby_combinations)
        Y_df = Y_df.reset_index()

        # Trenne in Trainings- und Testdaten
        Y_test_df = Y_df.groupby('unique_id').tail(365)
        Y_train_df = Y_df.drop(Y_test_df.index)

        Y_test_df = Y_test_df.set_index('unique_id')
        Y_train_df = Y_train_df.set_index('unique_id')

        # Sammle das Ergebnis für jeden DataFrame
        processed_dfs.append({
            "df_name": df_name,
            "groupby_combinations": groupby_combinations,
            "Y_df": Y_df,
            "S_df": S_df,
            "tags": tags,
            "Y_test_df": Y_test_df,
            "Y_train_df": Y_train_df
        })

    return processed_dfs
