# Databricks notebook source
import os
import sys

# Speichere das ursprüngliche Arbeitsverzeichnis
initial_dir = os.getcwd()  
main_dir = initial_dir

# Extrahiere nur den Namen des aktuellen Ordners
initial_folder_name = os.path.basename(initial_dir)

# Define the target directory name you want to find
target_dir = "P&L_Prediction"
relative_path = f'{initial_folder_name}'
dataset_name = initial_folder_name

print("Aktueller Ordnername:", initial_folder_name)

# Initialize a variable to track the found state
found = False

# Loop upwards in the directory structure until the target is found or the root is reached
while True:
    base_name = os.path.basename(main_dir)
    if base_name == target_dir:
        found = True
        break
    parent_dir = os.path.dirname(main_dir)
    if parent_dir == main_dir:  # We are at the root and can't go up anymore
        break
    main_dir = parent_dir

if found:
    sys.path.append(main_dir)
    os.chdir(main_dir)
    print("Found and set the working directory to:", os.getcwd())

    sys.path = list(set(sys.path))
    if main_dir not in sys.path:
        sys.path.append(main_dir)

else:
    print(f"The target directory '{target_dir}' was not found in the path hierarchy.")

# Kombiniere das aktuelle Verzeichnis mit dem relativen Pfad
current_dir = os.path.join(main_dir, relative_path)

# Setze das Arbeitsverzeichnis wieder auf das ursprüngliche zurück
os.chdir(initial_dir)

print("Main directory:", main_dir)
print("Current directory:", current_dir)

# COMMAND ----------

# DBTITLE 1,Count files
import os
import warnings

datasets = [
    "australian_labour_market", 
    "global_electricity_production", 
    "italian_grocery_store", 
    "M5",
    "natural_gas_usage", 
    "prison_population", 
    "retail_prices", 
    "store_item_demand", 
    "superstore", 
    "tourism",
    "website_traffic", 
    "Telefonica - bad_debt", 
    "Telefonica - commercial_costs", 
    "Telefonica - cos", 
    "Telefonica - fbb_fixed_other_revenue", 
    "Telefonica - hardware_revenue", 
    "Telefonica - mobile_service_revenue", 
    "Telefonica - non_commercial_costs", 
    "Telefonica - non_recurrent_income_cost"
]

check = "weights" # weights, forecasts
def count_files_in_folders(datasets, current_dir, max_files=None, return_datasets_below_threshold=False):
    """
    Zählt die Anzahl der Dateien in jedem Dataset-Ordner und gibt optional eine Liste der Dataset-Namen zurück,
    bei denen die Anzahl an Dateien unter dem gegebenen Schwellenwert max_files liegt.

    Args:
        datasets (list): Liste der Dataset-Namen.
        base_dir (str): Basisverzeichnis, in dem die Dataset-Ordner liegen.
        max_files (int, optional): Schwellenwert für die maximale Anzahl an Dateien pro Ordner.
        return_datasets_below_threshold (bool, optional): Wenn True, gibt die Funktion eine Liste der Dataset-Namen
                                                         zurück, bei denen die Dateianzahl unter max_files liegt.

    Returns:
        dict: Ein Dictionary mit Dataset-Namen als Schlüsseln und der Anzahl an Dateien im jeweiligen Ordner.
        list (optional): Eine Liste der Dataset-Namen, bei denen die Dateianzahl unter max_files liegt.
    """
    file_counts = {}
    below_threshold_datasets = []

    for dataset in datasets:
        dataset_dir = os.path.join(current_dir, dataset, check)
        try:
            # Zähle die Anzahl der Dateien im Verzeichnis
            file_count = len([f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))])
            file_counts[dataset] = file_count
            
            # Prüfe, ob die Anzahl der Dateien unter dem Schwellenwert liegt
            if max_files is not None and file_count < max_files:
                below_threshold_datasets.append(dataset)
                
        except FileNotFoundError:
            warnings.warn(f"Verzeichnis für Dataset '{dataset}' nicht gefunden.")
    
    if return_datasets_below_threshold:
        return file_counts, below_threshold_datasets
    return file_counts

threshold = 8
file_counts, datasets_below_threshold = count_files_in_folders(datasets, current_dir, max_files=threshold, return_datasets_below_threshold=True)
print("File counts:", file_counts)
print("Datasets below threshold:", datasets_below_threshold)



# COMMAND ----------

# DBTITLE 1,Check weight files
import os
import pickle
import warnings

datasets = [
    # "australian_labour_market", 
    # "global_electricity_production", 
    # "italian_grocery_store", 
    # "M5",
    # "natural_gas_usage", 
    # "prison_population", 
    # "retail_prices", 
    # "store_item_demand", 
    # "superstore", 
    # "tourism",
    #"website_traffic", 
    # "Telefonica - bad_debt", 
    # "Telefonica - commercial_costs", 
    # "Telefonica - cos", 
    # "Telefonica - fbb_fixed_other_revenue", 
    # "Telefonica - hardware_revenue", 
    "Telefonica - mobile_service_revenue", 
    # "Telefonica - non_commercial_costs", 
    # "Telefonica - non_recurrent_income_cost"
]

file_weights = {}
below_threshold_datasets = []

for dataset in datasets:
    dataset_dir = os.path.join(current_dir, dataset, "weights")
    try:
        # Liste alle Dateien im Verzeichnis auf
        file_list = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
        
        # Lade den Inhalt jeder pickle-Datei
        file_weight = {}
        for file_name in file_list:
            file_path = os.path.join(dataset_dir, file_name)
            with open(file_path, 'rb') as file:
                file_weight[file_name] = pickle.load(file)
        
        # Speichere die geladenen Daten im Dictionary
        file_weights[dataset] = file_weight
        
    except FileNotFoundError:
        warnings.warn(f"Verzeichnis für Dataset '{dataset}' nicht gefunden.")

print(file_weights["Telefonica - mobile_service_revenue"].keys())
file_weights["Telefonica - mobile_service_revenue"]["Forecast_Weights_AutoGluon_level_selection_6_4_600.pkl"]

# COMMAND ----------

# DBTITLE 1,Check forecast files
file_forecasts = {}
below_threshold_datasets = []

for dataset in datasets:
    dataset_dir = os.path.join(current_dir, dataset, "forecasts")
    try:
        # Liste alle Dateien im Verzeichnis auf
        file_list = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
        
        # Lade den Inhalt jeder pickle-Datei
        file_weight = {}
        for file_name in file_list:
            file_path = os.path.join(dataset_dir, file_name)
            with open(file_path, 'rb') as file:
                file_weight[file_name] = pickle.load(file)
        
        # Speichere die geladenen Daten im Dictionary
        file_forecasts[dataset] = file_weight
        
    except FileNotFoundError:
        warnings.warn(f"Verzeichnis für Dataset '{dataset}' nicht gefunden.")

#print(file_forecasts["website_traffic"].keys())
#file_forecasts["website_traffic"]["Forecast_AutoTS_global_180_D_1800.pkl"]

# COMMAND ----------

# DBTITLE 1,Check final Dataset Dictionary
import os
import pickle
import warnings
import pandas as pd
print(pd.__version__)

# pick only one
datasets = [
    #"australian_labour_market", 
    #"global_electricity_production", 
    # "italian_grocery_store", 
    # "M5",
    # "natural_gas_usage", 
    # "prison_population", 
    # "retail_prices", 
    # "store_item_demand", 
    # "superstore", 
    # "tourism",
    # "website_traffic", 
    # "Telefonica - bad_debt", 
    # "Telefonica - commercial_costs", 
    # "Telefonica - cos", 
    # "Telefonica - fbb_fixed_other_revenue", 
    # "Telefonica - hardware_revenue", 
     "Telefonica - mobile_service_revenue", 
    # "Telefonica - non_commercial_costs", 
    # "Telefonica - non_recurrent_income_cost"
]

for dataset in datasets:
    try:
        df_path = os.path.join(current_dir, dataset, dataset + ".pkl")
        with open(df_path, 'rb') as f:
            final_result = pd.read_pickle(f)
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(f"Error loading dataset {dataset}: {e}")

print(final_result.keys())
print(final_result["AutoGluon_level_600_ensemble_selection"].keys())
# final_result["AutoETS_global_ensemble_selection_1"]["combined_results"][('dataset',)].display()
# final_result["AutoETS_global_ensemble_selection_1"]["reconciliation_dct"].keys()

# COMMAND ----------

# DBTITLE 1,Count Datasets
import os
import pickle
import warnings

datasets = [
    "australian_labour_market", 
    "global_electricity_production", 
    "italian_grocery_store", 
    "M5",
    "natural_gas_usage", 
    "prison_population", 
    "retail_prices", 
    "store_item_demand", 
    "superstore", 
    "tourism",
    "website_traffic", 
    "Telefonica - bad_debt", 
    "Telefonica - commercial_costs", 
    "Telefonica - cos", 
    "Telefonica - fbb_fixed_other_revenue", 
    "Telefonica - hardware_revenue", 
    "Telefonica - mobile_service_revenue", 
    "Telefonica - non_commercial_costs", 
    "Telefonica - non_recurrent_income_cost"
]



dataframe_list = []
dataframe_names = []

for dataset in datasets:
    try:
        df_path = os.path.join(current_dir, dataset, dataset + ".pkl")
        with open(df_path, 'rb') as f:
            final_result = pd.read_pickle(f)['evaluation_metrics']
        dataframe_list.append(final_result)
        dataframe_names.append(dataset)
        print(f"Dataset: {dataset}, Length of evaluation_metrics: {len(final_result['Model'].unique())}")
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(f"Error loading dataset {dataset}: {e}")


# COMMAND ----------

import pandas as pd

def pivot_model_metrics(
    df_list, metric, row_values, column_values, add_to_row_values=None,
    add_to_col_values=None, forecast_method=None, method=None, forecast_model=None,
    time_limit=None, dataset_type=None, aggfunc="mean", show_control_table = False
):
    """
    Erstellt eine Pivot-Tabelle mit den Durchschnittswerten einer gegebenen Metrik basierend auf Zeilen- und Spaltenwerten.

    Zusätzliche Funktionalität:
    - add_to_row_values: Fügt zusätzliche Zeilenkategorien hinzu.
    - add_to_col_values: Iteriert über eine Spalte und fügt zusätzliche Spaltenkategorien hinzu.

    :return: DataFrame als Pivot-Tabelle mit Durchschnittswerten der gewählten Metrik.
    """
    results = []
    
    for i, df in enumerate(df_list):
        # Zeilen duplizieren, bei denen 'forecast_model' "AutoETS" oder "AutoARIMA" ist (um auch level Zeilen zu erstellen)
        #if i == 8:
        #df.display()
        df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
        df_to_duplicate['forecast_method'] = 'level'
        df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Filter nach den angegebenen Parametern
        if forecast_method is not None:
            df = df[df['forecast_method'] == forecast_method]

        if forecast_model is not None:
            df = df[df['forecast_model'] == forecast_model]

        if dataset_type is not None:
            df = df[df['dataset_type'] == dataset_type]

        if metric is not None:
            df = df[df['metric'] == metric]

        if method is not None:
            df = df[df['method'] == method]

        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]
            
        if 'Model' in df.columns:
            df = df.drop(columns=['Model'])

        # Neue Spalte für add_to_row_values
        if add_to_row_values is not None and add_to_row_values in df.columns:
            df["methodNew"] = df['method']
            df.loc[df['method'] == 'weighted_pred', 'methodNew'] = df['method'] + '_' + df['optim_method']

        results.append(df)

    # Ergebnisse zusammenführen
    df = pd.concat(results)
    df.display()
    # Bedingung für Filterung
    condition = (
        ((df['method'] != 'weighted_pred')) &
        (df['optim_method'] != 'ensemble_selection')
    )
    all_results = df[~condition].reset_index(drop=True)
    all_results.drop("method", axis=1, inplace=True)
    all_results = all_results.rename(columns={'methodNew': 'method'})

    # Methoden ersetzen
    replacements = {
        "base": "Base Forecast",
        "BottomUp": "Bottom-up",
        "MinTrace_method-ols": "Ordinary Least Squares",
        "MinTrace_method-wls_struct": "Structural Scaling",
        "equal_weights_pred": "WHC (equal weights)",
        "weighted_pred_differential_evolution": "WHC (differential evolution)",
        "weighted_pred_ensemble_selection": "WHC (ensemble selection)",
        "weighted_pred_optimize_nnls": "WHC (nnls)"
    }
    all_results["method"] = all_results["method"].replace(replacements)

    method_order = [
        "Base Forecast", "Bottom-up", "Ordinary Least Squares", "Structural Scaling",
        "WHC (equal weights)", "WHC (differential evolution)", "WHC (ensemble selection)", "WHC (nnls)"
    ]
    all_results["method"] = pd.Categorical(
        all_results["method"], categories=method_order, ordered=True
    )
    all_results = all_results.sort_values("method").reset_index(drop=True)

    # Wenn add_to_col_values angegeben ist
    if add_to_col_values:
        all_pivots = []
        for col_value in df[add_to_col_values].unique():
            filtered_df = all_results[all_results[add_to_col_values] == col_value]
            if add_to_col_values == "forecast_method":
                col_value = col_value.replace("level", "single-level").replace("global", "multi-level")
            
            pivot_table = filtered_df.pivot_table(
                index=row_values, columns=column_values, values="value", aggfunc=aggfunc
            )

            if show_control_table:
                print("the entries in the table must be the same length as the number of datasets")
                control_table = filtered_df.pivot_table(
                    index=row_values, columns=column_values, values="value", aggfunc="count"
                )
                control_table.display()

            pivot_table.columns = [f"{col} ({col_value})" for col in pivot_table.columns]
            all_pivots.append(pivot_table)

        # Alle Pivots zusammenführen
        final_pivot = all_pivots[0]
        for additional_pivot in all_pivots[1:]:
            final_pivot = final_pivot.merge(
                additional_pivot, left_index=True, right_index=True, how="outer"
            )
    else:
        if show_control_table:
            print("the entries in the table must be the same length as the number of datasets")
            control_table = all_results.pivot_table(
                index=row_values, columns=column_values, values="value", aggfunc="count"
            )
            control_table.display()
        final_pivot = all_results.pivot_table(
            index=row_values, columns=column_values, values="value", aggfunc=aggfunc
        )

    if row_values == "method":
        # Entferne spezifische Bezeichnungen aus Spaltennamen
        final_pivot.columns = [
            col.replace(" (global)", "").replace(" (single-level)", "")
            if "AutoETS" in col or "AutoARIMA" in col else col
            for col in final_pivot.columns
        ]
        
        # Logik für das Entfernen der Spalten
        columns_to_drop = [
            col for col in final_pivot.columns
            if ("AutoETS" in col or "AutoARIMA" in col) and
              ("multi-level" in col or "single-level" in col)
        ]

        final_pivot = final_pivot.drop(columns=columns_to_drop)

    for col in final_pivot.select_dtypes(include=["float", "int"]).columns:
      # Runden und sicherstellen, dass genau 2 Nachkommastellen erhalten bleiben
      final_pivot[col] = final_pivot[col].apply(lambda x: format(x, ".4f"))

    # Definiere die gewünschte Spaltenreihenfolge
    custom_column_order = [
        col for col in final_pivot.columns if all(x not in col for x in ["AutoARIMA", "AutoETS", "single-level", "multi-level"])
    ] + [
        col for col in final_pivot.columns if "AutoARIMA" in col
    ] + [
        col for col in final_pivot.columns if "AutoETS" in col
    ] + [
        col for col in final_pivot.columns if "single-level" in col
    ] + [
        col for col in final_pivot.columns if "multi-level" in col
    ]

    # Sortiere die Spalten entsprechend der benutzerdefinierten Reihenfolge
    final_pivot = final_pivot[custom_column_order]

    return final_pivot.reset_index()

# COMMAND ----------

#####################################################################################################
# 1.1 - main: MASE, 600 - main
#####################################################################################################
df_list = dataframe_list.copy()

metric = "MASE" #MAE, MAPE, MSE, RMSE, MASE, RMSSE, SMAPE, WAPE
column_values = "forecast_model"
row_values = "method" 
add_to_row_values = "optim_method"
forecast_method = None #level, global
forecast_model = None
method = None
time_limit = 600
dataset_type = None
aggfunc = "mean" # "std"
add_to_col_values = "forecast_method"
show_control_table = True

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, add_to_col_values = add_to_col_values, show_control_table = show_control_table)
final_results.display()

latex_table = final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.2f}".format  # Formats floats to two decimal places
)

latex_table

# COMMAND ----------

import pandas as pd

def rank_models_across_datasets(
    df_list, model=None, optim_method=None, forecast_method=None, method=None, 
    metric=None, forecast_model=None, drop_columns=None, remove_constant_columns=False, 
    grouping_variable=None, time_limit=None, row_values = None, add_to_row_values=None, columns_to_remove=None, show_control_table=False, sort_values = False,
):
    results = []

    for i, df in enumerate(df_list):
        # Zeilen duplizieren für bestimmte Modelle und Forecast-Methoden
        df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
        df_to_duplicate['forecast_method'] = 'level'
        df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Filter nach den angegebenen Parametern
        if model:
            df = df[df['Model'] == model]
        if optim_method:
            df = df[df['optim_method'] == optim_method]
        if method:
            df = df[df['method'] == method]
        if metric:
            df = df[df['metric'] == metric]
        if forecast_model:
            df = df[df['forecast_model'] == forecast_model]
        if forecast_method:
            df = df[df['forecast_method'] == forecast_method]
        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]

        #df = df.drop(columns=['Model'])

        df['forecast_method'] = df['forecast_method'].replace({
            'global': 'multi-level',
            'level': 'single-level'
        })

        condition = (
            ((df['method'] != 'weighted_pred')) & 
            (df['optim_method'] != 'ensemble_selection')
        )
        df = df[~condition].reset_index(drop=True)
       

        # Gruppierung nach forecast_method
        grouped_results = []
        if grouping_variable is None:
            df['rank'] = df['value'].rank(ascending=True, method='min')
            rank_1_count = (df['rank'] == 1).sum()
            df['wins'] = df['rank'].apply(lambda x: 1.0 / rank_1_count if x == 1 else 0)
            
            # Mehrere Spalten zu Rows hinzufügen
            if add_to_row_values:
                if not isinstance(add_to_row_values, list):
                    add_to_row_values = [add_to_row_values]
                df["methodNew"] = df[row_values]
                for col in add_to_row_values:
                    if col in df.columns:
                        df["methodNew"] += "," + df[col].astype(str)
            
            grouped_results.append(df)
        else:
            for f_method, group in df.groupby(grouping_variable):
                group = group.copy()
                group['rank'] = group['value'].rank(ascending=True, method='min')
                rank_1_count = (group['rank'] == 1).sum()
                group['wins'] = group['rank'].apply(lambda x: 1.0 / rank_1_count if x == 1 else 0)
                
                if add_to_row_values:
                    if not isinstance(add_to_row_values, list):
                        add_to_row_values = [add_to_row_values]
                    group["methodNew"] = group[row_values]
                    for col in add_to_row_values:
                        if col in group.columns:
                            group["methodNew"] += "," + group[col].astype(str)
                
                grouped_results.append(group)
        
        results.append(pd.concat(grouped_results, ignore_index=True))

    all_results = pd.concat(results, ignore_index=True)
    
    if add_to_row_values:
        all_results.drop(row_values, axis=1, inplace=True)
    all_results = all_results.rename(columns={'methodNew': row_values})


    all_results[row_values] = all_results[row_values].replace({
        "base": "Base Forecast",
        "BottomUp": "Bottom-up",
        "MinTrace_method-ols": "Ordinary Least Squares",
        "MinTrace_method-wls_struct": "Structural Scaling",
        "equal_weights_pred": "WHC (equal weights)",
        "base,ensemble_selection": "Base Forecast",
        "BottomUp,ensemble_selection": "Bottom-up",
        "MinTrace_method-ols,ensemble_selection": "Ordinary Least Squares",
        "MinTrace_method-wls_struct,ensemble_selection": "Structural Scaling",
        "equal_weights_pred,ensemble_selection": "WHC (equal weights)",
        "weighted_pred,differential_evolution": "WHC (differential evolution)",
        "weighted_pred,ensemble_selection": "WHC (ensemble selection)",
        "weighted_pred,optimize_nnls": "WHC (nnls)"
    })

    method_order = [
        "Base Forecast",
        "Bottom-up",
        "Ordinary Least Squares",
        "Structural Scaling",
        "WHC (equal weights)",
        "WHC (differential evolution)",
        "WHC (ensemble selection)",
        "WHC (nnls)"
    ]

    all_results["method"] = pd.Categorical(
    all_results["method"], categories=method_order, ordered=True
    )
    all_results = all_results.sort_values("method").reset_index(drop=True)


    existing_methods = [method for method in all_results[row_values].unique()]
    print(existing_methods)
    all_results[row_values] = pd.Categorical(
        all_results[row_values], 
        categories=existing_methods, 
        ordered=True
    )
   
    if grouping_variable is not None:
        grouping_variable_values = all_results[grouping_variable].unique()
        pivoted_dfs = []

        for f_method in grouping_variable_values:
            print(f_method)
            temp_df = all_results[all_results[grouping_variable] == f_method].copy()
            
            if show_control_table:
                print("the entries in the table must be the same length as the number of datasets")
                control_table = temp_df.groupby(row_values).agg(
                    avg_value=('value', 'count')
                ).reset_index()
                control_table.display()

            temp_df = temp_df.groupby(row_values).agg(
                avg_value=('value', 'mean'),
                std_value=('value', 'std'),
                avg_rank=('rank', 'mean'),
                total_wins=('wins', 'sum')
            ).reset_index()
            
            temp_df.columns = [f"{col}_{f_method}" if col != 'method' else col for col in temp_df.columns]
            pivoted_dfs.append(temp_df)

        final_df = pd.concat(pivoted_dfs, axis=1).reset_index(drop=True)
    else:
        group_cols = [row_values]
        final_df = (
            all_results.groupby(group_cols)
            .agg(
                avg_value=('value', 'mean'),
                std_value=('value', 'std'),
                avg_rank=('rank', 'mean'),
                total_wins=('wins', 'sum'),
                avg_elapsed_time=('elapsed_time', 'mean')
            )
            .reset_index()
        )

        if show_control_table:
            print("the entries in the table must be the same length as the number of datasets")
            control_table = all_results.groupby(group_cols).agg(
                avg_value=('value', 'count')
            ).reset_index()
            control_table.display()


    # elapsed_time formatieren
    if 'avg_elapsed_time' in final_df.columns:
      final_df['avg_elapsed_time'] = final_df['avg_elapsed_time'].round().astype(int)
      final_df['avg_elapsed_time'] = final_df['avg_elapsed_time'].apply(
          lambda x: f"{x // 3600:02d} h {x % 3600 // 60:02d} m"
      )

    for col in final_df.select_dtypes(include=["float"]).columns:
        final_df[col] = final_df[col].apply(lambda x: format(x, ".2f"))

    if columns_to_remove is not None:
        columns_to_drop = [col for col in final_df.columns if any(keyword in col for keyword in columns_to_remove)]
        final_df = final_df.drop(columns=columns_to_drop)

    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    if sort_values == True and grouping_variable is None:
      final_df = final_df.sort_values("avg_value").reset_index(drop=True)

    # Definiere die gewünschte Spaltenreihenfolge
    custom_column_order = [
        col for col in final_df.columns if all(x not in col for x in ["AutoARIMA", "AutoETS", "single-level", "multi-level"])
    ] + [
        col for col in final_df.columns if "AutoARIMA" in col
    ] + [
        col for col in final_df.columns if "AutoETS" in col
    ] + [
        col for col in final_df.columns if "single-level" in col
    ] + [
        col for col in final_df.columns if "multi-level" in col
    ]

    # Sortiere die Spalten entsprechend der benutzerdefinierten Reihenfolge
    final_df = final_df[custom_column_order]

    return final_df


# COMMAND ----------

#####################################################################################################
# 1.2 - main: MASE, 600 - main
#####################################################################################################

forecast_method = None #"global" # "global" #level,global
optim_method = None #"ensemble_selection" #diferential evolution,ensemble_selection
metric = 'MASE' # MASE,MSE,RMSE,MAPE,RMSSE,SMAPE
#drop_columns = ['metric', "elapsed_time"]
drop_columns = None
forecast_model = "AutoGluon" # "AutoGluon"
remove_constant_columns = True
row_values = "method" #"method"
add_to_row_values = "optim_method"
grouping_variable = "forecast_method" #"forecast_model" # "forecast_model" #"forecast_method"
time_limit = 600
columns_to_remove = None
show_control_table = True

final_results = rank_models_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table)
final_results.display()

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
# 1.3 - appendix: Datasets single-LEVEL
#####################################################################################################
# dataset, multi-level, MASE, 600 
df_list = dataframe_list.copy()

metric = "MASE" #MAE, MAPE, MSE, RMSE, MASE, RMSSE, SMAPE, WAPE
column_values = "method"
row_values = "dataset" 
add_to_row_values = "optim_method" # "optim_method"
forecast_method = "level" #level, global
forecast_model = "AutoGluon" #AutoETS, AutoGluon
time_limit = 600
method = None
dataset_type = None
aggfunc = "mean" # "std"
show_control_table = True

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)
final_results.display()



final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
# 1.3.2 - appendix: DATASET global/multi-level
#####################################################################################################
# dataset, multi-level, MASE, 600 
df_list = dataframe_list.copy()

metric = "MASE" #MAE, MAPE, MSE, RMSE, MASE, RMSSE, SMAPE, WAPE
column_values = "method"
row_values = "dataset" 
add_to_row_values = "optim_method" # "optim_method"
forecast_method = "global" #level, global
forecast_model = "AutoGluon" #AutoETS, AutoGluon
time_limit = 600
method = None
dataset_type = None
aggfunc = "mean" # "std"
show_control_table = True

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)
final_results.display()



final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

import pandas as pd

def rank_models_across_datasets2(
    df_list,
    model=None,
    optim_method=None,
    method=None,
    metric=None,
    forecast_model=None,
    forecast_method = None,
    columns_to_remove=None,
    remove_constant_columns=False,
    grouping_variable="forecast_method",
    time_limit=None,
    add_to_row_values=None,
    show_control_table = False
):
    """
    Aggregiert die Ränge und Metriken von Modellen aus mehreren Datensätzen basierend auf einer allgemeinen Gruppierungsvariable.
    """
    results = []

    for df in df_list:
        # Duplizieren bestimmter Zeilen (bei Bedarf)
        if "forecast_model" in df.columns:
            df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
            df_to_duplicate["forecast_method"] = 'level'
            df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Filterung nach Kriterien
        if model:
            df = df[df['Model'] == model]
        if optim_method:
            df = df[df['optim_method'] == optim_method]
        if method:
            df = df[df['method'] == method]
        if forecast_model:
            df = df[df['forecast_model'] == forecast_model]
        if forecast_method:
            df = df[df['forecast_method'] == forecast_method]
        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]
            
            

        df = df.drop(columns=['Model'], errors='ignore')

        # Anpassung von Zeilenwerten
        if add_to_row_values and add_to_row_values in df.columns:
            df["methodNew"] = df['method']
            df.loc[df['method'] == 'weighted_pred', 'methodNew'] = df['method'] + '_' + df['optim_method']

        # Ergebnisse sammeln
        for m in metric:
            metric_df = df[df['metric'] == m].copy()
            results.append(metric_df)

    # Zusammenführen und Gruppieren der Ergebnisse
    all_results = pd.concat(results, ignore_index=True)

    # Entferne irrelevante Kombinationen
    condition = (
        ((all_results['method'] != 'weighted_pred')) &
        (all_results['optim_method'] != 'ensemble_selection')
    )
    all_results = all_results[~condition].reset_index(drop=True)

    # Spaltenbereinigung und Umbenennung
    if "method" in all_results.columns and "methodNew" in all_results.columns:
        all_results.drop("method", axis=1, inplace=True)
        all_results.rename(columns={'methodNew': 'method'}, inplace=True)

    # Gruppierung basierend auf der definierten `grouping_variable`
    grouping_values = all_results[grouping_variable].unique()
  
    all_results["method"] = all_results["method"].replace({"base": "Base Forecast"})
    all_results["method"] = all_results["method"].replace({"BottomUp": "Bottom-up"})
    all_results["method"] = all_results["method"].replace({"MinTrace_method-ols": "Ordinary Least Squares"}) 
    all_results["method"] = all_results["method"].replace({"MinTrace_method-wls_struct": "Structural Scaling"})
    all_results["method"] = all_results["method"].replace({"equal_weights_pred": "WHC (equal weights)"})
    all_results["method"] = all_results["method"].replace({"weighted_pred_differential_evolution": "WHC (differential evolution)"})
    all_results["method"] = all_results["method"].replace({"weighted_pred_ensemble_selection": "WHC (ensemble selection)"})
    all_results["method"] = all_results["method"].replace({"weighted_pred_optimize_nnls": "WHC (nnls)"})


    method_order = [
        "Base Forecast",
        "Bottom-up",
        "Ordinary Least Squares",
        "Structural Scaling",
        "WHC (equal weights)",
        "WHC (differential evolution)",
        "WHC (ensemble selection)",
        "WHC (nnls)"
    ]

    all_results["method"] = pd.Categorical(
    all_results["method"], categories=method_order, ordered=True
    )
    all_results = all_results.sort_values("method").reset_index(drop=True)



    forecast_methods = all_results["forecast_method"].unique()
    pivoted_dfs = []

    for group_value in grouping_values:
        temp_df = all_results[all_results[grouping_variable] == group_value].copy()

        # Aggregation basierend auf Metrik
        group_cols = [
            col for col in temp_df.columns
            if col not in ['value', 'elapsed_time', "dataset_type", "time_limit", 'dataset', add_to_row_values]
        ]


        if show_control_table:
          print("the entries in the table must be the same length as the number of datasets")
          control_table = temp_df.groupby(group_cols).agg(
              avg_value=('value', 'count')
          ).reset_index()
          control_table.display()

        aggregated = temp_df.groupby(group_cols).agg(
            avg_value=('value', 'mean'),
        ).reset_index()

        # Pivot-Tabelle für Metriken
        aggregated = aggregated.pivot(index="method", columns='metric', values='avg_value').reset_index()

        # Spalten formatieren
        aggregated.columns = [
            f"{col}_{group_value}" if col not in ['method'] else col
            for col in aggregated.columns
        ]
        pivoted_dfs.append(aggregated)

    # Endgültige Tabelle zusammenführen
    final_df = pd.concat(pivoted_dfs, axis=1)

    # Entferne doppelte Spalten
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    # Entferne konstante Spalten (optional)
    if remove_constant_columns:
        const_columns = [col for col in final_df.columns if final_df[col].nunique() == 1]
        final_df.drop(columns=const_columns, inplace=True)

    # Unnötige Spalten entfernen (optional)
        # Spalten filtern, die einen der Begriffe in 'keywords_to_remove' enthalten
    if columns_to_remove is not None:
      columns_to_drop = [col for col in final_df.columns if any(keyword in col for keyword in columns_to_remove)]
      final_df = final_df.drop(columns=columns_to_drop)
      print(columns_to_remove)

    # Numerische Werte formatieren
    for col in final_df.select_dtypes(include=["float"]).columns:
        final_df[col] = final_df[col].apply(lambda x: f"{x:.4f}")

    # Spaltennamen anpassen
    final_df.columns = final_df.columns.str.replace("global", "multi-levl")
    final_df.columns = final_df.columns.str.replace("level", "single-level")
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    # elapsed_time formatieren
    if 'elapsed_time' in final_df.columns:
      final_df['elapsed_time'] = final_df['elapsed_time'].round().astype(int)
      final_df['elapsed_time'] = final_df['elapsed_time'].apply(
          lambda x: f"{x // 3600:02d} h {x % 3600 // 60:02d} m"
      )

    # Definiere die gewünschte Spaltenreihenfolge
    custom_column_order = [
        col for col in final_df.columns if all(x not in col for x in ["AutoARIMA", "AutoETS", "single-level", "multi-levl"])
    ] + [
        col for col in final_df.columns if "AutoARIMA" in col
    ] + [
        col for col in final_df.columns if "AutoETS" in col
    ] + [
        col for col in final_df.columns if "single-level" in col
    ] + [
        col for col in final_df.columns if "multi-levl" in col
    ]

    # Sortiere die Spalten entsprechend der benutzerdefinierten Reihenfolge
    final_df = final_df[custom_column_order]

    return final_df
  


# COMMAND ----------

#####################################################################################################
# 1.5 - appendix: MULTIPLE METRICS, 600 
#####################################################################################################
# Experiment 7 different metrics - main
forecast_method = None # "global" #"level"# None #"level" # "level" #level,global
optim_method = None #"ensemble_selection" #diferential evolution,ensemble_selection
metric = ["MASE","RMSSE","SMAPE", "WAPE"]
#drop_columns = ['metric', "elapsed_time"]
forecast_model = "AutoGluon" # AutoGluon, None
remove_constant_columns = True
add_to_row_values = "optim_method"
grouping_variable = "forecast_method" # "forecast_method" #"forecast_model" # "forecast_model" #"forecast_method"
time_limit = 600
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = True

final_results = rank_models_across_datasets2(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, columns_to_remove = columns_to_remove, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table)
final_results.display()

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
#Part 2.1: WHC in multi-Level Settings 1800, MASE, main
#####################################################################################################
forecast_method = "global" #"global" # "global" #level,global
optim_method = None #"ensemble_selection" #diferential evolution,ensemble_selection
metric = 'MASE' #'MASE' # MASE,MSE,RMSE,MAPE,RMSSE,SMAPE
#drop_columns = ['metric', "elapsed_time"]
forecast_model = None # "AutoGluon" # "AutoGluon"
remove_constant_columns = True
row_values = "method"
add_to_row_values = "optim_method"
time_limit = 1800
grouping_variable = "forecast_model" #"forecast_model" # "forecast_model" #"forecast_method"
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = True

final_results = rank_models_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, drop_columns = drop_columns, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, grouping_variable = grouping_variable, time_limit = time_limit, columns_to_remove = columns_to_remove, show_control_table = show_control_table)
final_results.display()

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
#  2.2:- appendix: global
#####################################################################################################
# dataset, multi-level, MASE, 1800 
df_list = dataframe_list.copy()
forecast_method = "global" #"global" # "global" #level,global
optim_method = None #"ensemble_selection" #diferential evolution,ensemble_selection
metric = 'MASE' #'MASE' # MASE,MSE,RMSE,MAPE,RMSSE,SMAPE
#drop_columns = ['metric', "elapsed_time"]
remove_constant_columns = True
add_to_row_values = "optim_method"
time_limit = 1800
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = True
column_values = "method"
row_values = "dataset" 
forecast_model = "AutoGluon" #AutoETS, AutoGluon
method = None
dataset_type = None
aggfunc = "mean" # "std"

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)
final_results.display()



final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
#Part 3.1: WHC in Single-Level Settings 1800, MASE, main
#####################################################################################################
forecast_method = None  #"global" # "global" #level,global
optim_method = "ensemble_selection" #"ensemble_selection" #diferential evolution,ensemble_selection
method = "weighted_pred" # "MinTrace_method-ols", 
metric = 'MASE' # MASE,MSE,RMSE,MAPE,RMSSE,SMAPE
model = None
#drop_columns = ['metric', "elapsed_time"]
forecast_model = None # "AutoGluon"
remove_constant_columns = True
row_values = "forecast_model" #"method"
add_to_row_values = ["optim_method", "forecast_method", "time_limit"]
grouping_variable = None # "forecast_method" #"forecast_model" # "forecast_model" #"forecast_method"
#grouping_variable = "forecast_method" 
time_limit = None
columns_to_remove = None
show_control_table = True
sort_values = True

final_results = rank_models_across_datasets(dataframe_list, model = model, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = method, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table, sort_values = sort_values)
final_results.display()

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

# COMMAND ----------

#####################################################################################################
#Part 3.2: WHC in Single-Level Settings 1800, MASE, main
#####################################################################################################
forecast_method = None #"global" # "global" #level,global
optim_method = None #"ensemble_selection" #diferential evolution,ensemble_selection
method = "base" # "MinTrace_method-ols", 
metric = 'MASE' # MASE,MSE,RMSE,MAPE,RMSSE,SMAPE
model = None
#drop_columns = ['metric', "elapsed_time"]
forecast_model = None # "AutoGluon"
remove_constant_columns = True
row_values = "forecast_model" #"method"
add_to_row_values = ["optim_method", "forecast_method", "time_limit"]
grouping_variable = None # "forecast_method" #"forecast_model" # "forecast_model" #"forecast_method"
#grouping_variable = "forecast_method" 
time_limit = None
columns_to_remove = None
show_control_table = True
sort_values = True

final_results = rank_models_across_datasets(dataframe_list, model = model, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = method, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table, sort_values = sort_values)
final_results.display()

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)
