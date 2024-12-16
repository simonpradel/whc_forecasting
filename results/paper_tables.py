# This notebook creates the results tables that are used in the thesis.
# To reproduce the results for the public datasets, the datasets have to be created 
# using the file `modified_method_weighted_aggregation`.

################################ EDIT HERE ###################################
# Define the root directory (e.g., name of the project)
target_dir = "whc_forecasting"
################################ EDIT HERE ###################################

# COMMAND ----------
# Load the packages and define the path to load functions and results from

# Import packages
import os
import sys
import warnings
import pickle
import pandas as pd

# Load custom functions
from tools.paper_results.count_files_in_folders import count_files_in_folders
from tools.paper_results.pivot_model_metrics import pivot_model_metrics
from tools.paper_results.rank_models_across_datasets import rank_models_across_datasets
from tools.paper_results.calc_different_metrics_across_datasets import calc_different_metrics_across_datasets

# Save the initial working directory
initial_dir = os.getcwd()  
main_dir = initial_dir

# Extract the name of the current folder
initial_folder_name = os.path.basename(initial_dir)

relative_path = f'{initial_folder_name}'
dataset_name = initial_folder_name

print("Current folder name:", initial_folder_name)

# Initialize a variable to track whether the target directory was found
found = False

# Traverse upwards in the directory hierarchy to locate the target directory
while True:
    base_name = os.path.basename(main_dir)
    if base_name == target_dir:
        found = True
        break
    parent_dir = os.path.dirname(main_dir)
    if parent_dir == main_dir:  # Reached the root directory
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
    print(f"The target directory '{target_dir}' was not found in the directory hierarchy.")

# Combine the main directory with the relative path
current_dir = os.path.join(main_dir, relative_path)

# Reset the working directory to the initial directory
os.chdir(initial_dir)

print("Main directory:", main_dir)
print("Current directory:", current_dir)

# COMMAND ----------

# DBTITLE 1,Count files


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
    "website_traffic"
]

check = "weights" # weights, forecasts


threshold = 8
file_counts, datasets_below_threshold = count_files_in_folders(datasets, current_dir, check, max_files=threshold, return_datasets_below_threshold=True)
print("File counts:", file_counts)
print("Datasets below threshold:", datasets_below_threshold)



# COMMAND ----------
# Before Creating the tables, several tests can be made. For example, 
# DBTITLE 1,Check weight files

datasets = [
    # "australian_labour_market", 
    # "global_electricity_production", 
    "italian_grocery_store", 
    # "M5",
    # "natural_gas_usage", 
    # "prison_population", 
    # "retail_prices", 
    # "store_item_demand", 
    # "superstore", 
    # "tourism",
    #"website_traffic", 
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

print(file_forecasts.keys())
print(file_forecasts["italian_grocery_store"].keys())
file_forecasts["italian_grocery_store"]["Forecast_AutoTS_global_180_D_1800.pkl"]

# COMMAND ----------

# DBTITLE 1,Check final Dataset Dictionary

print(pd.__version__)

# pick only one
datasets = [
    #"australian_labour_market", 
    #"global_electricity_production", 
    "italian_grocery_store", 
    # "M5",
    # "natural_gas_usage", 
    # "prison_population", 
    # "retail_prices", 
    # "store_item_demand", 
    # "superstore", 
    # "tourism",
    # "website_traffic"
]

for dataset in datasets:
    try:
        df_path = os.path.join(current_dir, dataset, dataset + ".pkl")
        with open(df_path, 'rb') as f:
            final_result = pd.read_pickle(f)
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(f"Error loading dataset {dataset}: {e}")

print(final_result.keys())
print(final_result)

# COMMAND ----------

# DBTITLE 1,Count Datasets


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
    "website_traffic"
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
        #print(f"Dataset: {dataset}, Length of evaluation_metrics: {len(final_result['Model'].unique())}")
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(f"Error loading dataset {dataset}: {e}")

print(dataframe_list)
df_list = dataframe_list.copy()
# COMMAND ----------



# COMMAND ----------

#####################################################################################################
# 4.5.1 Single- vs. Multi-Level Benchmarks
# Table 7
#####################################################################################################
metric = "MASE" 
column_values = "forecast_model"
row_values = "method" 
add_to_row_values = "optim_method"
forecast_method = None #level, global
forecast_model = None
method = None
time_limit = 600
dataset_type = None
aggfunc = "mean" 
add_to_col_values = "forecast_method"
show_control_table = False

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values,
                                    add_to_row_values = add_to_row_values, forecast_method = forecast_method,
                                    forecast_model = forecast_model, method = method, time_limit = time_limit, 
                                    dataset_type = dataset_type, aggfunc = aggfunc, add_to_col_values = add_to_col_values, 
                                    show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.2f}".format  # Formats floats to 2 decimal places
)

# COMMAND ----------

#####################################################################################################
# 4.5.1 Single- vs. Multi-Level Benchmarks
# Table 8
#####################################################################################################
metric = 'MASE'
forecast_method = None
optim_method = None 
drop_columns = None
forecast_model = "AutoGluon" 
remove_constant_columns = True
row_values = "method" 
add_to_row_values = "optim_method"
grouping_variable = "forecast_method" 
time_limit = 600
show_control_table = False

final_results = rank_models_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

# COMMAND ----------

#####################################################################################################
# 4.5.2 ExtendedMulti-LevelBenchmarks
# Table 9
#####################################################################################################
forecast_method = "global"
optim_method = None 
metric = 'MASE' 
drop_columns = None
forecast_model = None 
remove_constant_columns = True
row_values = "method"
add_to_row_values = "optim_method"
time_limit = 1800
grouping_variable = "forecast_model" 
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = False

final_results = rank_models_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, drop_columns = drop_columns, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, grouping_variable = grouping_variable, time_limit = time_limit, columns_to_remove = columns_to_remove, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

# COMMAND ----------

#####################################################################################################
# 4.5.3 Comprehensive Comparison
# Table 10
#####################################################################################################
forecast_method = None  
optim_method = "ensemble_selection" 
method = "weighted_pred" 
metric = 'MASE' 
model = None
drop_columns = None
forecast_model = None 
remove_constant_columns = True
row_values = "forecast_model" 
add_to_row_values = ["optim_method", "forecast_method", "time_limit"]
grouping_variable = None 
time_limit = None
show_control_table = False
sort_values = True

final_results = rank_models_across_datasets(dataframe_list, model = model, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = method, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table, sort_values = sort_values)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)


# COMMAND ----------

#####################################################################################################
# Appendix
# Table 13
#####################################################################################################
metric = "MASE" 
column_values = "method"
row_values = "dataset" 
add_to_row_values = "optim_method" 
forecast_method = "level" 
forecast_model = "AutoGluon" 
time_limit = 600
method = None
dataset_type = None
aggfunc = "mean" 
show_control_table = False

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

# COMMAND ----------

#####################################################################################################
# Appendix
# Table 14
#####################################################################################################
metric = "MASE" 
column_values = "method"
row_values = "dataset" 
add_to_row_values = "optim_method" 
forecast_method = "global" 
forecast_model = "AutoGluon" 
time_limit = 600
method = None
dataset_type = None
aggfunc = "mean" 
show_control_table = False

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

# COMMAND ----------

#####################################################################################################
# Appendix
# Table 15
#####################################################################################################
forecast_method = None 
optim_method = None 
metric = ["MASE","RMSSE","SMAPE", "WAPE"]
forecast_model = "AutoGluon" 
remove_constant_columns = True
add_to_row_values = "optim_method"
grouping_variable = "forecast_method" 
time_limit = 600
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = False

final_results = calc_different_metrics_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, columns_to_remove = columns_to_remove, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)


# COMMAND ----------

#####################################################################################################
# Appendix
# Table 16
#####################################################################################################
forecast_method = "global" 
optim_method = None 
metric = 'MASE' 
remove_constant_columns = True
add_to_row_values = "optim_method"
time_limit = 1800
show_control_table = False
column_values = "method"
row_values = "dataset" 
forecast_model = "AutoGluon" 
method = None
dataset_type = None
aggfunc = "mean"

final_results = pivot_model_metrics(df_list, metric=metric, row_values=row_values, column_values=column_values, add_to_row_values = add_to_row_values, forecast_method = forecast_method, forecast_model = forecast_model, method = method, time_limit = time_limit, dataset_type = dataset_type, aggfunc = aggfunc, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

#####################################################################################################
# Appendix
# Table 17
#####################################################################################################
forecast_method = None 
optim_method = None 
metric = ['MASE', "RMSSE", "SMAPE", "WAPE"] 
forecast_model = None 
remove_constant_columns = True
add_to_row_values = "optim_method"
grouping_variable = "forecast_model" 
time_limit = 1800
columns_to_remove = ["AutoARIMA", "AutoETS"]
show_control_table = False
final_results = calc_different_metrics_across_datasets(dataframe_list, model = None, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = None, forecast_model = forecast_model, columns_to_remove = columns_to_remove, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)

# COMMAND ----------


#####################################################################################################
# Appendix
# Table 18
#####################################################################################################
forecast_method = None 
optim_method = None
method = "base"  
metric = 'MASE' 
model = None
drop_columns = None
forecast_model = None 
remove_constant_columns = True
row_values = "forecast_model" 
add_to_row_values = ["optim_method", "forecast_method", "time_limit"]
grouping_variable = None 
time_limit = None
show_control_table = False
sort_values = True

final_results = rank_models_across_datasets(dataframe_list, model = model, optim_method = optim_method, forecast_method = forecast_method, metric=metric , method = method, forecast_model = forecast_model, drop_columns = drop_columns, grouping_variable = grouping_variable, remove_constant_columns = remove_constant_columns, row_values = row_values, add_to_row_values = add_to_row_values, time_limit = time_limit, show_control_table = show_control_table, sort_values = sort_values)

final_results.to_latex(
    index=False,  # To not include the DataFrame index as a column in the table
    caption="Comparison of ML Model Performance Metrics",  # The caption to appear above the table in the LaTeX document
    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="llll",  # The format of the columns: left-aligned with vertical lines between them
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to 4 decimal places
)
