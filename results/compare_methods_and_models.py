# Databricks notebook source
# DBTITLE 1,load path
import os
import sys

# Speichere das ursprüngliche Arbeitsverzeichnis
initial_dir = os.getcwd()  
main_dir = initial_dir

# Extrahiere nur den Namen des aktuellen Ordners
initial_folder_name = os.path.basename(initial_dir)

# Define the target directory name you want to find
target_dir = "P&L_Prediction"
relative_path = f'results/{initial_folder_name}'
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
weights_path = os.path.join(current_dir, "weights")
forecasts_path = os.path.join(current_dir, "forecasts")

# Setze das Arbeitsverzeichnis wieder auf das ursprüngliche zurück
os.chdir(initial_dir)

print("Main directory:", main_dir)
print("Current directory:", current_dir)
print("weights_path:", weights_path)
print("forecasts_path:", forecasts_path)

# COMMAND ----------

# DBTITLE 1,Load Forecasts
# Forecast Models
import pickle 
file = os.path.join(current_dir, 'Telefonica - bad_debt.pkl')
with open(file, 'rb') as f:
    final_result = pickle.load(f)   

final_result.keys()
final_result['AutoETS_global_ensemble_selection'].keys()
#final_result['AutoETS_global_ensemble_selection']["combined_results"][('dataset',)].display()

# COMMAND ----------

final_result['AutoETS_global_ensemble_selection']['matched_values']

# COMMAND ----------

# DBTITLE 1,calculate_metrics
# from tools.evaluation.calculate_metrics import calculate_metrics

# # Beispielaufruf der Funktion:
# method_list = ["base", "mean_base", "BottomUp", "MinTrace_method-wls_struct", "MinTrace_method-ols", "weighted_pred"]
# metric = ["MAE", "MSE", "MAE", "MAPE", "MASE", "MSE", "MAE", "WAPE", "RMSE", "RMSSE", "SMAPE"]  

# metrics_table = calculate_metrics(final_result, method_list, metric)
final_result["evaluation_metrics"].display()


# COMMAND ----------

# DBTITLE 1,Plot results
from tools.plots.plot_methods_comparison import plot_methods_comparison

chooseForecast = "AutoARIMA_global_ensemble_selection"
print(final_result[chooseForecast]["combined_results"].keys())
level = ('dataset',)

df = final_result[chooseForecast]["combined_results"][level]

plot_methods_comparison(df, date_column = "date", title = chooseForecast)

# COMMAND ----------

# DBTITLE 1,plot a forecast for a specific level
chooseForecast = "AutoARIMA_global_ensemble_selection"
dict_results = final_result[chooseForecast]["combined_results"]

from tools.plots.plot_timeseries_from_dict import plot_timeseries_from_dict

# pred_columns
print("Prediction columns")
df = final_result[chooseForecast]["combined_results"][('dataset',)]
method = "weighted_pred"
print(df.keys())
print("")

longest_element = max(final_result[chooseForecast]["combined_results"], key=len)
print("included segments")
print(longest_element)
includeVariables = ['dataset', 'PL_line', 'Segment', ]

# Beispielaufruf
plot_timeseries_from_dict(dic=dict_results, date_col="date", actuals_col="y", pred_col = method, variables = includeVariables, plot_individual=False, legend_below=True, top_n = 10)


