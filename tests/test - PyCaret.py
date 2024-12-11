# Databricks notebook source
# MAGIC %pip install pycaret

# COMMAND ----------

# check installed version
import pycaret 
pycaret.__version__

# COMMAND ----------

import os
import sys

# Get the current working directory
current_dir = os.getcwd()
main_dir = current_dir

# Define the target directory name you want to find
target_dir = "P&L_Prediction"

# Initialize a variable to track the found state
found = False

# Loop upwards in the directory structure until the target is found or the root is reached
while True:
    # Get the name of the last component of the current path
    base_name = os.path.basename(main_dir)
    # Check if the current directory is the target directory
    if base_name == target_dir:
        found = True
        break
    # Move one directory up
    parent_dir = os.path.dirname(main_dir)
    # If we have reached the root directory and haven't found the target
    if parent_dir == main_dir:  # We are at the root, and can't go up anymore
        break
    # Update the current directory to the parent
    main_dir = parent_dir

# Check if the directory was found and handle the outcome
if found:
    # Append the target directory to the system path
    sys.path.append(main_dir)
    # Optionally, change the working directory to the target directory
    os.chdir(main_dir)
    print("Found and set the working directory to:", os.getcwd())
    
    # Add the new working directory (P&L_Prediction) to sys.path if not already present
    if main_dir not in sys.path:
        sys.path.append(main_dir)
    
else:
    print(f"The target directory '{target_dir}' was not found in the path hierarchy.")

# Teilpfad, der angehängt werden soll
relative_path = 'data_forecast/results/Telefonica - cos/forecasts_results'

# Kombiniere das aktuelle Verzeichnis mit dem relativen Pfad
current_dir = os.path.join(main_dir, relative_path)
print("Target directory with relative path:", current_dir)

# COMMAND ----------

from tools.load_data.load_data import load_data_from_catalog

load_data = load_data_from_catalog("Telefonica - cos")

# COMMAND ----------

from tools.transformations.prepare_data import prepare_data
data = prepare_data(load_data, cutoff_date='2024-04-30', fill_missing_rows = True)

# COMMAND ----------

from tools.transformations.aggregate_by_level import aggregate_by_levels

aggregated_data = aggregate_by_levels(data = data, method='dictionary', show_dict_infos=False)

# COMMAND ----------

from tools.methods.split_data import split_data

train_dic, test_dic = split_data(aggregated_data, period=12, unique_id="ts_id", format="dictionary")


# COMMAND ----------

train_dic

# COMMAND ----------

# import TSForecastingExperiment and init the class
from pycaret.time_series import TSForecastingExperiment

# COMMAND ----------


from pycaret.time_series import *
s = setup(train_dic[('dataset',
  'PL_line',
  'Product',
  'Segment')], fh = 3, session_id = 123)

# COMMAND ----------

# import TSForecastingExperiment and init the class
from pycaret.time_series import TSForecastingExperiment
exp = TSForecastingExperiment()
# check the type of exp
type(exp)

# init setup on exp
exp.setup(data, fh = 3, session_id = 123)

# COMMAND ----------

# compare baseline models
best = compare_models()

# COMMAND ----------

# predict on test set
holdout_pred = predict_model(best)

# COMMAND ----------

compare_models(include = ['ets', 'arima', 'theta', 'naive', 'snaive', 'grand_means', 'polytrend'])
