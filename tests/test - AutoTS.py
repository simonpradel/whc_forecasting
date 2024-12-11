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

import os
import pickle
from tools.load_data.load_data import load_data_from_catalog
from tools.transformations.prepare_data import prepare_data
from tools.transformations.aggregate_by_level import aggregate_by_levels
from tools.methods.split_data import split_data
import pandas as pd 

load_data = load_data_from_catalog("Telefonica - cos", maindir = main_dir)
data = prepare_data(load_data, cutoff_date=None, fill_missing_rows = True)
aggregated_data = aggregate_by_levels(data = data,  method='long', show_dict_infos=False)

# COMMAND ----------

from tools.models.AutoTS import *

# def train_autots_and_forecast(train_df: pd.DataFrame, test_period: int, freq: str, date_col: str = "ds", actuals_col: str = "y", id_col: str = "unique_id", set_index=True, includeModels=None, excludeModels=None,) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Trainiert ein AutoTS-Modell und erstellt Prognosen sowie fitted values für den Trainingsbereich.

#     Parameters:
#     ----------
#     train_df : pd.DataFrame
#         Trainingsdaten mit Spalten 'ds' (Datum), 'y' (Wert) und einer zusätzlichen ID-Spalte (z.B. 'unique_id').
#     test_period : int
#         Anzahl der Zukunftsperioden für die Prognose.
#     freq : str
#         Frequenz der Daten, z.B. 'D' für täglich, 'W' für wöchentlich.
#     date_col : str
#         Der Name der Spalte, die Datumsinformationen enthält. Standardmäßig 'ds'.
#     actuals_col : str
#         Der Name der Spalte, die die abhängige Variable enthält. Standardmäßig 'y'.
#     id_col : str
#         Der Name der Spalte im DataFrame, die die Zeitreihen-IDs enthält. Standardmäßig 'unique_id'.
#     set_index : bool
#         Ob die `id_col` als Index gesetzt werden soll. Standardmäßig True.

#     Returns:
#     -------
#     Tuple[pd.DataFrame, pd.DataFrame]
#         Prognosewerte und Fitted Values mit den ursprünglichen Spaltennamen.
#     """

#     # Speichern der ursprünglichen Spaltennamen
#     original_columns = { 'date': date_col, 'ts_id': id_col, 'target': actuals_col }
    
#     # Prüfen, ob id_col eine Index-Spalte ist und ggf. zurücksetzen
#     if id_col in train_df.index.names:
#         train_df = train_df.reset_index(level=id_col)

#     # Umbenennen der Spalten für AutoTS
#     train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})

#     # Konvertiere das Datumsformat
#     train_df['ds'] = pd.to_datetime(train_df['ds'])

#     list_full = model_lists["all"]
#     #model_list = []

#     if(includeModels == None):
#         if(excludeModels != None):
#             model_list = [item for item in list_full if item not in exclude_model]
#         else:
#             model_list = list_full
#     else:
#         model_list = includeModels

#     print(f"model_list: {model_list}")
    
#     # Überprüfen, ob die Liste nur ein Element enthält
#     if isinstance(model_list, list) and len(model_list) == 1:
#         ensemble = None
#     else:
#         ensemble = "simple"

#     if freq == "M":
#         freq = "ME"
  
#     # Initialisiere das AutoTS-Modell
#     model = AutoTS(
#         forecast_length=test_period,
#         frequency=freq,
#         ensemble=ensemble,
#         max_generations=5,
#         num_validations=2,
#         model_list=model_list, #"superfast",  # 'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'
#         #validation_method='backwards',
#         #prediction_interval = 0.5,
#         n_jobs='auto',
#         verbose=2
#     )

#     # Trainiere das Modell
#     model = model.fit(train_df, date_col='ds', value_col='y', id_col="unique_id")

#     # Erstellen der Prognose
#     prediction = model.predict()
        
#     #forecast = prediction.forecast #.reset_index()
#     forecast = prediction.long_form_results().reset_index()

#     # Spalten zurück in die Originalnamen umbenennen
#     Y_hat_df_AutoTS = forecast.rename(columns={'datetime': original_columns['date'], 'SeriesID': original_columns['ts_id'], 'Value': "pred"})

#     # Extrahieren der Prognosen für den zukünftigen Zeitraum
#     Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS[date_col] > train_df["ds"].max()]

#     Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS["PredictionInterval"] == "50%"]
#     Y_hat_df_AutoTS = Y_hat_df_AutoTS.drop('PredictionInterval', axis=1)

#     # Fitted values extrahieren
#     fitted_model_AutoTS = model

#     # Setze 'ts_id' als Index zurück, wenn `set_index` auf True gesetzt ist
#     if set_index:
#         Y_hat_df_AutoTS.set_index(original_columns['ts_id'], inplace=True)

#     return Y_hat_df_AutoTS, fitted_model_AutoTS

df = aggregated_data["Y_df"][aggregated_data["Y_df"]["ds"] < '2022-12-31']
#test = train_autots_and_forecast(df, test_period = 15, freq=data["freq"], date_col = "ds", id_col = "unique_id", actuals_col = "y", includeModels=["ETS"], excludeModels=None, set_index = False)
test = train_autots_and_forecast(df, test_period = 12, freq=data["freq"], date_col = "ds", id_col = "unique_id", actuals_col = "y", includeModels="superfast", excludeModels=None, set_index = False, verbosity = 4)

# COMMAND ----------

predicted_values, fitted_values = train_autots_and_forecast(
    df, future_periods, freq=freq, includeModels=includeModels,
    excludeModels=excludeModels, date_col="date", actuals_col="total", id_col="ts_id", set_index=False)
print(predicted_values.head())

# COMMAND ----------

aggregated_data["Y_df"]

# COMMAND ----------

test[0]

# COMMAND ----------

aggregated_data["Y_df"].iloc[60:80]

# COMMAND ----------

aggregated_data["Y_df"].info()

# COMMAND ----------

aggregated_data["Y_df"].merge(test[0], on=["ds", "unique_id"],  how = "outer")

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Filter nach "unique_id = 'dataset'"
df = aggregated_data["Y_df"].merge(test[0], on=["ds", "unique_id"],  how = "outer")
filtered_df = df[df['unique_id'] == 'Telefonica - cos']

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['ds'], filtered_df['y'], label='Total', marker='o')
plt.plot(filtered_df['ds'], filtered_df['pred'], label='Predicted', marker='x')

# Achsenbeschriftungen und Titel
plt.xlabel('ds')
plt.ylabel('Values')
plt.title('Total vs Predicted')
plt.legend()

# Plot anzeigen
plt.show()

# COMMAND ----------

test[0].display()

# COMMAND ----------

import contextlib
import sys
from autots import AutoTS

# Funktion zur Unterdrückung der Ausgabe
@contextlib.contextmanager
def suppress_output():
    with open('/dev/null', 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Verwenden Sie suppress_output, um den AutoTS-Output zu unterdrücken
from autots import AutoTS
from autots.datasets import load_hourly

df_wide = load_hourly(long=False)

# here we care most about traffic volume, all other series assumed to be weight of 1
weights_hourly = {'traffic_volume': 20}

model_list = [
    'ETS'
]

with suppress_output():
    model = AutoTS(
        forecast_length=49,
        frequency='infer',
        prediction_interval=0.95,
        ensemble=['simple'], # ['simple', 'horizontal-min']
        #max_generations=5,
        #num_validations=2,
        #validation_method='seasonal 168',
        model_list=model_list,
        #transformer_list='all',
        #models_to_validate=0.2,
        #drop_most_recent=1,
        n_jobs='auto',
        verbose = 0
    )

    model = model.fit(
        df_wide,
        weights=weights_hourly,
    )

prediction = model.predict()
forecasts_df = prediction.forecast
prediction.long_form_results()

# COMMAND ----------

prediction

# COMMAND ----------

forecasts_df.reset_index().display()

# COMMAND ----------

prediction.long_form_results().reset_index().display()
