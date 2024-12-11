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

# MAGIC %pip show cffi

# COMMAND ----------

from autogluon.core import __version__
print(__version__)

# COMMAND ----------

# DBTITLE 1,prepare
import os
import pickle
from tools.load_data.load_data import load_data_from_catalog
from tools.transform_data.prepare_data import prepare_data
from tools.transform_data.aggregate_by_level import aggregate_by_levels
from tools.methods.split_data import split_data
import pandas as pd 

load_data = load_data_from_catalog("M5", maindir = main_dir)
data = prepare_data(load_data, cutoff_date=None, fill_missing_rows = True)
aggregated_data = aggregate_by_levels(data = data,  method='long', show_dict_infos=False)


# COMMAND ----------

# Assuming df is your pandas DataFrame
print(aggregated_data['Y_df'].dtypes)
aggregated_data['Y_df'].head()


# COMMAND ----------

from tools.models.AutoGluon import *
test = train_autogluon_and_forecast(train_dic[('dataset','Product')], future_periods = 12, freq=data["freq"], date_col = "date", id_col = "dataset", actuals_col = "total", models=["AutoETS"], excludeModels=None, set_index = False)

# COMMAND ----------

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

train_data_AutoGluon = aggregated_data['Y_df']
train_data_AutoGluon = train_data_AutoGluon.rename(columns={"date": 'timestamp', "total": 'target', "ts_id": 'item_id'})

train_data_AutoGluon = aggregated_data['Y_df']
train_data_AutoGluon = train_data_AutoGluon.rename(columns={"ds": 'timestamp', "y": 'target', "unique_id": 'item_id'})
train_data_AutoGluon['item_id'] = train_data_AutoGluon['item_id'].astype(str)
train_data_AutoGluon['timestamp'] = pd.to_datetime(train_data_AutoGluon['timestamp'])


print(train_data_AutoGluon.dtypes)
models=["AutoETS"]

model_dict = {model: {} for model in models}

df = TimeSeriesDataFrame.from_data_frame(
    train_data_AutoGluon,
    id_column="item_id",
    timestamp_column="timestamp"
)
print(df.head())

# Initialisieren des Predictors
predictor = TimeSeriesPredictor(
    prediction_length=12,
    target="target",
    eval_metric="MAE",
)

print(f"model_list: {models}")

# include models
predictor.fit(
    df, 
    presets="fast_training",
    #hyperparameters=model_dict,
)

#print("MODEL FIT DONE; NOW PREDICTION")
# Erstellen der Prognose
#forecast = predictor.predict(df)


# COMMAND ----------

# DBTITLE 1,Split data
from data_forecast.methods.split_data import split_data

train_dic, test_dic = split_data(aggregated_data, period = 365, format = "dictionary")

# COMMAND ----------

# DBTITLE 1,Cluster TS per combination group
# Optional: preparation step to reduce the number of ts per combination group
# aggregated_df = cluster_dataframes(aggregated_df)
# aggregated_df


# COMMAND ----------

model_list = ['AverageValueNaive',
 'GLS',
 'GLM',
 'ETS',
 'ARIMA',
 'FBProphet',
 'RollingRegression',
 'GluonTS',
 'SeasonalNaive',
 'UnobservedComponents',
 'VECM',
 'DynamicFactor',
 'MotifSimulation',
 'WindowRegression',
 'VAR',
 'DatepartRegression',
 'UnivariateRegression',
 'UnivariateMotif',
 'MultivariateMotif',
 'NVAR',
 'MultivariateRegression',
 'SectionalMotif',
 'Theta',
 'ARDL',
 'NeuralProphet',
 'DynamicFactorMQ',
 'PytorchForecasting',
 'ARCH',
 'RRVAR',
 'MAR',
 'TMF',
 'LATC',
 'KalmanStateSpace',
 'MetricMotif',
 'Cassandra',
 'SeasonalityMotif',
 'MLEnsemble',
 'PreprocessingRegression',
 'FFT',
 'BallTreeMultivariateMotif',
 'TiDE',
 'NeuralForecast',
 'DMD']

includeModels = ['AverageValueNaive',
'GLS',
'GLM',
'ETS',
'ARIMA',
'FBProphet',
'RollingRegression',
'GluonTS',
'SeasonalNaive',
'UnobservedComponents',
'VECM',
'DynamicFactor',
'MotifSimulation',
'WindowRegression',
'VAR',
'DatepartRegression',
'UnivariateRegression']

# COMMAND ----------

# DBTITLE 1,Calculate Weights
from data_forecast.methods.calculate_weights import calculate_weights

model = "AutoTS" # "AutoGluon" "AutoTS"
forecast_method = "global" #qualification
optim_method = "ensemble_selection" #differential_evolution
includeModels = includeModels # ["LastValueNaive", "ETS"] # 'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'. None
excludeModels = ["UnivariateRegression"]

# model = "AutoGluon" # "AutoGluon" "AutoTS"
# forecast_method = "global" #qualification
# optim_method = "ensemble_selection" #differential_evolution
# includeModels = ["Naive"]
# excludeModels = None

training_results = calculate_weights(train_dic, freq = data["freq"], n_splits = 5, test_size_percent = 0.70, forecast_method = forecast_method, model = model, optim_method = optim_method, hillclimbsets = 5, max_iterations = 200, saveResults = True, excludeModels = excludeModels, includeModels = includeModels)

# COMMAND ----------

# DBTITLE 1,load weights
import pickle
file_name = 'Weights_AutoTS_global_ETS_ensemble_selection_5_200.pkl'
with open(file_name, 'rb') as f:
    training_results = pickle.load(f)

training_results

# COMMAND ----------

# DBTITLE 1,simple forecast
from data_forecast.methods.create_weighted_forecast import create_weighted_forecast

model = "AutoGluon"
excludeModels = None #["AutoETS", "RecursiveTabular","DirectTabular","Chronos"]
includeModels = ["Naive"]

# model = "AutoTS"
# excludeModels = ["AutoETS", "RecursiveTabular","DirectTabular","Chronos"]
# includeModels = ["ETS"]

# AutoETS crasht gerne mal
top_level_forecast_df, predicted_dic = create_weighted_forecast(train_dic, test_dic, training_results, future_periods = 370, freq = data["freq"], model = model, includeModels = includeModels, excludeModels = excludeModels, saveResults = True, selection_mode="global_all")

top_level_forecast_df.display()

# COMMAND ----------

# DBTITLE 1,load simple forecast
import pickle

file_name = 'Forecast_AutoGluon_global_all_370_D.pkl'

with open(file_name, 'rb') as f:
    loaded_data = pickle.load(f)

loaded_data


# COMMAND ----------

# DBTITLE 1,plot simple forecast
from data_forecast.plots.plot_simple_forecast import plot_simple_forecast

plot_simple_forecast(top_level_forecast_df)

# COMMAND ----------

# DBTITLE 1,evaluate simple forecast
from data_forecast.evaluation.evaluation import *

# Beispielaufruf
evaluate_forecast_performance(top_level_forecast_df, date_col="date", actual_col="total", modelName=file_name,  exclude_cols="ts_id", saveResults = False)


# COMMAND ----------

# DBTITLE 1,load weights
import pickle
file_name = 'Weights_AutoTS_global_ETS_ensemble_selection_5_200.pkl'
with open(file_name, 'rb') as f:
    training_results = pickle.load(f)

training_results

# COMMAND ----------

# DBTITLE 1,calculate rolling weighted forecast
from data_forecast.methods.create_weighted_folds_forecast import rolling_weighted_forecast

model = "Naive"
excludeModels =  ["AutoETS", "RecursiveTabular","DirectTabular","Chronos"]
includeModels = ["TemporalFusionTransformer"]
folds = 5

rolling_forecast_results, combined_predicted_dic = rolling_weighted_forecast(train_dic, test_dic, training_results, folds = folds, freq = data["freq"],  model = model, includeModels = None, excludeModels = excludeModels, selection_mode="selected_only", saveResults = True)

# COMMAND ----------

# DBTITLE 1,Load CV Forecast
import pandas as pd

rolling_forecast_results_dic_file = 'Fold_Forecast_Naive_365_D_5.pkl'
with open(rolling_forecast_results_dic_file, 'rb') as f:
    rolling_forecast_results = pickle.load(f)

combined_predicted_dic = rolling_forecast_results['combined_predicted_dic'] 
rolling_forecast_results = rolling_forecast_results["final_forecast_df"]

# COMMAND ----------

rolling_forecast_results

# COMMAND ----------

# DBTITLE 1,Plot CV results
from data_forecast.plots.plot_rolling_forecast import plot_rolling_forecasts


# Beispielaufruf der Funktion
plot_rolling_forecasts(rolling_forecast_results, plot_individual_folds=False, forecast_col='opt_method', actual_col='total', date_col='date')


# COMMAND ----------

from data_forecast.evaluation.evaluation import *

# Beispielaufruf
evaluate_forecast_performance(rolling_forecast_results, metric = "MAPE", date_col="date", actual_col="total", modelName=rolling_forecast_results_dic_file, exclude_cols=["fold"])

# COMMAND ----------

# DBTITLE 1,Plot ts of a specific level
# import matplotlib.pyplot as plt
# level = ('Region', 'Purpose')

# df = lower_level_series[level]
# print(lower_level_series[level])
# plt.figure(figsize=(12, 8))

# # Tatsächliche Werte plotten
# for ts_id, group in df.groupby('ts_id'):
#     plt.plot(group['date'], group['total'])

# # Gewichtete Prognosen plotten
# # Du musst sicherstellen, dass du hier die richtigen gewichteten Prognosen hast, z.B. aus aggregated_forecasts_per_level oder weighted_forecast

# # Beispiel für das Plotten gewichteter Prognosen, falls vorhanden
# # plt.plot(df_name['date'], weighted_forecast, label='Gewichtete Prognosen', color='orange')

# plt.xlabel('Datum')
# plt.ylabel('Wert')
# #plt.title(df_name)
# plt.legend()
# plt.grid(True)
# plt.show()



# num_series = len(lower_level_series) + 1  # +1 for top_level_series
# fig, axs = plt.subplots(num_series, 1, figsize=(10, 5 * num_series), sharex=True)

# for i, (key, df) in enumerate(lower_level_series.items()):
#     axs[i].plot(df['total'], df['total'], label=key)
#     axs[i].set_title(f'Series: {key}')
#     axs[i].set_ylabel('Value')
#     axs[i].legend()

# axs[num_series - 1].plot(top_level_series['date'], top_level_series['total'], label='top_level_series', color='red')
# axs[num_series - 1].set_title('Top Level Series')
# axs[num_series - 1].set_ylabel('Value')
# axs[num_series - 1].legend()

# plt.xlabel('Date')
# plt.show()

# COMMAND ----------

# DBTITLE 1,COMBINE metrics
def combine_rolling_forecasts(forecast_dict):
    """
    Kombiniert mehrere DataFrames aus einem Dictionary, aggregiert die Forecast-Werte 
    nach Datum und berechnet den Mittelwert, wenn es mehrere Werte für dasselbe Datum gibt.

    Parameters:
    forecast_dict (dict): Dictionary, dessen Werte DataFrames sind, die kombiniert werden sollen.

    Returns:
    pd.DataFrame: Ein DataFrame, der die aggregierten Forecast-Werte nach Datum enthält.
    """
    # Kombiniere alle DataFrames in einem
    combined_df = pd.concat(forecast_dict.values(), ignore_index=True)
    
    # Gruppiere nach Datum und berechne den Mittelwert der Forecast-Werte
    aggregated_df = combined_df.groupby('date').agg({
        'total': 'mean',        # Summiere die 'total' Werte (falls notwendig)
        'opt_method': 'mean'     # Berechne den Mittelwert der 'forecast' Werte
    }).reset_index()
    
    return aggregated_df
    
def evaluate_forecast_performance(df):
    """
    Evaluates the performance of the forecast in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'date', 'total', and 'forecast'
    
    Returns:
    float: Mean Absolute Percentage Error (MAPE)
    pd.DataFrame: DataFrame containing the true deviation and percentage deviation for each date
    """
    # Entferne alle Zeilen mit NA im 'forecast'
    df_clean = df.dropna(subset=['opt_method']).copy()
    
    # Berechne die wahre Abweichung
    df_clean['true_deviation'] = df_clean['total'] - df_clean['opt_method']
    
    # Berechne die absolute Abweichung
    df_clean['absolute_deviation'] = abs(df_clean['true_deviation'])
    
    # Berechne die prozentuale Abweichung
    df_clean['percentage_deviation'] = (df_clean['absolute_deviation'] / df_clean['total']) * 100
    
    # Berechne den MAPE
    mape = df_clean['percentage_deviation'].median() #.mean()
   
    
    # Formatierung der Zahlen mit Punkten als Tausendertrennzeichen
    df_clean['total'] = df_clean['total'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['opt_method'] = df_clean['opt_method'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['true_deviation'] = df_clean['true_deviation'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['percentage_deviation'] = df_clean['percentage_deviation'].apply(lambda x: f"{x:,.2f}".replace(',', '.'))

    return mape, df_clean[['date', 'total', 'opt_method', 'true_deviation', 'percentage_deviation']]


result_df = combine_rolling_forecasts(rolling_forecast_results)

result = evaluate_forecast_performance(result_df)

result[0]

pd.to_numeric(result[1]["percentage_deviation"], errors='coerce').mean()

result

# COMMAND ----------

# DBTITLE 1,Reduce weights even more
# # Finde die Indizes der 10 größten Gewichte
# top_10_indices = np.argpartition(avg_weights, -10)[-10:]

# # Erzeuge ein Array mit Nullen der gleichen Form wie avg_weights
# top_weights = np.zeros_like(avg_weights)

# # Setze die 10 größten Gewichte in das neue Array
# top_weights[top_10_indices] = avg_weights[top_10_indices]

# # Skaliere die 10 größten Gewichte so, dass ihre Summe 1 ergibt
# scaled_top_weights = top_weights / np.sum(top_weights)

# print("Top 10 Gewichte:", scaled_top_weights)
