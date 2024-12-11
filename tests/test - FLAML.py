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

# MAGIC %pip install FLAML[automl,forecast,ts_forecast] openml
# MAGIC %pip install pytorch-forecasting==0.9.0 pytorch-lightning==1.5.10
# MAGIC

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

#https://github.com/microsoft/FLAML/blob/main/notebook/automl_time_series_forecast.ipynb

# COMMAND ----------

import datetime
import pandas as pd

train_df = aggregated_data["Y_df"].copy()
# Rename columns for FLAML compatibility
train_df = train_df.rename(columns={"ds": "date", "y": 'total', "unique_id": 'item_id'})

# Ensure the 'ds' column is in datetime format
train_df['date'] = pd.to_datetime(train_df['date']) + datetime.timedelta(days=1)

# Create time_idx based on year and month
train_df["time_idx"] = train_df["date"].dt.year * 12 + train_df["date"].dt.month 
train_df["time_idx"] -= train_df["time_idx"].min()  # Normalize so that time_idx starts from 0


train_df = train_df[['item_id', 'date', 'total', 'time_idx']]
train_df = train_df.sort_values(["item_id", "date"])
ts_col = train_df.pop("date")
train_df.insert(0, "date", ts_col)
# FLAML assumes input is not sorted, but we sort here for comparison purposes with y_test
train_df = train_df.sort_values(["item_id", "date"])
#train_df['time_idx'] = train_df['time_idx'].astype(str)
#train_df['date'] = pd.to_datetime(train_df['date']).dt.date
#train_df.set_index('date', inplace=True)
# Beispiel: Gruppierung nach 'ts_id' und Verarbeitung der Zeitstempel innerhalb jeder Gruppe

# # Beispiel: Gruppierung nach 'ts_id' und Verarbeitung der Zeitstempel innerhalb jeder Gruppe
# grouped = train_df.groupby('item_id')

# # Entfernen von Duplikaten in der 'date'-Spalte innerhalb jeder Gruppe
# df_cleaned = grouped.apply(lambda group: group.drop_duplicates(subset='date'))

# # Stelle sicher, dass der Index sortiert ist
# df_cleaned = df_cleaned.sort_values(by=['date'])

# # Setze den Index auf die 'date'-Spalte und weise die Frequenz zu
# df_cleaned = df_cleaned.set_index('date', inplace=False)  # Setze 'date' als Index

# # Falls gewünscht, Frequenz für jede Gruppe hinzufügen (Tagesfrequenz im Beispiel)
# df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('M')

# # Zeige den Index, um sicherzustellen, dass 'date' nun der Index ist
# print(df_cleaned.index)

#train_df = df_cleaned.reset_index()
train_df = train_df.sort_values('item_id')
train_df = train_df.sort_index()
train_df = train_df.sort_values(["item_id", "date"])

y_train = train_df['total']
X_train = train_df.drop(columns=['total'])  # Drop target column


#X_train.info()
X_train

# COMMAND ----------

train_df.info()
train_df

# COMMAND ----------

print('Best ML leaner:', automl.best_estimator)

# COMMAND ----------

from flaml import AutoML
automl = AutoML()
settings = {
    "time_budget": 300,  # total running time in seconds
    "metric": "mape",  # primary metric
    "task": "ts_forecast_panel",  # task type
    #"log_file_name": "stallion_forecast.log",  # flaml log file
    "eval_method": "holdout",
    "label": "total",
}

includeModels=["rf"]
if includeModels:
    estimator_list = includeModels


fit_kwargs_by_estimator = {
    "tft": {
        "max_encoder_length": 24,
        "static_categoricals": ["item_id"],
        "time_varying_known_reals": [
            "time_idx",
        ],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [
            "y",  # always need a 'y' column for the target column
        ],
        "batch_size": 128,
        "gpu_per_trial": 0,
    }
}
"""The main flaml automl API"""
automl.fit(
    dataframe=train_df,
    #X_train=X_train,
    #y_train=y_train,
    **settings,
    period=12,
    group_ids=["item_id"],
    #fit_kwargs_by_estimator=fit_kwargs_by_estimator, 
    estimator_list=estimator_list, 
)

# ['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'prophet', 'arima', 'sarimax']

# COMMAND ----------


# Create future dates for forecasting
last_date = train_df.index.max()
future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq=freq)[1:]

future_df = pd.DataFrame({'ds': future_dates})

# Generate forecast
try:
    forecast = automl.predict(future_df)
except Exception as e:
    print("Error during forecasting:", e)
    return None

# Format forecast output
forecast_df = pd.DataFrame({
    original_columns['date']: future_df['ds'],
    'pred': forecast
})

if set_index:
    forecast_df.set_index(original_columns['date'], inplace=True)

return forecast_df, automl

# Debug the input DataFrame
print("Aggregated Data Y_df Columns:", aggregated_data["Y_df"].columns)
print("Data Structure of Y_df:")
print(aggregated_data["Y_df"].head())

# Try calling the function with debugging
test = train_flaml_and_forecast(
aggregated_data["Y_df"], 
future_periods=12, 
freq=data["freq"], 
date_col="ds", 
id_col="unique_id", 
actuals_col="y", 
includeModels=["arima"], 
excludeModels=None, 
set_index=False
)

# COMMAND ----------

def train_flaml_and_forecast(train_df, future_periods, freq, date_col, id_col, actuals_col, includeModels=None, excludeModels=None, set_index=False):
    """
    Train a FLAML model and forecast future values.
    """

    # Save original column names
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}

    # Rename columns for FLAML compatibility
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'item_id'})

    # Ensure the 'ds' column is in datetime format
    train_df['ds'] = pd.to_datetime(train_df['ds'])

    # Check if train_df looks correct
    print("Train DataFrame Structure:")
    print(train_df.head())
    print("Train DataFrame Columns:", train_df.columns)

    # Create time_idx based on year and month
    train_df['time_idx'] = (train_df['ds'] - train_df['ds'].min()).dt.days
    train_df["time_idx"] -= train_df["time_idx"].min()  # Normalize so that time_idx starts from 0
    # Berechne den time_idx basierend auf dem Unterschied in Tagen
    train_df['time_idx'] = (train_df['ds'] - train_df['ds'].min()).dt.days
    
    ts_col = train_df.pop("ds")
    train_df.insert(0, "ds", ts_col)    
    train_df = train_df.sort_values(["item_id", "ds"])
    # Überprüfen, ob 'time_idx' eine einzelne Spalte ist
    print(train_df['time_idx'].head())  # Sollte eine einfache Spalte von Ganzzahlen zeigen

    # Jetzt die Datenstruktur nach dem Hinzufügen von time_idx überprüfen
    print(train_df.head())

    # Sicherstellen, dass kein DataFrame mit mehreren Spalten in time_idx eingefügt wird

    # Check if time_idx was created correctly
    print("After adding time_idx:")
    print(train_df[['item_id', 'ds', 'time_idx']].head())

    # Prepare the data for training (check it before continuing)
    try:
        # Ensure required columns exist
        train_df = train_df[['item_id', 'ds', 'y', 'time_idx']]
        #train_df.set_index('ds', inplace=True)
    except Exception as e:
        print("Error during DataFrame preparation:", e)
        return None

    # Ensure no MultiIndex is used (for sanity check)
    print("Final DataFrame structure for training:")
    print(train_df.head())
    print("DataFrame Index:", train_df.index)

    # Initialize AutoML model
    automl = AutoML()

    # Set up custom hyperparameters for panel forecasting
    settings = {
        "task": "ts_forecast_panel",
        "time_budget": 600,
        "eval_method": "holdout",
        "time_col": "time_idx",  # Use the 'time_idx' column for time series
        "metric": "mape",
    }


    # Include specific models if provided
    if includeModels:
        settings['estimator_list'] = includeModels

    # Exclude certain models if provided
    if excludeModels:
        if 'estimator_list' in custom_hp:
            custom_hp['estimator_list'] = [m for m in custom_hp['estimator_list'] if m not in excludeModels]

    # Train the FLAML model
    # Explicitly pass X_train and y_train with appropriate columns
    # print("HEY")
    # print(train_df.info())
    X_train = train_df.drop(columns=['y'])  # Drop target column
    y_train = train_df['y']
    # print("Start")
    # print(y_train.info())
    # print("Start")
    # print(X_train["item_id"])
    # y_train["test"] = X_train["item_id"]
    # print(y_train.info())
    # print("Ende")
###############################

    # Specify kwargs for TimeSeriesDataSet used by TemporalFusionTransformerEstimator
    fit_kwargs_by_estimator = {
        "tft": {
            "max_encoder_length": 24,
            "static_categoricals": ["item_id"],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "variable_groups": {
                
            },  # group of categorical variables can be treated as one variable
            "time_varying_known_reals": [
                "time_idx",
            ],
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": [
                "y",  # always need a 'y' column for the target column
            ],
            "batch_size": 256,
            "max_epochs": 1,
            "gpu_per_trial": -1,
        }
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings, period = 12,  group_ids = ["item_id"], fit_kwargs_by_estimator = fit_kwargs_by_estimator)  # Grouping by 'item_id')


    # Create future dates for forecasting
    last_date = train_df.index.max()
    future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq=freq)[1:]

    future_df = pd.DataFrame({'ds': future_dates})

    # Generate forecast
    try:
        forecast = automl.predict(future_df)
    except Exception as e:
        print("Error during forecasting:", e)
        return None

    # Format forecast output
    forecast_df = pd.DataFrame({
        original_columns['date']: future_df['ds'],
        'pred': forecast
    })

    if set_index:
        forecast_df.set_index(original_columns['date'], inplace=True)

    return forecast_df, automl

# Debug the input DataFrame
print("Aggregated Data Y_df Columns:", aggregated_data["Y_df"].columns)
print("Data Structure of Y_df:")
print(aggregated_data["Y_df"].head())

# Try calling the function with debugging
test = train_flaml_and_forecast(
    aggregated_data["Y_df"], 
    future_periods=12, 
    freq=data["freq"], 
    date_col="ds", 
    id_col="unique_id", 
    actuals_col="y", 
    includeModels=["arima"], 
    excludeModels=None, 
    set_index=False
)


# COMMAND ----------

import pandas as pd
from flaml import AutoML
import datetime


def train_flaml_and_forecast(train_df, future_periods, freq, date_col, id_col, actuals_col, includeModels=None, excludeModels=None, set_index=False):
    """
    Trainiert ein FLAML-Modell und erstellt Prognosen.
    """

    # Speichern der ursprünglichen Spaltennamen
    original_columns = {'date': date_col, 'ts_id': id_col, 'target': actuals_col}

    # Umbenennen der Spalten für FLAML
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'item_id'})
    

    # Konvertieren der Zeitstempel in das richtige Format
    train_df['ds'] = pd.to_datetime(train_df['ds'])

    # Statische Features extrahieren (alle Spalten außer 'ds' und 'y')
    remaining_cols = train_df.drop(columns=['ds', 'y']).columns
    static_features_df = train_df[remaining_cols].drop_duplicates()
    
    # Daten für das Training vorbereiten
    train_df = train_df[['item_id', 'ds', 'y']]
    train_df['time_idx'] = train_df['ds'].dt.year * 12 + train_df['ds'].dt.month
    train_df.set_index('ds', inplace=True)
    print(train_df.head())
    # Initialisieren des AutoML Modells
    automl = AutoML()

    # Panel-spezifische Einstellungen
    custom_hp = {
        "task": "ts_forecast_panel",
        "time_budget": 600,
        "eval_method": "holdout",
        "group_ids": ["item_id"],  # Gruppierungsvariable
        "time_col": "ds",  # Zeitspalte
        "period": 12,
        "metric": "mape",
    }

    # Include specific models if provided
    if includeModels:
        custom_hp['estimator_list'] = includeModels

    # Exclude certain models if provided
    if excludeModels:
        if 'estimator_list' in custom_hp:
            custom_hp['estimator_list'] = [m for m in custom_hp['estimator_list'] if m not in excludeModels]

    # FLAML-Modelltraining
    automl.fit(dataframe=train_df, label='y', **custom_hp)

    # Zukunftsdaten für Vorhersage erstellen
    last_date = train_df['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=future_periods + 1, freq=freq)[1:]  # Startet ab dem Tag nach der letzten Beobachtung

    future_df = pd.DataFrame({'ds': future_dates})

    # Prognose erstellen
    forecast = automl.predict(future_df)

    # Formatierung der Ausgabe
    forecast_df = pd.DataFrame({
        original_columns['date']: future_df['ds'],
        'pred': forecast
    })

    if set_index:
        forecast_df.set_index(original_columns['date'], inplace=True)

    return forecast_df, automl


# ['xgboost', 'xgb_limitdepth', 'rf', 'lgbm', 'extra_tree', 'arima', 'sarimax', 'holt-winters', 'catboost', 'tft', 'prophet']
test = train_flaml_and_forecast(aggregated_data["Y_df"], future_periods = 12, freq=data["freq"], date_col = "ds", id_col = "unique_id", actuals_col = "y", includeModels=["arima"], excludeModels=None, set_index = False)

# COMMAND ----------

from tools.models.AutoTS import *
test = train_flaml_and_forecast(aggregated_data["Y_df"], test_periods = 12, freq=data["freq"], date_col = "ds", id_col = "unique_id", actuals_col = "y", includeModels=["ETS"], excludeModels=None, set_index = False)

# COMMAND ----------

import statsmodels.api as sm

data = sm.datasets.co2.load_pandas().data
# data is given in weeks, but the task is to predict monthly, so use monthly averages instead
data = data["co2"].resample("MS").mean()
data = data.bfill().ffill()  # makes sure there are no missing values
data = data.to_frame().reset_index()
data

# COMMAND ----------


num_samples = data.shape[0]
time_horizon = 12
split_idx = num_samples - time_horizon
train_df = data[
    :split_idx
]  # train_df is a dataframe with two columns: timestamp and label
X_test = data[split_idx:][
    "index"
].to_frame()  # X_test is a dataframe with dates for prediction
y_test = data[split_idx:][
    "co2"
]  # y_test is a series of the values corresponding to the dates for prediction

from flaml import AutoML

automl = AutoML()
settings = {
    "time_budget": 10,  # total running time in seconds
    "metric": "mape",  # primary metric for validation: 'mape' is generally used for forecast tasks
    "task": "ts_forecast",  # task type
    "log_file_name": "CO2_forecast.log",  # flaml log file
    "eval_method": "holdout",  # validation method can be chosen from ['auto', 'holdout', 'cv']
    "seed": 7654321,  # random seed
}

automl.fit(
    dataframe=train_df,  # training data
    label="co2",  # label column
    period=time_horizon,  # key word argument 'period' must be included for forecast task)
    **settings
)

# COMMAND ----------

flaml_y_pred = automl.predict(X_test)
import matplotlib.pyplot as plt
#plt.plot(X_train, y_train, label="Actual level")
plt.plot(X_test, y_test, label="Actual level")
plt.plot(X_test, flaml_y_pred, label="FLAML forecast")
plt.xlabel("Date")
plt.ylabel("CO2 Levels")
plt.legend()

# COMMAND ----------

def get_stalliion_data():
    from pytorch_forecasting.data.examples import get_stallion_data

    data = get_stallion_data()
    # add time index - For datasets with no missing values, FLAML will automate this process
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_sku"] = data.groupby(
        ["time_idx", "sku"], observed=True
    ).volume.transform("mean")
    data["avg_volume_by_agency"] = data.groupby(
        ["time_idx", "agency"], observed=True
    ).volume.transform("mean")
    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "beer_capital",
        "music_fest",
    ]
    data[special_days] = (
        data[special_days]
        .apply(lambda x: x.map({0: "-", 1: x.name}))
        .astype("category")
    )
    return data, special_days


data, special_days = get_stalliion_data()
time_horizon = 6  # predict six months
training_cutoff = data["time_idx"].max() - time_horizon
data["time_idx"] = data["time_idx"].astype("int")
ts_col = data.pop("date")
data.insert(0, "date", ts_col)
# FLAML assumes input is not sorted, but we sort here for comparison purposes with y_test
data = data.sort_values(["agency", "sku", "date"])
X_train = data[lambda x: x.time_idx <= training_cutoff]
X_test = data[lambda x: x.time_idx > training_cutoff]
y_train = X_train.pop("volume")
y_test = X_test.pop("volume")
automl = AutoML()
# Configure settings for FLAML model
settings = {
    "time_budget": budget,  # total running time in seconds
    "metric": "mape",  # primary metric
    "task": "ts_forecast_panel",  # task type
    "log_file_name": "test/stallion_forecast.log",  # flaml log file
    "eval_method": "holdout",
}
# Specify kwargs for TimeSeriesDataSet used by TemporalFusionTransformerEstimator
fit_kwargs_by_estimator = {
    "tft": {
        "max_encoder_length": 24,
        "static_categoricals": ["agency", "sku"],
        "static_reals": ["avg_population_2017", "avg_yearly_household_income_2017"],
        "time_varying_known_categoricals": ["special_days", "month"],
        "variable_groups": {
            "special_days": special_days
        },  # group of categorical variables can be treated as one variable
        "time_varying_known_reals": [
            "time_idx",
            "price_regular",
            "discount_in_percent",
        ],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [
            "y",  # always need a 'y' column for the target column
            "log_volume",
            "industry_volume",
            "soda_volume",
            "avg_max_temp",
            "avg_volume_by_agency",
            "avg_volume_by_sku",
        ],
        "batch_size": 256,
        "max_epochs": 1,
        "gpu_per_trial": -1,
    }
}
# Train the model
automl.fit(
    X_train=X_train,
    y_train=y_train,
    **settings,
    period=time_horizon,
    group_ids=["agency", "sku"],
    fit_kwargs_by_estimator=fit_kwargs_by_estimator,
)
# Compute predictions of testing dataset
y_pred = automl.predict(X_test)
print(y_test)
print(y_pred)
# best model
print(automl.model.estimator)
