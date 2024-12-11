# Databricks notebook source
# MAGIC %pip install hyperts

# COMMAND ----------

# MAGIC %pip install --upgrade pandas

# COMMAND ----------

from hyperts.datasets import load_network_traffic
from sklearn.model_selection import train_test_split
from hyperts import make_experiment
import pandas as pd
#nicht mit neuestem pandas pkg kompatibel

# Lade die Daten
data = load_network_traffic()


# Manuelle Aufteilung der Daten
train_data, test_data = train_test_split(data, test_size=168, shuffle=False) # shuffle=False für Zeitreihen

print(train_data)

experiment = make_experiment(train_data=train_data.copy(),
                            task='forecast',
                            timestamp='TimeStamp',
                            covariables=['HourSin', 'WeekCos', 'CBWD'])
model = experiment.run()


# Erstelle ein Experiment und gebe bestimmte Modelle an
experiment = make_experiment(train_data=train_data.copy(), 
                            #task='forecast',
                            timestamp='TimeStamp',
                             mode='dl', 
                             target='value', 
                             task='univariate-forecast',
                             algorithms=['ARIMA', 'Prophet', 'LSTM'])  # Bestimmte Modelle angeben

# Führe das Experiment durch
model = experiment.run()

# Extrahiere das beste Modell
best_model = model.best_trial
print(best_model)



# COMMAND ----------

from hyperts import make_experiment
from hyperts.datasets import load_network_traffic

# Lade die Daten
data = load_network_traffic()
train_data, test_data = data.split_data()

# Erstelle ein Experiment und gebe bestimmte Modelle an
experiment = make_experiment(train_data, 
                             mode='dl', 
                             target='value', 
                             task='univariate-forecast',
                             algorithms=['ARIMA', 'Prophet', 'LSTM']) # Bestimmte Modelle angeben

# Führe das Experiment durch
model = experiment.run()

# Extrahiere das beste Modell
best_model = model.best_trial
print(best_model)
