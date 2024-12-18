# Databricks notebook source
# Importiere notwendige Bibliotheken
import pandas as pd
from pyspark.sql import SparkSession

# Lade die Daten in ein Spark DataFrame

# Lade die Daten in ein Spark DataFrame
df = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`m_5_sales_long`
""")

# Konvertiere das Spark DataFrame in ein Pandas DataFrame
pandas_df = df.toPandas()

pandas_df

# COMMAND ----------

# Speichere das Pandas DataFrame als CSV-Datei
#metric_result.to_csv('/Workspace/Users/simon.pradel@telefonica.de/P&L_Prediction/metric_result.csv', index=False)
pandas_df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/m_5_sales_long.csv', index=False)

# COMMAND ----------

# read csv from workspace
from pathlib import Path
import pandas as pd 

# Pandas can handle relative paths
path = '/Workspace/Users/simon.pradel@telefonica.de/P&L_Prediction/metric_result.csv'

pandas_df = pd.read_csv(path)

pandas_df
