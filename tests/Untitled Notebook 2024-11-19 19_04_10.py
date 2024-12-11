# Databricks notebook source
# MAGIC %pip list

# COMMAND ----------

# MAGIC %pip install pipreqs
# MAGIC

# COMMAND ----------

# MAGIC %pip freeze

# COMMAND ----------

pipreqs /Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction

# COMMAND ----------

C = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

# Erstelle das Dictionary mit den Teilmengen als Keys
dict_of_subsets = {subset: None for subset in C}

# Ausgabe des Dictionaries
print(dict_of_subsets)

# COMMAND ----------

import pandas as pd

# Beispiel-Daten
data = {
    'date': ['2024-11-01', '2024-11-01', '2024-11-02', '2024-11-02', '2024-11-01', '2024-11-01', '2024-11-01'],
    'group1': ['A', 'A', 'B', 'B', 'A', 'B', 'B'],
    'group2': ['X', 'Y', 'X', 'Y', 'X', 'X', 'X'],
    'y': [10, 15, 20, 25, 5, 6, 14]
}
df = pd.DataFrame(data)

# Gruppieren und Summe bilden
grouped_dataset = df.groupby(['date', 'group1', 'group2'])['y'].sum().reset_index()

# Umformen in eine Matrix (Multi-Index für Gruppen)
matrix = grouped_dataset.pivot(index=['group1', 'group2'], columns='date', values='y')

# Ergebnis anzeigen
# Ergebnis anzeigen
print("Matrix:")
print(matrix)

# Zeilenweise Iteration
print("\nZeilenweise Iteration:")
for (group1, group2), row in matrix.iterrows():
    print(f"Group: {group1}-{group2}")
    print(row.to_dict()) 

