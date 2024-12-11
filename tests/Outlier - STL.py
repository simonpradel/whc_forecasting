# Databricks notebook source
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Beispiel: Zeitreihe erstellen
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = pd.Series(np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100), index=date_range)

# Simuliere einige Ausreißer
data.iloc[5] = 5
data.iloc[50] = -2

# STL-Dekomposition anwenden
stl = STL(data, seasonal=13)
result = stl.fit()

# Residuen extrahieren
residual = result.resid

# Schwellenwert für Ausreißer definieren (z.B. 3 Standardabweichungen)
threshold = 3 * np.std(residual)

# Ausreißer identifizieren
outliers = np.abs(residual) > threshold

# Ausreißer interpolieren
data_clean = data.copy()
data_clean[outliers] = np.nan
data_clean.interpolate(inplace=True)

# Ergebnisse plotten
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original')
plt.plot(data_clean, label='Bereinigte Daten', linestyle='--')
plt.scatter(data.index[outliers], data[outliers], color='red', label='Ausreißer')
plt.legend()
plt.show()


# COMMAND ----------

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Beispiel: Zeitreihe erstellen
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = pd.Series(np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100), index=date_range)

# Simuliere einige Ausreißer
data.iloc[5] = 5
data.iloc[50] = -5

# Erstelle einen DataFrame
df = pd.DataFrame({'Original': data})

# STL-Dekomposition anwenden
stl = STL(data, seasonal=13)
result = stl.fit()

# Residuen extrahieren
residual = result.resid

# Schwellenwert für Ausreißer definieren (z.B. 3 Standardabweichungen)
threshold = 3 * np.std(residual)

# Automatisch identifizierte Ausreißer
automatic_outliers = np.abs(residual) > threshold

# Manuell festgelegte Ausreißer
manual_outliers = [17, 23]  # Indexe der Punkte, die du manuell interpolieren möchtest

# Kopiere die Originaldaten
data_clean = data.copy()
print(data_clean)
# Setze sowohl automatische als auch manuelle Ausreißer auf NaN
data_clean[automatic_outliers] = np.nan
data_clean.iloc[manual_outliers] = np.nan

# Interpoliere die fehlenden Werte (z.B. linear)
data_clean_interpolated = data_clean.interpolate()

# Speichere die interpolierten Werte in eine neue Spalte im DataFrame
df['Interpolated'] = data_clean_interpolated

# Ergebnisse plotten
plt.figure(figsize=(12, 6))
plt.plot(df['Original'], label='Original', color='blue')
plt.plot(df['Interpolated'], label='Interpolierte Daten', linestyle='--', color='orange')
plt.scatter(df.index[automatic_outliers], df['Original'][automatic_outliers], color='red', label='Automatische Ausreißer')
plt.scatter(df.index[manual_outliers], df['Original'].iloc[manual_outliers], color='green', label='Manuelle Ausreißer')
plt.legend()
plt.show()

# Ausgabe des DataFrames
df.display()





# COMMAND ----------

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Beispiel: Zeitreihen-Datensatz mit zwei Gruppierungsspalten erstellen (z.B. 'group' und 'subgroup')
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
groups = ['A', 'B', 'C']  # Hauptgruppen
subgroups = ['X', 'Y']  # Untergruppen

# Erstelle einen DataFrame mit hierarchischen Zeitreihen-Daten
data = []
for group in groups:
    for subgroup in subgroups:
        values = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
        if group == 'A':
            values += 1  # Unterscheidung der Gruppe A
        elif group == 'B':
            values += 0.5  # Unterscheidung der Gruppe B
        elif group == 'C':
            values += 0  # Oberste Zeitreihe, kein Offset
        data.append(pd.DataFrame({'date': date_range, 'value': values, 'group': group, 'subgroup': subgroup}))

df = pd.concat(data).reset_index(drop=True)

# Simuliere einige Ausreißer
df.loc[(df['group'] == 'A') & (df['subgroup'] == 'X') & (df['date'] == '2020-01-06'), 'value'] = 3
df.loc[(df['group'] == 'B') & (df['subgroup'] == 'Y') & (df['date'] == '2020-02-20'), 'value'] = -5

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Funktion zur STL-Dekomposition und Ausreißerbehandlung pro Gruppe
def clean_outliers(df, group_columns, value_column='value', seasonal=12, threshold_factor=3):
    df_cleaned = df.copy()
    
    # Gruppiere den DataFrame basierend auf den angegebenen Spalten
    groups = df.groupby(group_columns)
    
    for name, group in groups:
        group_data = group.set_index('date')[value_column]
        
        try:
            # STL-Dekomposition
            stl = STL(group_data, seasonal=seasonal)
            result = stl.fit()
            
            # Residuen und Schwellenwert für Ausreißer
            residual = result.resid
            threshold = threshold_factor * np.std(residual)
            
            # Automatische Ausreißer identifizieren
            automatic_outliers = np.abs(residual) > threshold
            
            # Ersetze Ausreißer mit NaN
            group_data_clean = group_data.copy()
            group_data_clean[automatic_outliers] = np.nan
            
            # Interpoliere fehlende Werte
            group_data_clean_interpolated = group_data_clean.interpolate()
            
            # Gefittete und bereinigte Werte in den DataFrame einfügen
            mask = (df_cleaned[group_columns] == pd.Series(name, index=group_columns)).all(axis=1)
            df_cleaned.loc[mask, 'cleaned_value'] = group_data_clean_interpolated.values
        
        except Exception as e:
            print(f"Fehler bei Gruppe {name}: {e}")
            # Falls ein Fehler auftritt, setze die bereinigten Werte gleich den Originalwerten
            mask = (df_cleaned[group_columns] == pd.Series(name, index=group_columns)).all(axis=1)
            df_cleaned.loc[mask, 'cleaned_value'] = group_data.values
    
    return df_cleaned

# Beispiel für die Anwendung der Funktion
# Erstelle hier deinen DataFrame df, gruppiere nach den gewünschten Spalten und wähle die Spalte, die bereinigt werden soll
# df_cleaned =


# Definiere die Spalten, die die Zeitreihen identifizieren (zwei Gruppierungsspalten: 'group' und 'subgroup')
# Definiere die Spalten, die die Zeitreihen identifizieren (zwei Gruppierungsspalten: 'group' und 'subgroup')
group_columns = ['group', 'subgroup']

# Daten bereinigen
df_cleaned = clean_outliers(df, group_columns, value_column='value', seasonal=13, threshold_factor=3)

# Ergebnisse plotten für jede Kombination von Gruppen
unique_combinations = df_cleaned[group_columns].drop_duplicates()

plt.figure(figsize=(12, 10))
for i, (_, row) in enumerate(unique_combinations.iterrows()):
    # Dynamische Extraktion der Gruppenspalten
    query_conditions = ' & '.join([f"{col} == '{row[col]}'" for col in group_columns])
    df_group = df_cleaned.query(query_conditions)
    
    plt.subplot(len(unique_combinations), 1, i + 1)
    plt.plot(df_group['date'], df_group['value'], label=f'Original - {row.to_dict()}', color='blue')
    plt.plot(df_group['date'], df_group['cleaned_value'], label=f'Bereinigt - {row.to_dict()}', linestyle='--', color='orange')
    
    plt.title(f"Gruppe {row.to_dict()}")
    plt.legend()

plt.tight_layout()
plt.show()

# Ausgabe des bereinigten DataFrames
df_cleaned.head()


