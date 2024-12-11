import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def plot_all_forecasts(forecast_dict, train_df=None, date='date', actual_label="total", pred_label="opt_method", combine_plots=True):
    """
    Plottet die rollierenden Prognosen gegen die tatsächlichen Werte aus einem Dictionary mit DataFrames.
    Optional: Alle Forecasts in einem Diagramm oder separat darstellen.
    
    Parameters:
    forecast_dict (dict): Dictionary mit DataFrames, wobei jeder Key einem Dateinamen entspricht.
    train_df (pd.DataFrame): Optional, Trainingsdaten, die gejoint werden (auf Basis der Datums-Spalte).
    date (str): Spaltenname für die Datumswerte.
    actual_label (str): Spaltenname für die tatsächlichen Werte.
    pred_label (str): Spaltenname für die Prognosewerte.
    combine_plots (bool): Wenn True, werden alle Forecasts in einem Diagramm dargestellt. Wenn False, wird für jeden DataFrame ein eigenes Diagramm erstellt.
    """
    
    if combine_plots:
        plt.figure(figsize=(14, 7))
    
    # Über alle Forecasts im Dictionary iterieren
    for file_name, combined_result in forecast_dict.items():
        forecast_df = combined_result.copy()
        
        # Wenn train_df gegeben ist, auf Basis der 'date'-Spalte joinen
        if train_df is not None:
            # Join auf 'date' durchführen
            forecast_df = pd.merge(forecast_df, train_df[[date, actual_label]], on=date, how='outer', suffixes=('_x', '_y'))

            # Kombiniere die total_x und total_y Spalten, wobei total_y Priorität hat
            forecast_df['total'] = forecast_df[f'{actual_label}_y'].combine_first(forecast_df[f'{actual_label}_x'])
            print(forecast_df)
        if combine_plots:
            # Plotte tatsächliche Werte nur einmal
            if list(forecast_dict.keys()).index(file_name) == 0:
                plt.plot(forecast_df[date], forecast_df['total'], 
                         label='Tatsächliche Werte', color='blue', linewidth=2)
            
            # Prognose plotten
            plt.plot(forecast_df[date], forecast_df[pred_label], 
                     label=file_name, linewidth=1.5)
        else:
            # Einzelnes Plot pro Forecast
            plt.figure(figsize=(14, 7))
            plt.plot(forecast_df[date], forecast_df['total'], 
                     label='Tatsächliche Werte', color='blue', linewidth=2)
            plt.plot(forecast_df[date], forecast_df[pred_label], 
                     label='Prognose', color='orange', linewidth=1.5)
            plt.xlabel('Datum')
            plt.ylabel('Wert')
            plt.title(f'Tatsächliche Werte vs. Prognose - {file_name}')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()
    
    if combine_plots:
        plt.xlabel('Datum')
        plt.ylabel('Wert')
        plt.title('Tatsächliche Werte vs. Gewichtete Prognosen')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()



# import os
# import pickle
# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_all_forecasts(forecasts_path, date='date', actual_label="total", pred_label="opt_method", combine_plots=True):
#     """
#     Plottet die rollierenden Prognosen gegen die tatsächlichen Werte.
#     Optional: Alle Forecasts in einem Diagramm oder separat darstellen.
    
#     Parameters:
#     forecasts_path (str): Pfad zu den Verzeichnissen, die die .pkl Forecast-Dateien enthalten.
#     date (str): Spaltenname für die Datumswerte.
#     actual_label (str): Spaltenname für die tatsächlichen Werte.
#     pred_label (str): Spaltenname für die Prognosewerte.
#     combine_plots (bool): Wenn True, werden alle Forecasts in einem Diagramm dargestellt. Wenn False, wird für jede Datei ein eigenes Diagramm erstellt.
#     """

#     if combine_plots:
#         plt.figure(figsize=(14, 7))
    
#     # Alle Dateien im Verzeichnis auflisten
#     for file_name in os.listdir(forecasts_path):
#         # Überprüfen, ob der Dateiname mit 'Forecast' beginnt und die Endung '.pkl' hat
#         if file_name.startswith('Forecast') and file_name.endswith('.pkl'):
#             pkl_filename = os.path.join(forecasts_path, file_name)
            
#             # Datei laden
#             with open(pkl_filename, 'rb') as f:
#                 combined_result = pickle.load(f)
            
#             if combine_plots:
#                 # Plotte tatsächliche Werte nur einmal
#                 if file_name == os.listdir(forecasts_path)[0]:
#                     plt.plot(combined_result["forecast_df"][date], combined_result["forecast_df"][actual_label], 
#                              label='Tatsächliche Werte', color='blue', linewidth=2)
                
#                 # Prognose plotten
#                 plt.plot(combined_result["forecast_df"][date], combined_result["forecast_df"][pred_label], 
#                          label=file_name, linewidth=1.5)
#             else:
#                 # Einzelnes Plot pro Datei
#                 plt.figure(figsize=(14, 7))
#                 plt.plot(combined_result["forecast_df"][date], combined_result["forecast_df"][actual_label], 
#                          label='Tatsächliche Werte', color='blue', linewidth=2)
#                 plt.plot(combined_result["forecast_df"][date], combined_result["forecast_df"][pred_label], 
#                          label='Prognose', color='orange', linewidth=1.5)
#                 plt.xlabel('Datum')
#                 plt.ylabel('Wert')
#                 plt.title(f'Tatsächliche Werte vs. Prognose - {file_name}')
#                 plt.legend(loc='upper left')
#                 plt.grid(True)
#                 plt.show()
    
#     if combine_plots:
#         plt.xlabel('Datum')
#         plt.ylabel('Wert')
#         plt.title('Tatsächliche Werte vs. Gewichtete Prognosen')
#         plt.legend(loc='upper left')
#         plt.grid(True)
#         plt.show()