from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
from typing import Tuple
import signal
import os

def train_AutoETS_and_forecast(train_df, test_periods, freq, date_col = "ds", id_col = "unique_id", actuals_col= "y", set_index =True, enable_ensemble = True, eval_metric = "MAE", verbosity = 0, time_limit = 60 * 60 * 24, random_seed = 123):
    """
    Trainiert ein AutoETS-Modell und erstellt Prognosen sowie fitted values für den Trainingsbereich.

    Parameters:
    ----------
    train_df : pd.DataFrame
        Trainingsdaten mit Spalten 'ds' (Datum), 'y' (Wert) und einer zusätzlichen ID-Spalte (z.B. 'unique_id').
    test_periods : int
        Anzahl der Zukunftsperioden für die Prognose.
    freq : str
        Frequenz der Daten, z.B. 'D' für täglich, 'W' für wöchentlich.
    date_col : str
        Der Name der Spalte, die Datumsinformationen enthält. Standardmäßig 'ds'.
    actuals_col : str
        Der Name der Spalte, die die abhängige Variable enthält. Standardmäßig 'y'.
    id_col : str
        Der Name der Spalte im DataFrame, die die Zeitreihen-IDs enthält. Standardmäßig 'unique_id'.
    set_index : bool
        Ob die `id_col` als Index gesetzt werden soll. Standardmäßig True.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Prognosewerte und Fitted Values mit den ursprünglichen Spaltennamen.
    """
    
    # Funktion, die bei Timeout aufgerufen wird
    def handler(signum, frame):
        raise TimeoutError("Das Training hat das Zeitlimit überschritten.")


    # Speichern der ursprünglichen Spaltennamen
    original_columns = { 'date': date_col, 'ts_id': id_col, 'target': actuals_col }
    
    # Prüfen, ob id_col eine Index-Spalte ist und ggf. zurücksetzen
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Umbenennen der Spalten für StatsForecast
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})

    # DataFrame auf relevante Spalten beschränken
    train_df = train_df[['unique_id', 'ds', 'y']]

    # data auf ende des Monats setzen:
    
    if freq in ["M", "ME"]:
        season_length = 12
    elif freq == "D":
        season_length = 7
    elif freq in ["Q", "QE", "QS"]:
        season_length = 4

    #print("TRAIN DF")
    #print(train_df.drop(columns=['unique_id']))

    # Setze die Umgebungsvariable, um die Warnung bezüglich der ID zu unterdrücken
    os.environ["NIXTLA_ID_AS_COL"] = "true"


    # Initialisiere das AutoETS-Modell ohne df
    fcst_AutoETS = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq=freq,
        n_jobs=-1
    )

    # Passe das Modell an die Trainingsdaten an, indem `train_df` direkt an `fit` übergeben wird
    fcst_AutoETS = fcst_AutoETS.fit(train_df)  # df-Argument wird nur hier verwendet

    # Erstellen der Prognose für Out-of-Sample und In-Sample Werte, indem `train_df` an `forecast` übergeben wird
    forecast_result = fcst_AutoETS.forecast(df=train_df, h=test_periods, fitted=True).reset_index()
        
    last_dates = forecast_result["ds"].drop_duplicates().sort_values().tail(test_periods)
    # # Out-of-Sample
    Y_hat_df_AutoETS = forecast_result[forecast_result["ds"].isin(last_dates)].rename(columns={'AutoETS': 'pred'})
    # In-Sample
    Y_fitted_df_AutoETS = forecast_result[~forecast_result["ds"].isin(last_dates)].rename(columns={'AutoETS': 'pred'})


    # Spalten zurück in die Originalnamen umbenennen
    Y_hat_df_AutoETS = Y_hat_df_AutoETS.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})
    Y_fitted_df_AutoETS = Y_fitted_df_AutoETS.rename(columns={'ds': original_columns['date'], 'unique_id': original_columns['ts_id'], 'y': original_columns['target']})

    # Setze 'ts_id' als Index zurück, wenn `set_index` auf True gesetzt ist
    if set_index:
        Y_hat_df_AutoETS.set_index(original_columns['ts_id'], inplace=True)
        Y_fitted_df_AutoETS.set_index(original_columns['ts_id'], inplace=True)


    return Y_hat_df_AutoETS, Y_fitted_df_AutoETS
