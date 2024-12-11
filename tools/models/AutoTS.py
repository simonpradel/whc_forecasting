import pandas as pd
from autots import AutoTS
from typing import Tuple
from autots.models.model_list import model_lists
import contextlib
import sys
import signal


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


#print(model_lists.keys())
#print(model_lists.values())
# all models in the package
# https://github.com/winedarksea/AutoTS/blob/6e473e89957c69446355d53118f65502840f4c94/autots/models/model_list.py#L404
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

# presets =  "superfast",  "fast",  "fast_parallel_no_arima", "multivariate", 'probabilistic', "best" "all"
 

def adjust_metric_weights(eval_metric):

    metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
    }
        
    # Alle Gewichte auf 0 setzen
    adjusted_weights = {key: 0 for key in metric_weighting}

    # Standardgewicht auf 1 setzen, falls eval_metric passt
    metric_key = eval_metric.lower() + '_weighting'
    
    if metric_key in adjusted_weights:
        adjusted_weights[metric_key] = 15
    else:
        # Wenn keine passende Metrik gefunden wurde, Standardgewichte verwenden
        adjusted_weights = metric_weighting

    return adjusted_weights

def train_autots_and_forecast(train_df, test_period, freq, date_col = "ds", id_col = "unique_id", actuals_col= "y", includeModels=None, excludeModels=None, set_index=True, enable_ensemble = True, eval_metric = "MAE", verbosity = 0, time_limit = None, random_seed = 123):
    """
    Trainiert ein AutoTS-Modell und erstellt Prognosen sowie fitted values für den Trainingsbereich.

    Parameters:
    ----------
    train_df : pd.DataFrame
        Trainingsdaten mit Spalten 'ds' (Datum), 'y' (Wert) und einer zusätzlichen ID-Spalte (z.B. 'unique_id').
    test_period : int
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

    # Umbenennen der Spalten für AutoTS
    train_df = train_df[[date_col, actuals_col, id_col]]
    train_df = train_df.rename(columns={date_col: 'ds', actuals_col: 'y', id_col: 'unique_id'})

    # Konvertiere das Datumsformat
    train_df['ds'] = pd.to_datetime(train_df['ds'])

    if time_limit is not None: 
        if time_limit <= 600: # 10 Minuten
            list_full = model_lists["superfast"]
        elif time_limit <= 1800: # 30 Minuten
            list_full = model_lists["fast"]
            fast_no_arima = {
                i: list_full[i]
                for i in list_full
                if i
                not in [
                    'NVAR',
                    "UnobservedComponents",
                    "VECM",
                    "MAR",
                    "BallTreeMultivariateMotif",  # might need sample_fraction tuning
                    "WindowRegression"  # same base shaping as BallTreeMM
                ]
            }
            list_full = fast_no_arima  # only include models that are not in the fast_no_arima list
        else:
            list_full = model_lists["all"]
    
    if includeModels is None or includeModels == "":
        if(excludeModels != None):
            model_list = [item for item in list_full if item not in exclude_model]
        else:
            model_list = list_full
    else:
        model_list = includeModels

    print(f"model_list: {model_list}")
    
    # Überprüfen, ob die Liste nur ein Element enthält
    if (isinstance(model_list, list) and len(model_list) == 1) or (enable_ensemble == False):
        ensemble = None
    else:
        ensemble = "simple"

    if freq in ["M", "ME"]:
        if freq == "M":
            freq = "ME"
        season_length = 12
    elif freq == "D":
        season_length = 7
    elif freq in ["Q", "QE", "QS"]:
        season_length = 4

    adjusted_weights = adjust_metric_weights(eval_metric)
    
    if time_limit is not None:
        generation_timeout = time_limit / 5 # seconds + max_generations
        generation_timeout = generation_timeout / 60 # minutes

    try:
        # Dein Modelltraining
        model = AutoTS(
            forecast_length=test_period,
            frequency=freq,
            ensemble=ensemble,
            max_generations=5,
            num_validations=2,
            model_list=model_list, #"superfast",  # 'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'
            n_jobs='auto',
            verbose=0,
            metric_weighting=adjusted_weights,
            random_seed = random_seed,
            generation_timeout = time_limit
        )

        if verbosity > 2:
            print(train_df.info())
        model = model.fit(train_df, date_col='ds', value_col='y', id_col="unique_id")

    except TimeoutError as e:
        print(e)

    finally:
        # Deaktiviere den Alarm, falls das Training vorher beendet wird
        signal.alarm(0)

    # Erstellen der Prognose
    prediction = model.predict(forecast_length = test_period, verbose=0)
        
    #forecast = prediction.forecast #.reset_index()
    forecast = prediction.long_form_results().reset_index()

    # Spalten zurück in die Originalnamen umbenennen
    Y_hat_df_AutoTS = forecast.rename(columns={'datetime': original_columns['date'], 'SeriesID': original_columns['ts_id'], 'Value': "pred"})

    # Extrahieren der Prognosen für den zukünftigen Zeitraum
    Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS[date_col] > train_df["ds"].max()]

    Y_hat_df_AutoTS = Y_hat_df_AutoTS[Y_hat_df_AutoTS["PredictionInterval"] == "50%"]
    Y_hat_df_AutoTS = Y_hat_df_AutoTS.drop('PredictionInterval', axis=1)

    best_model = [model.best_model_name]
    if verbosity > 2:
        print("Best Model")
        print(best_model)

    # Setze 'ts_id' als Index zurück, wenn `set_index` auf True gesetzt ist
    if set_index:
        Y_hat_df_AutoTS.set_index(original_columns['ts_id'], inplace=True)

    return Y_hat_df_AutoTS, best_model




