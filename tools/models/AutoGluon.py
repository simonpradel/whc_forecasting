# included standard models in the full default setting of autoGluon: https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/models/presets.py#L109

models = [
    "SimpleFeedForward",
    "DeepAR",
    "DLinear",
    "PatchTST",
    "TemporalFusionTransformer",
    "TiDE",
    "WaveNet",
    "RecursiveTabular",
    "DirectTabular",
    "Average",
    "SeasonalAverage",
    "Naive",
    "SeasonalNaive",
    "Zero",
    "AutoETS",
    "AutoCES",
    "AutoARIMA",
    "DynamicOptimizedTheta",
    "NPTS",
    "Theta",
    "ETS",
    "ADIDA",
    "CrostonSBA",
    "IMAPA",
    "Chronos",
]

light_inference = ["SeasonalNaive", "DirectTabular", "RecursiveTabular", "TemporalFusionTransformer", "PatchTST"]

# presets = "fast_training", "medium_quality", "high_quality", "best_quality"


# exclude models which cause problems
excludeModels = ['TemporalFusionTransformer', 'RecursiveTabular', 'DirectTabular', 'Chronos'] 
excludeModels = ["Naive", "SeasonalNaive"]


# AutoGluon
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

#random_seed (int or None, default = 123) – If provided, fixes the seed of the random number generator for all models. This guarantees reproducible results for most models (except those trained on GPU because of the non-determinism of GPU operations).

def train_autogluon_and_forecast(train_df, test_period, freq, date_col, id_col, actuals_col, includeModels=None, excludeModels=None, set_index = False, enable_ensemble = True, eval_metric = "MAE", verbosity = 0, time_limit = 60 * 60 * 24, random_seed = 123):
    """
    Trainiert ein AutoGluon-Modell und erstellt Prognosen.

    Parameters:
    train_df (pd.DataFrame): Trainingsdaten mit benutzerdefinierten Spalten für Datum, Zeitreihen-ID und Zielwert.
    test_period (int): Anzahl der Zukunftsperioden für die Prognose.
    freq (str): Frequenz der Daten, z.B. 'D' für täglich, 'W' für wöchentlich.
    date_col (str): Name der Spalte mit den Datumsinformationen.
    id_col (str): Name der Spalte mit den Zeitreihen-IDs.
    target_col (str): Name der Spalte mit den Zielwerten.
    includeModels (str or list of strings): e.g. "AutoARIMA", "AutoETSModel", "DeepAR", "chronos_mini", "medium_quality", "high_quality", "best_quality", "PatchTST"
    excludeModels (list of strings): Modelle, die von der Modellierung ausgeschlossen werden sollen.

    Returns:
    pd.DataFrame, pd.DataFrame: Fitted values und Prognosewerte mit den ursprünglichen Spaltennamen.
    """
    # Speichern der ursprünglichen Spaltennamen
    original_columns = { 'date': date_col, 'ts_id': id_col, 'target': actuals_col }
    
    # Prüfen, ob id_col eine Index-Spalte ist und ggf. zurücksetzen
    if id_col in train_df.index.names:
        train_df = train_df.reset_index(level=id_col)

    # Umbenennen der Spalten für AutoGluon
    train_df = train_df.rename(columns={date_col: 'timestamp', actuals_col: 'target', id_col: 'item_id'})

    # print("COLUMN TYPE AUTOGLUON")
    # print(train_df.dtypes)

    train_df['target'] = train_df['target'].astype(float)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

    # static features
    remaining_cols = train_df.drop(columns=['timestamp', 'target']).columns
    static_features_df = train_df[remaining_cols].drop_duplicates()
    train_df = train_df[['item_id', 'timestamp', 'target']]

    # Konvertieren in TimeSeriesDataFrame
    train_data_AutoGluon = TimeSeriesDataFrame.from_data_frame(train_df, id_column='item_id', timestamp_column='timestamp', static_features_df = static_features_df)

    if freq in ["M", "ME"]:
        if freq == "M":
            freq = "ME"
        season_length = 12
    elif freq == "D":
        season_length = 7
    elif freq in ["Q", "QE", "QS"]:
        season_length = 4


    # Initialisieren des Predictors
    predictor = TimeSeriesPredictor(
        prediction_length=test_period,
        freq=freq,
        target='target',
        eval_metric=eval_metric,
        verbosity=verbosity,
        log_to_file = False,
        quantile_levels=[0.05, 0.5, 0.95]
    )

    print(f"model_list: {includeModels}")

    # exclude models
    if excludeModels is None:
        excludeModels = []
    if isinstance(excludeModels, str):
        excludeModels = [excludeModels]
    
    # include includeModels
    if includeModels is None or includeModels == "":
        print("start fitting")
        predictor.fit(
            train_data_AutoGluon, 
            presets="high_quality", 
            excluded_model_types=excludeModels,
            enable_ensemble = enable_ensemble,
            time_limit = time_limit,
            random_seed = random_seed
        )
    elif includeModels in ["fast_training", "medium_quality", "good_quality", "high_quality", "best_quality"]:
        predictor.fit(
            train_data_AutoGluon, 
            presets=includeModels, 
            excluded_model_types=excludeModels,
            enable_ensemble = enable_ensemble
        )
    else:
        if isinstance(includeModels, str):
            includeModels = [includeModels]
        model_dict = {model: {} for model in includeModels}

        predictor.fit(
            train_data_AutoGluon, 
            presets="best_quality",
            hyperparameters=model_dict,
            excluded_model_types=excludeModels,
            enable_ensemble = enable_ensemble
        )

    #print("MODEL FIT DONE; NOW PREDICTION")
    # Erstellen der Prognose
    forecast = predictor.predict(train_data_AutoGluon)

    # Bestes Modell herausfinden
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard['model'][0]

    # Timestamp in eine normale Spalte konvertieren und umbenennen
    forecast = forecast.reset_index().rename(columns={'timestamp': original_columns['date'], 'item_id': original_columns['ts_id'], 'mean': 'pred'})

    # Setze 'ts_id' als Index zurück
    if ( set_index == True):
        forecast.set_index(original_columns['ts_id'], inplace=True)
        Y_hat_df = forecast[[original_columns['date'], 'pred']]
    else:
        forecast.reset_index(inplace=True)
        Y_hat_df = forecast[[original_columns['date'], original_columns['ts_id'], 'pred']]


    return Y_hat_df, best_model 


