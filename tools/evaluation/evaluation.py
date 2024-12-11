import pandas as pd

def evaluate_forecast_performance(dfs, metric='MAPE', date_col='ds', actual_col='y', modelName="AutoGluon", saveResults=True, exclude_cols=None):
    """Evaluates the forecast performance of different models using the specified metric.
    
    Args:
        dfs (dict or pd.DataFrame): Dictionary of DataFrames or a single DataFrame.
        metric (str): Metric to evaluate ('MAPE', 'RMSE', etc.).
        date_col (str): Name of the column containing the date/time information.
        actual_col (str): Name of the column containing the actual values.
        modelName (str): Name of the model (used as prefix for saved results).
        saveResults (bool): Whether to save the results to a CSV file.
        exclude_cols (list of str): List of column names to exclude from metric calculation.
    
    Returns:
        pd.DataFrame: DataFrame containing the metric results for each model.
    """
    
    # Wenn dfs kein Dictionary ist, wandle es in ein Dictionary um
    if isinstance(dfs, pd.DataFrame):
        dfs = {modelName: dfs}
    
    # Wenn keine Spalten zum Ausschließen angegeben sind, initialisiere eine leere Liste
    if exclude_cols is None:
        exclude_cols = []
    
    # Initialisiere eine Liste, um die Ergebnisse der Metrik für jeden DataFrame zu speichern
    metric_results = []
    
    # Iteriere über das Dictionary von DataFrames
    for model_name, df in dfs.items():
        # Entferne alle Zeilen mit NA in den relevanten Prognose-Spalten
        forecast_cols = [col for col in df.columns if col not in [date_col, actual_col] and pd.api.types.is_float_dtype(df[col])]
        forecast_cols = [col for col in forecast_cols if col not in exclude_cols]
        df_clean = df.dropna(subset=forecast_cols).copy()
        
        # Initialisiere ein Dictionary, um die Metrik-Werte zu speichern
        metric_dict = {'Model': model_name}
        
        # Iteriere über die Float-Spalten (die Prognosen repräsentieren)
        for column in forecast_cols:
            # Berechne die Metrik für die aktuelle Spalte
            metric_dict[column] = calculate_metric(df_clean, column, metric, actual_col)
        
        # Füge die Metrik-Daten des aktuellen DataFrames zur Ergebnisliste hinzu
        metric_results.append(metric_dict)
    
    # Erstelle einen DataFrame mit den Metrik-Werten
    metric_df = pd.DataFrame(metric_results)
    
    # Speichere das Ergebnis als CSV, falls saveResults aktiviert ist
    if saveResults:
        metric_df_name = f"{modelName}_{metric}_metric.csv"
        metric_df.to_csv(metric_df_name, index=False)
        print(f"Metric '{metric_df_name}' als CSV gespeichert.")
    
    return metric_df

def calculate_metric(df, forecast_col, metric, actual_col):
    """Berechnet die spezifizierte Metrik. Beispiel: MAPE."""
    if metric == 'MAPE':
        return ((df[actual_col].astype(float) - df[forecast_col].astype(float)).abs() / df[actual_col].astype(float)).median() * 100
    else:
        raise ValueError(f"Unsupported metric: {metric}")





import pandas as pd

def evaluate_forecast_performance3(df):
    """
    Evaluates the performance of the forecast in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'date', 'total', and 'forecast'
    
    Returns:
    float: Mean Absolute Percentage Error (MAPE)
    pd.DataFrame: DataFrame containing the true deviation and percentage deviation for each date
    """
    # Entferne alle Zeilen mit NA im 'forecast'
    df_clean = df.dropna(subset=['forecast']).copy()
    
    # Berechne die wahre Abweichung
    df_clean['true_deviation'] = df_clean['total'] - df_clean['forecast']
    
    # Berechne die absolute Abweichung
    df_clean['absolute_deviation'] = abs(df_clean['true_deviation'])
    
    # Berechne die prozentuale Abweichung
    df_clean['percentage_deviation'] = (df_clean['absolute_deviation'] / df_clean['total']) * 100
    
    # Berechne den MAPE
    mape = df_clean['percentage_deviation'].mean()

    
    # Formatierung der Zahlen mit Punkten als Tausendertrennzeichen
    df_clean['total'] = df_clean['total'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['forecast'] = df_clean['forecast'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['true_deviation'] = df_clean['true_deviation'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    df_clean['percentage_deviation'] = df_clean['percentage_deviation'].apply(lambda x: f"{x:,.2f}".replace(',', '.'))

    return mape, df_clean[['date', 'total', 'forecast', 'true_deviation', 'percentage_deviation']]




# Beispielaufruf:
# Angenommen, df ist ein DataFrame mit den Spalten 'date', 'total' und 'forecast'
# mape, evaluation_df = evaluate_forecast_performance(df)

def evaluate_forecast_performance2(df):
    """
    Evaluates the performance of the forecast in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'date', 'total', 'forecast', and optionally 'ts_id'
    
    Returns:
    dict: Dictionary where keys are ts_id (or 'overall' if no ts_id is present) and values are tuples containing:
          - float: Mean Absolute Percentage Error (MAPE)
          - pd.DataFrame: DataFrame containing the true deviation and percentage deviation for each date
    """
    def calculate_metrics(sub_df):
        # Entferne alle Zeilen mit NA im 'forecast'
        sub_df_clean = sub_df.dropna(subset=['forecast']).copy()
        
        # Berechne die wahre Abweichung
        sub_df_clean['true_deviation'] = sub_df_clean['total'] - sub_df_clean['forecast']
        
        # Berechne die absolute Abweichung
        sub_df_clean['absolute_deviation'] = abs(sub_df_clean['true_deviation'])
        
        # Berechne die prozentuale Abweichung
        sub_df_clean['percentage_deviation'] = (sub_df_clean['absolute_deviation'] / sub_df_clean['total']) * 100
        
        # Berechne den MAPE
        mape = sub_df_clean['percentage_deviation'].mean()
        
        # Formatierung der Zahlen mit Punkten als Tausendertrennzeichen
        sub_df_clean['total'] = sub_df_clean['total'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
        sub_df_clean['forecast'] = sub_df_clean['forecast'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
        sub_df_clean['true_deviation'] = sub_df_clean['true_deviation'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
        sub_df_clean['percentage_deviation'] = sub_df_clean['percentage_deviation'].apply(lambda x: f"{x:,.2f}".replace(',', '.'))

        return mape, sub_df_clean[['date', 'total', 'forecast', 'true_deviation', 'percentage_deviation']]
    
    results = {}
    
    if 'ts_id' in df.columns:
        grouped = df.groupby('ts_id')
        for ts_id, group in grouped:
            mape, evaluation_df = calculate_metrics(group)
            results[ts_id] = (mape, evaluation_df)
    else:
        mape, evaluation_df = calculate_metrics(df)
        results['overall'] = (mape, evaluation_df)
    
    return results