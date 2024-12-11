import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_forecasts(results):
    """
    Plottet die rollierenden Prognosen gegen die tatsächlichen Werte.
    
    Parameters:
    results (dict): Dictionary mit den rollierenden Prognosen für den Testbereich.
    """
    # Gesamter Datensatz aus den Ergebnissen extrahieren
    first_key = list(results.keys())[0]
    total_df = results[first_key][['date', 'total']].copy()
    
    # Konvertiere die 'date'-Spalte in datetime
    total_df['date'] = pd.to_datetime(total_df['date'])
    
    # Setze die Größe der Plots
    plt.figure(figsize=(15, len(results) * 5))
    
    # Plot für jede Vorhersageperiode
    for i, (period, forecast_df) in enumerate(results.items()):
        start_date, end_date = period.split(' - ')
        
        # Konvertiere die 'date'-Spalte im forecast_df in datetime
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Filtere die tatsächlichen Werte für den aktuellen Zeitraum
        actuals = total_df[(total_df['date'] >= start_date) & (total_df['date'] <= end_date)]
        
        # Erstelle den Plot
        plt.subplot(len(results), 1, i + 1)
        plt.plot(total_df['date'], total_df['total'], label='Actuals', color='blue')
        
        # Nur den Bereich plotten, in dem `forecast` nicht `NULL` ist
        forecast_period = forecast_df.dropna(subset=['forecast'])
        plt.plot(forecast_period['date'], forecast_period['forecast'], label='Forecast', color='red')
        
        # Markiere den Beginn und das Ende des Vorhersagezeitraums
        plt.axvline(x=forecast_period['date'].iloc[0], color='gray', linestyle='--', label='Forecast Start')
        plt.axvline(x=forecast_period['date'].iloc[-1], color='gray', linestyle='--', label='Forecast End')
        
        # Fülle den Vorhersagezeitraum mit grauem Hintergrund
        plt.fill_between(forecast_period['date'], total_df['total'].min(), total_df['total'].max(), color='gray', alpha=0.2)
        
        plt.title(f"Forecast vs Actuals: {start_date} to {end_date}")
        plt.xlabel('Date')
        plt.grid(True)
        plt.ylabel('Values')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_forecasts(results):
    """
    Plottet die rollierenden Prognosen gegen die tatsächlichen Werte.
    
    Parameters:
    results (dict): Dictionary mit den rollierenden Prognosen für den Testbereich.
    """
    # Plotten der tatsächlichen Werte und der gewichteten Prognosen
    plt.figure(figsize=(14, 7))
    plt.plot(results['date'], results['total'], label='Tatsächliche Werte', color='blue')
    plt.plot(results['date'], results['forecast'], label='Gewichtete Prognosen', color='orange')
    plt.xlabel('Datum')
    plt.ylabel('Wert')
    plt.title('Tatsächliche Werte vs. Gewichtete Prognosen')
    plt.legend()
    plt.grid(True)
    plt.show()



import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast2(results_dict_or_df, aggregate_cols=None):
    """
    Plots the actual and predicted parts of the time series forecasts for each unique combination of aggregate columns.

    Parameters:
    results_dict_or_df (dict or DataFrame): Dictionary of DataFrames containing forecasted values or a single DataFrame.
    aggregate_cols (tuple or list, optional): Columns to group by for plotting (e.g., ('segment', 'brand')).

    Returns:
    None
    """
    if isinstance(results_dict_or_df, dict):
        if aggregate_cols:
            try:
                df = results_dict_or_df[aggregate_cols]
            except KeyError:
                print(f"No data found for {aggregate_cols}. Check your results_dict keys.")
                return
        else:
            # If no aggregate_cols are provided, concatenate all dataframes in the dictionary
            df = pd.concat(results_dict_or_df.values(), ignore_index=True)
    elif isinstance(results_dict_or_df, pd.DataFrame):
        df = results_dict_or_df
    else:
        raise ValueError("Input should be either a dictionary of DataFrames or a single DataFrame.")

    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Determine the grouping columns
    grouping_cols = list(aggregate_cols) if aggregate_cols else ['ts_id']

    # Group by grouping_cols and iterate over each group to plot
    grouped = df.groupby(grouping_cols)

    for name, group in grouped:
        # Separate actual and predicted parts
        actual = group[group['total'].notna()]
        predicted = group[group['forecast'].notna()]

        # Plotting actual and predicted parts
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(actual['date'], actual['total'], marker='o', linestyle='-', color='b', label='Actual')
        ax.plot(predicted['date'], predicted['forecast'], marker='o', linestyle='--', color='r', label='Predicted')

        ax.set_ylabel('Total')
        ax.set_title(f"Forecast Plot for {' - '.join(name) if isinstance(name, tuple) else name}")
        ax.legend()

        plt.xlabel('Date')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()





def plot_forecast_dic(results_dict, aggregate_cols=None):
    """
    Plots the actual and predicted parts of the time series forecasts for each unique combination of aggregate columns.

    Parameters:
    results_dict (dict or DataFrame): Dictionary of DataFrames containing forecasted values or a single DataFrame.
    aggregate_cols (tuple or list, optional): Columns to group by for plotting (e.g., ('segment', 'brand')).

    Returns:
    None
    """
    if isinstance(results_dict, dict):
        if aggregate_cols:
            try:
                df = results_dict[aggregate_cols]
            except KeyError:
                print(f"No data found for {aggregate_cols}. Check your results_dict keys.")
                return
        else:
            # If no aggregate_cols are provided, concatenate all dataframes in the dictionary
            df = pd.concat(results_dict.values(), ignore_index=True)
    elif isinstance(results_dict, pd.DataFrame):
        df = results_dict
    else:
        raise ValueError("Input should be either a dictionary of DataFrames or a single DataFrame.")

    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Determine the grouping columns
    grouping_cols = list(aggregate_cols) if aggregate_cols else ['ts_id']

    # Group by grouping_cols and iterate over each group to plot
    grouped = df.groupby(grouping_cols)

    for name, group in grouped:
        # Separate actual and predicted parts
        actual = group[group['total'].notna()]
        predicted = group[group['forecast'].notna()]

        # Plotting actual and predicted parts
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(actual['date'], actual['total'], marker='o', linestyle='-', color='b', label='Actual')
        ax.plot(predicted['date'], predicted['forecast'], marker='o', linestyle='--', color='r', label='Predicted')

        ax.set_ylabel('Total')
        ax.set_title(f"Forecast Plot for {' - '.join(name) if isinstance(name, tuple) else name}")
        ax.legend()

        plt.xlabel('Date')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

