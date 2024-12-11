import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_forecasts(final_forecast_df, plot_individual_folds=True, 
                           forecast_col='forecast', actual_col='actual', date_col='date'):
    """
    Plottet die rollierenden Prognosen gegen die tatsächlichen Werte.

    Parameters:
    ----------
    final_forecast_df : pd.DataFrame
        DataFrame mit den Forecasts für jeden Fold und einer Spalte 'fold'.
    plot_individual_folds : bool, optional
        Wenn True, werden die Ergebnisse jedes Folds in einem eigenen Plot dargestellt.
        Wenn False, werden alle Folds zusammen in einem Plot dargestellt.
    forecast_col : str, optional
        Name der Spalte, die die Vorhersagewerte enthält.
    actual_col : str, optional
        Name der Spalte, die die tatsächlichen Werte enthält.
    date_col : str, optional
        Name der Spalte, die die Datumswerte enthält.
    """
    # Konvertiere die 'date'-Spalte in datetime
    final_forecast_df[date_col] = pd.to_datetime(final_forecast_df[date_col])

    unique_folds = final_forecast_df['fold'].unique()
    plt.figure(figsize=(15, len(unique_folds) * 5))

    if plot_individual_folds:
        # Plotte jeden Fold separat

        for i, fold in enumerate(unique_folds):
            fold_df = final_forecast_df[final_forecast_df['fold'] == fold]

            # Filtere Folds, die keine NaN-Werte in der Vorhersage-Spalte enthalten
            if fold_df[forecast_col].isna().all():
                continue

            start_date = fold_df[date_col].min()
            end_date = fold_df[date_col].max()

            plt.subplot(len(unique_folds), 1, i + 1)
            
            # Plotte die Vorhersagen über den tatsächlichen Werten
            plt.plot(final_forecast_df[date_col], final_forecast_df[actual_col], label='Actuals', color='lightblue')
            plt.plot(fold_df[date_col], fold_df[forecast_col], label='Forecast', color='red')
 
            plt.axvline(x=start_date, color='gray', linestyle='--', label='Forecast Start')
            plt.axvline(x=end_date, color='gray', linestyle='--', label='Forecast End')
            plt.fill_between(fold_df[date_col], fold_df[actual_col].min(), fold_df[actual_col].max(), color='gray', alpha=0.2)

            plt.title(f"Fold {fold}: Forecast vs Actuals")
            plt.xlabel('Date')
            plt.grid(True)
            plt.ylabel('Values')
            plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Finde das Start- und Enddatum über alle Folds hinweg
        overall_start_date = final_forecast_df[date_col].min()
        overall_end_date = final_forecast_df[date_col].max()

        # Plotte alle Folds zusammen
        plt.figure(figsize=(15, 5))

        # Plotte die tatsächlichen Werte zuletzt
        plt.plot(final_forecast_df[date_col], final_forecast_df[actual_col], label='Actuals', color='lightblue')

        # Filtere Folds ohne NaN-Werte
        for fold in unique_folds:
            fold_df = final_forecast_df[final_forecast_df['fold'] == fold]

            if fold_df[forecast_col].isna().all():
                continue

            # Plotte die Vorhersagen über den tatsächlichen Werten
            plt.plot(fold_df[date_col], fold_df[forecast_col], label=f'Forecast Fold {fold}', linestyle='solid')

        # Setze die x-Achse auf das gefundene Intervall
        plt.xlim(overall_start_date, overall_end_date)

        plt.title("Forecast vs Actuals (all Folds)")
        plt.xlabel('Date')
        plt.grid(True)
        plt.ylabel('Values')
        plt.legend()

        plt.tight_layout()
        plt.show()