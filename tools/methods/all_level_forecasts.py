def create_all_level_forecast(lower_level_series, top_level_series, future_periods, freq, train_end_date=None, model = "Prophet"):
    forecast_dict = {}

    for group_key, group_df in lower_level_series.items():
        if train_end_date is not None:
            # Filtere Daten bis zum Trainingsenddatum
            truncated_df = group_df[group_df['date'] <= train_end_date]
            full_df = group_df
        else:
            truncated_df = full_df = group_df

        ts_ids = truncated_df['ts_id'].unique()
        group_forecasts = []

        for ts_id in ts_ids:
            series = truncated_df[truncated_df['ts_id'] == ts_id]

            if (model == "Prophet"):
                forecast = train_prophet_and_forecast(series, future_periods, freq=freq)
            elif(model == "ETS"):
                forecast = train_autogluon_and_forecast(series, future_periods, freq=freq, model = "AutoETS")

            # Bestimme das letzte Datum der aktuellen Zeitreihe
            last_date = series['date'].max() if train_end_date is None else pd.to_datetime(train_end_date)

            # Erstelle DataFrame mit zukünftigen Prognosewerten
            if freq == "D":
                future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=future_periods, freq=freq)
            elif freq == "M":
                # Berechne den letzten Tag des aktuellen Monats
                last_day_current_month = last_date + MonthEnd(0)
                # Starte beim letzten Tag des nächsten Monats
                future_dates = pd.date_range(start=last_day_current_month + MonthEnd(1), periods=future_periods, freq='M')

            forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast})

            # Kombiniere den aktuellen series DataFrame mit dem forecast_df
            result_df = pd.merge(full_df[full_df['ts_id'] == ts_id], forecast_df, on='date', how='outer')

            # Fülle die fehlenden Werte für die Variablen in den keys/tupels auf
            for col in group_df.columns:
                if col not in ['date', 'total', 'forecast']:
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')

            result_df['ts_id'] = ts_id  # Sicherstellen, dass ts_id in allen Zeilen korrekt gesetzt ist
            group_forecasts.append(result_df)

        # Füge alle DataFrames der aktuellen Gruppe zusammen
        group_forecast_df = pd.concat(group_forecasts, ignore_index=True)
        forecast_dict[group_key] = group_forecast_df

    return forecast_dict
