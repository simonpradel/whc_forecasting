import pandas as pd
import matplotlib.pyplot as plt

def plot_time_series(df, ts_ids=None, exclude_ts_ids=None, show_column_values=False, max_legend_entries=None, top_n_time_series=None, top_n_type='positive', show_aggregated_series=False, show_individual_plots=False):
    """
    Plots the time series for a list of ts_ids and optionally displays column values in the legend. Optionally,
    displays only the top N time series based on their total sum. Can also display a vertical line at January of each year.
    Additionally, can plot the aggregated series of the selected time series. It can also plot each time series in a separate plot.

    Parameters:
    ----------
    - df (pd.DataFrame): The DataFrame containing the time series data.
    - ts_ids (list or None): A list of ts_ids of the time series to plot and analyze. If None, plots all groups.
    - exclude_ts_ids (list or None): A list of ts_ids to exclude from the plot. If None, no ts_ids are excluded.
    - show_column_values (bool): Whether to display column values in the legend instead of ts_id. Default is False.
    - max_legend_entries (int or None): Maximum number of entries to show in the legend. If None, show all entries.
    - top_n_time_series (int or None): Number of top time series to display based on their total sum. If None, show all time series.
    - top_n_type (str): Type of top time series to display: 'positive', 'negative', or 'absolute'. Default is 'positive'.
    - show_aggregated_series (bool): Whether to show the aggregated series of the selected time series. Default is False.
    - show_individual_plots (bool): Whether to plot each time series in a separate plot. Default is False.

    Returns:
    -------
    - pd.DataFrame: DataFrame with information about the selected time series.
    """
    
    ts_info_columns = ['ts_id', 'total_sum', 'percentage_of_total']
    ts_info = pd.DataFrame(columns=ts_info_columns)

    # Convert 'date' column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Sort by 'date' if not already sorted
    if not df['date'].is_monotonic_increasing:
        df = df.sort_values(by='date').reset_index(drop=True)

    # If ts_ids is None, select all existing ts_ids
    if ts_ids is None:
        ts_ids = df['ts_id'].unique().tolist()

    # Exclude specified ts_ids
    if exclude_ts_ids is not None:
        ts_ids = [ts_id for ts_id in ts_ids if ts_id not in exclude_ts_ids]

    # Aggregate total sum for each ts_id based on the selected top_n_type
    if top_n_type == 'positive':
        ts_sum = df.groupby('ts_id')['total'].sum().sort_values(ascending=False)
    elif top_n_type == 'negative':
        ts_sum = df.groupby('ts_id')['total'].sum().sort_values()
    elif top_n_type == 'absolute':
        ts_sum = df.groupby('ts_id')['total'].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
    else:
        raise ValueError("Invalid value for top_n_type. Choose from 'positive', 'negative', or 'absolute'.")

    total_sum_all_abs = df['total'].abs().sum()  # Calculate the sum of absolute values for percentage calculation

    # Select top N time series based on total sum
    if top_n_time_series is not None:
        top_ts_ids = ts_sum.head(top_n_time_series).index.tolist()
        ts_ids = [ts_id for ts_id in ts_ids if ts_id in top_ts_ids]

    # Initialize list to hold individual DataFrames for concatenation
    ts_info_list = []

    # Prepare DataFrame to store information about ts_ids
    for ts_id in ts_ids:
        ts_data = df[df['ts_id'] == ts_id]
        actual_total_sum = df[df['ts_id'] == ts_id]['total'].sum()  # Calculate the actual total sum for display
        percentage_of_total = round((abs(actual_total_sum) / total_sum_all_abs) * 100, 2)

        ts_info_row = {
            'ts_id': ts_id,
            'total_sum': actual_total_sum,  # Use the actual total sum
            'percentage_of_total': percentage_of_total
        }

        # Add individual columns to ts_info_row
        for column in ts_data.columns:
            if column not in ['date', 'total', 'ts_id']:
                if column not in ts_info_columns:
                    ts_info_columns.append(column)
                ts_info_row[column] = ts_data[column].iloc[0]

        ts_info_list.append(pd.DataFrame(ts_info_row, index=[0]))

    # Concatenate all individual DataFrames into the final ts_info DataFrame
    ts_info = pd.concat(ts_info_list, ignore_index=True)

    # Sort ts_info by the total sum of each time series
    ts_info = ts_info.sort_values(by='total_sum', ascending=False).reset_index(drop=True)

    # Plot aggregated series or individual time series
    if not show_individual_plots:
        plt.figure(figsize=(12, 6))
        handles_labels = []
        if show_aggregated_series:
            aggregated_series = df[df['ts_id'].isin(ts_ids)].groupby('date')['total'].sum()
            plt.plot(aggregated_series.index, aggregated_series.values, label='Aggregated Series', linewidth=2, linestyle='--')
            handles_labels.append((plt.Line2D([], [], color='black', linestyle='--', linewidth=2), 'Aggregated Series', 'aggregated_series'))
        else:
            for ts_id in ts_ids:
                ts_data = df[df['ts_id'] == ts_id]
                if show_column_values:
                    legend_label = f"TS {ts_id}: {', '.join([f'{col}: {ts_data[col].iloc[0]}' for col in ts_info_columns if col not in ['date', 'total', 'ts_id', 'total_sum', 'percentage_of_total']])}"
                else:
                    legend_label = f'TS {ts_id}'
                line, = plt.plot(ts_data['date'], ts_data['total'], label=legend_label)
                handles_labels.append((line, legend_label, ts_id))

        # Sort handles_labels by the total sum of each time series
        handles_labels.sort(key=lambda hl: ts_sum.get(hl[2], float('inf')), reverse=True)

        # Determine the number of legend entries to show based on max_legend_entries
        if max_legend_entries is not None:
            handles_labels = handles_labels[:max_legend_entries]

        handles, labels, _ = zip(*handles_labels) if handles_labels else ([], [], [])

        plt.xlabel('Date')
        plt.ylabel('Total')
        plt.title(f'Time Series for Selected TS')
        plt.grid(True)  # Add grid to the plot

        # Show legend only if multiple groups are plotted and within max_legend_entries limit
        if len(ts_ids) > 1 or show_aggregated_series:
            plt.legend(handles, labels)

        plt.show()
    else:
        # Sort ts_ids by the total sum of each time series
        sorted_ts_ids = ts_info['ts_id'].tolist()
        for ts_id in sorted_ts_ids:
            ts_data = df[df['ts_id'] == ts_id]
            unique_values = ', '.join([f"{col}: {ts_data[col].iloc[0]}" for col in ts_info_columns if col not in ['date', 'total', 'ts_id', 'total_sum', 'percentage_of_total']])
            plt.figure(figsize=(12, 6))
            plt.plot(ts_data['date'], ts_data['total'], label=f'TS {ts_id}')
            if show_column_values:
                legend_label = f"TS {ts_id}: {', '.join([f'{col}: {ts_data[col].iloc[0]}' for col in ts_info_columns if col not in ['date', 'total', 'ts_id', 'total_sum', 'percentage_of_total']])}"
                plt.legend([legend_label])
            plt.xlabel('Date')
            plt.ylabel('Total')
            plt.title(f'TS {ts_id} - {unique_values}')
            plt.grid(True)  # Add grid to the plot
            plt.show()

    return ts_info



import pandas as pd
import matplotlib.pyplot as plt

def plot_aggregated_time_series(df, group_by_columns=None, max_legend_entries=None, top_n_groups=None, top_n_type='positive', show_aggregated_series=False, show_individual_plots=False):
    """
    Plots aggregated time series based on specified columns.

    Parameters:
    df (pd.DataFrame): A Pandas DataFrame containing the time series data.
    group_by_columns (list or None): A list of columns to group by. If None, aggregates the entire DataFrame.
    max_legend_entries (int or None): Maximum number of entries to show in the legend. If None, show all entries.
    top_n_groups (int or None): Number of top groups to display based on their total sum. If None, show all groups.
    top_n_type (str): Type of top groups to display: 'positive', 'negative', or 'absolute'. Default is 'positive'.
    show_aggregated_series (bool): Whether to show the aggregated series of the selected time series. Default is False.
    show_individual_plots (bool): Whether to plot each time series in a separate plot. Default is False.

    Returns:
    pd.DataFrame: DataFrame with information about the groups.
    """
    
    # Convert 'date' column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Sort by 'date' if not already sorted
    if not df['date'].is_monotonic_increasing:
        df = df.sort_values(by='date').reset_index(drop=True)

    # Group by the specified columns and aggregate
    if group_by_columns is None:
        group_by_columns = []
    grouped = df.groupby(['date'] + group_by_columns)['total'].sum().reset_index()

    # Calculate total sums for each group
    if top_n_type == 'positive':
        group_totals = grouped.groupby(group_by_columns)['total'].sum().sort_values(ascending=False).reset_index()
    elif top_n_type == 'negative':
        group_totals = grouped.groupby(group_by_columns)['total'].sum().sort_values(ascending=True).reset_index()
    elif top_n_type == 'absolute':
        group_totals = grouped.groupby(group_by_columns)['total'].apply(lambda x: x.abs().sum()).sort_values(ascending=False).reset_index()
        # Ensure the correct sign is shown in the table for the 'absolute' case
        actual_totals = grouped.groupby(group_by_columns)['total'].sum().reset_index()
        group_totals = group_totals.merge(actual_totals, on=group_by_columns, suffixes=('_abs', ''))
    else:
        raise ValueError("Invalid value for top_n_type. Choose from 'positive', 'negative', or 'absolute'.")

    total_abs_sum = group_totals['total_abs' if top_n_type == 'absolute' else 'total'].abs().sum()
    group_totals['percentage_of_total'] = (group_totals['total_abs' if top_n_type == 'absolute' else 'total'].abs() / total_abs_sum) * 100
    group_totals = group_totals.sort_values(by='total_abs' if top_n_type == 'absolute' else 'total', ascending=(top_n_type == 'negative')).reset_index(drop=True)

    # Select top N groups based on total sum
    if top_n_groups is not None:
        top_groups = group_totals.head(top_n_groups)[group_by_columns]
        grouped = grouped[grouped[group_by_columns].apply(tuple, axis=1).isin(top_groups.apply(tuple, axis=1))]
        group_totals = group_totals.head(top_n_groups)

    if not show_individual_plots:
        plt.figure(figsize=(12, 6))
        handles_labels = []
        if show_aggregated_series:
            aggregated_series = grouped.groupby('date')['total'].sum()
            plt.plot(aggregated_series.index, aggregated_series.values, label='Aggregated Series', linewidth=2, linestyle='--')
            handles_labels.append((plt.Line2D([], [], color='black', linestyle='--', linewidth=2), 'Aggregated Series', 'aggregated_series'))
        else:
            for name, group in grouped.groupby(group_by_columns):
                line, = plt.plot(group['date'], group['total'], label=str(name))
                handles_labels.append((line, str(name), name))

        # Sort handles_labels by the total sum of each group
        handles_labels.sort(key=lambda hl: group_totals.set_index(group_by_columns).loc[hl[2], 'total_abs' if top_n_type == 'absolute' else 'total'], reverse=(top_n_type != 'negative'))

        # Determine the number of legend entries to show based on max_legend_entries
        if max_legend_entries is not None:
            handles_labels = handles_labels[:max_legend_entries]

        handles, labels, _ = zip(*handles_labels) if handles_labels else ([], [], [])

        plt.xlabel('Date')
        plt.ylabel('Total')
        plt.title(f'Aggregated Time Series grouped by {group_by_columns}')
        plt.grid(True)
        plt.legend(handles, labels)
        plt.show()
    else:
        for name, group in grouped.groupby(group_by_columns):
            plt.figure(figsize=(12, 6))
            plt.plot(group['date'], group['total'], label=str(name))
            plt.xlabel('Date')
            plt.ylabel('Total')
            plt.title(f'Aggregated Time Series - {str(name)}')
            plt.legend()
            plt.grid(True)
            plt.show()

    return group_totals

