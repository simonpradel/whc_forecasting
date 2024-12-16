import pandas as pd

def calc_different_metrics_across_datasets(
    df_list,
    model=None,
    optim_method=None,
    method=None,
    metric=None,
    forecast_model=None,
    forecast_method=None,
    columns_to_remove=None,
    remove_constant_columns=False,
    grouping_variable="forecast_method",
    time_limit=None,
    add_to_row_values=None,
    show_control_table=False
):
    """
    Aggregates the ranks and metrics of models across multiple datasets based on a common grouping variable.
    
    Parameters:
    - df_list (list of DataFrames): List of DataFrames containing model metrics for aggregation.
    - model (str, optional): Filter to include results for a specific model only.
    - optim_method (str, optional): Filter to include results for a specific optimization method only.
    - method (str, optional): Filter to include results for a specific method only.
    - metric (list of str): List of metrics to aggregate.
    - forecast_model (str, optional): Filter to include results for a specific forecast model.
    - forecast_method (str, optional): Filter to include results for a specific forecast method.
    - columns_to_remove (list of str, optional): Keywords to identify columns for removal.
    - remove_constant_columns (bool, optional): Whether to remove columns with constant values.
    - grouping_variable (str, optional): Variable for grouping data when aggregating.
    - time_limit (int, optional): Filter to include results within a specific time limit.
    - add_to_row_values (str, optional): Column to modify row values for specific methods.
    - show_control_table (bool, optional): Whether to display a control table for debugging.
    
    Returns:
    - DataFrame: Aggregated metrics and ranks for the specified grouping variable.
    """
    results = []

    for df in df_list:
        # Duplicate specific rows if forecast_model column exists
        if "forecast_model" in df.columns:
            df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
            df_to_duplicate["forecast_method"] = 'level'
            df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Apply filters
        if model:
            df = df[df['Model'] == model]
        if optim_method:
            df = df[df['optim_method'] == optim_method]
        if method:
            df = df[df['method'] == method]
        if forecast_model:
            df = df[df['forecast_model'] == forecast_model]
        if forecast_method:
            df = df[df['forecast_method'] == forecast_method]
        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]

        df = df.drop(columns=['Model'], errors='ignore')

        # Modify row values if required
        if add_to_row_values and add_to_row_values in df.columns:
            df["methodNew"] = df['method']
            df.loc[df['method'] == 'weighted_pred', 'methodNew'] = df['method'] + '_' + df['optim_method']

        # Collect results for each metric
        for m in metric:
            metric_df = df[df['metric'] == m].copy()
            results.append(metric_df)

    # Combine all results
    all_results = pd.concat(results, ignore_index=True)

    # Remove irrelevant combinations
    condition = (
        ((all_results['method'] != 'weighted_pred')) &
        (all_results['optim_method'] != 'ensemble_selection')
    )
    all_results = all_results[~condition].reset_index(drop=True)

    # Clean and rename columns
    if "method" in all_results.columns and "methodNew" in all_results.columns:
        all_results.drop("method", axis=1, inplace=True)
        all_results.rename(columns={'methodNew': 'method'}, inplace=True)

    # Grouping based on the specified variable
    grouping_values = all_results[grouping_variable].unique()

    # Replace method names with descriptive labels
    method_mapping = {
        "base": "Base Forecast",
        "BottomUp": "Bottom-up",
        "MinTrace_method-ols": "Ordinary Least Squares",
        "MinTrace_method-wls_struct": "Structural Scaling",
        "equal_weights_pred": "WHC (equal weights)",
        "weighted_pred_differential_evolution": "WHC (differential evolution)",
        "weighted_pred_ensemble_selection": "WHC (ensemble selection)",
        "weighted_pred_optimize_nnls": "WHC (nnls)"
    }
    all_results["method"] = all_results["method"].replace(method_mapping)

    method_order = list(method_mapping.values())
    all_results["method"] = pd.Categorical(all_results["method"], categories=method_order, ordered=True)
    all_results = all_results.sort_values("method").reset_index(drop=True)

    pivoted_dfs = []

    for group_value in grouping_values:
        temp_df = all_results[all_results[grouping_variable] == group_value].copy()

        # Define columns to group by
        group_cols = [
            col for col in temp_df.columns
            if col not in ['value', 'elapsed_time', "dataset_type", "time_limit", 'dataset', add_to_row_values]
        ]

        # Display control table for debugging if enabled
        if show_control_table:
            print("The entries in the table must match the number of datasets.")
            control_table = temp_df.groupby(group_cols).agg(avg_value=('value', 'count')).reset_index()
            print(control_table)

        # Aggregate metrics
        aggregated = temp_df.groupby(group_cols).agg(avg_value=('value', 'mean')).reset_index()

        # Pivot table for metrics
        aggregated = aggregated.pivot(index="method", columns='metric', values='avg_value').reset_index()

        # Rename columns for clarity
        aggregated.columns = [
            f"{col}_{group_value}" if col not in ['method'] else col
            for col in aggregated.columns
        ]
        pivoted_dfs.append(aggregated)

    # Combine all pivoted DataFrames
    final_df = pd.concat(pivoted_dfs, axis=1)

    # Remove duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    # Remove constant columns if required
    if remove_constant_columns:
        const_columns = [col for col in final_df.columns if final_df[col].nunique() == 1]
        final_df.drop(columns=const_columns, inplace=True)

    # Remove specified columns based on keywords
    if columns_to_remove is not None:
        columns_to_drop = [col for col in final_df.columns if any(keyword in col for keyword in columns_to_remove)]
        final_df = final_df.drop(columns=columns_to_drop)

    # Format numeric values to 4 decimal places
    for col in final_df.select_dtypes(include=["float"]).columns:
        final_df[col] = final_df[col].apply(lambda x: f"{x:.4f}")

    # Rename column headers for clarity
    final_df.columns = final_df.columns.str.replace("global", "multi-level")
    final_df.columns = final_df.columns.str.replace("level", "single-level")
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    # Format elapsed time
    if 'elapsed_time' in final_df.columns:
        final_df['elapsed_time'] = final_df['elapsed_time'].round().astype(int)
        final_df['elapsed_time'] = final_df['elapsed_time'].apply(
            lambda x: f"{x // 3600:02d} h {x % 3600 // 60:02d} m"
        )

    # Define custom column order
    custom_column_order = [
        col for col in final_df.columns if all(x not in col for x in ["AutoARIMA", "AutoETS", "single-level", "multi-level"])
    ] + [
        col for col in final_df.columns if "AutoARIMA" in col
    ] + [
        col for col in final_df.columns if "AutoETS" in col
    ] + [
        col for col in final_df.columns if "single-level" in col
    ] + [
        col for col in final_df.columns if "multi-level" in col
    ]

    # Reorder columns
    final_df = final_df[custom_column_order]

    return final_df
