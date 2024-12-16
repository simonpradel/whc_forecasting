import pandas as pd

def rank_models_across_datasets(
    df_list, model=None, optim_method=None, forecast_method=None, method=None, 
    metric=None, forecast_model=None, drop_columns=None, remove_constant_columns=False, 
    grouping_variable=None, time_limit=None, row_values=None, add_to_row_values=None, 
    columns_to_remove=None, show_control_table=False, sort_values=False
):
    """
    This function ranks models across datasets based on their performance metrics 
    and aggregates results for analysis. It supports filtering by specific parameters, 
    ranking by value, and grouping by specified variables. Results can be pivoted 
    and formatted for better interpretability.

    Parameters:
    - df_list: List of DataFrames containing results for different datasets.
    - model, optim_method, forecast_method, method, metric, forecast_model: Filters for specific models or methods.
    - drop_columns: Columns to drop (if specified).
    - remove_constant_columns: Flag to remove constant columns.
    - grouping_variable: Variable to group results (e.g., forecast method).
    - time_limit: Optional time constraint to filter results.
    - row_values: Name of the primary row grouping variable.
    - add_to_row_values: Additional columns to concatenate with the row grouping variable.
    - columns_to_remove: Keywords for columns to be excluded in the final DataFrame.
    - show_control_table: Flag to display control table for verification.
    - sort_values: Whether to sort the final DataFrame by average value.

    Returns:
    - final_df: A DataFrame containing aggregated results with ranks, wins, and formatted metrics.
    """
    results = []

    # Iterate through the list of DataFrames
    for i, df in enumerate(df_list):
        # Duplicate rows for specific forecast models and set their forecast method to 'level'
        df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
        df_to_duplicate['forecast_method'] = 'level'
        df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Filter DataFrame based on specified parameters
        if model:
            df = df[df['Model'] == model]
        if optim_method:
            df = df[df['optim_method'] == optim_method]
        if method:
            df = df[df['method'] == method]
        if metric:
            df = df[df['metric'] == metric]
        if forecast_model:
            df = df[df['forecast_model'] == forecast_model]
        if forecast_method:
            df = df[df['forecast_method'] == forecast_method]
        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]

        # Rename forecast methods for consistency
        df['forecast_method'] = df['forecast_method'].replace({
            'global': 'multi-level',
            'level': 'single-level'
        })

        # Remove entries that do not meet specific conditions
        condition = (
            (df['method'] != 'weighted_pred') & 
            (df['optim_method'] != 'ensemble_selection')
        )
        df = df[~condition].reset_index(drop=True)

        # Group results based on the specified grouping variable
        grouped_results = []
        if grouping_variable is None:
            df['rank'] = df['value'].rank(ascending=True, method='min')
            rank_1_count = (df['rank'] == 1).sum()
            df['wins'] = df['rank'].apply(lambda x: 1.0 / rank_1_count if x == 1 else 0)

            # Add additional columns to row grouping if specified
            if add_to_row_values:
                if not isinstance(add_to_row_values, list):
                    add_to_row_values = [add_to_row_values]
                df["methodNew"] = df[row_values]
                for col in add_to_row_values:
                    if col in df.columns:
                        df["methodNew"] += "," + df[col].astype(str)
            
            grouped_results.append(df)
        else:
            for f_method, group in df.groupby(grouping_variable):
                group = group.copy()
                group['rank'] = group['value'].rank(ascending=True, method='min')
                rank_1_count = (group['rank'] == 1).sum()
                group['wins'] = group['rank'].apply(lambda x: 1.0 / rank_1_count if x == 1 else 0)
                
                if add_to_row_values:
                    if not isinstance(add_to_row_values, list):
                        add_to_row_values = [add_to_row_values]
                    group["methodNew"] = group[row_values]
                    for col in add_to_row_values:
                        if col in group.columns:
                            group["methodNew"] += "," + group[col].astype(str)
                
                grouped_results.append(group)
        
        results.append(pd.concat(grouped_results, ignore_index=True))

    # Combine results from all datasets
    all_results = pd.concat(results, ignore_index=True)
    
    if add_to_row_values:
        all_results.drop(row_values, axis=1, inplace=True)
    all_results = all_results.rename(columns={'methodNew': row_values})

    # Replace method names for better readability
    all_results[row_values] = all_results[row_values].replace({
        "base": "Base Forecast",
        "BottomUp": "Bottom-up",
        "MinTrace_method-ols": "Ordinary Least Squares",
        "MinTrace_method-wls_struct": "Structural Scaling",
        "equal_weights_pred": "WHC (equal weights)",
        "weighted_pred,differential_evolution": "WHC (differential evolution)",
        "weighted_pred,ensemble_selection": "WHC (ensemble selection)",
        "weighted_pred,optimize_nnls": "WHC (nnls)"
    })

    # Define method order for sorting
    method_order = [
        "Base Forecast",
        "Bottom-up",
        "Ordinary Least Squares",
        "Structural Scaling",
        "WHC (equal weights)",
        "WHC (differential evolution)",
        "WHC (ensemble selection)",
        "WHC (nnls)"
    ]
    all_results["method"] = pd.Categorical(
        all_results["method"], categories=method_order, ordered=True
    )
    all_results = all_results.sort_values("method").reset_index(drop=True)

    # Create pivot tables based on grouping variable
    if grouping_variable is not None:
        pivoted_dfs = []
        for f_method in all_results[grouping_variable].unique():
            temp_df = all_results[all_results[grouping_variable] == f_method].copy()
            if show_control_table:
                print("Control table:")
                print(temp_df.groupby(row_values).size())
            temp_df = temp_df.groupby(row_values).agg(
                avg_value=('value', 'mean'),
                std_value=('value', 'std'),
                avg_rank=('rank', 'mean'),
                total_wins=('wins', 'sum')
            ).reset_index()
            pivoted_dfs.append(temp_df)
        final_df = pd.concat(pivoted_dfs, axis=1)
    else:
        final_df = all_results.groupby(row_values).agg(
            avg_value=('value', 'mean'),
            std_value=('value', 'std'),
            avg_rank=('rank', 'mean'),
            total_wins=('wins', 'sum'),
            avg_elapsed_time=('elapsed_time', 'mean')
        ).reset_index()

    # Format elapsed time and numerical columns
    if 'avg_elapsed_time' in final_df.columns:
        final_df['avg_elapsed_time'] = final_df['avg_elapsed_time'].round().astype(int)
        final_df['avg_elapsed_time'] = final_df['avg_elapsed_time'].apply(
            lambda x: f"{x // 3600:02d} h {x % 3600 // 60:02d} m"
        )

    for col in final_df.select_dtypes(include=["float"]).columns:
        final_df[col] = final_df[col].apply(lambda x: format(x, ".2f"))

    # Drop unwanted columns
    if columns_to_remove:
        final_df = final_df.loc[:, ~final_df.columns.str.contains('|'.join(columns_to_remove))]

    return final_df
