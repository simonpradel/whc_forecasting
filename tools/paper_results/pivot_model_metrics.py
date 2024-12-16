import pandas as pd


def pivot_model_metrics(
    df_list, metric, row_values, column_values, add_to_row_values=None,
    add_to_col_values=None, forecast_method=None, method=None, forecast_model=None,
    time_limit=None, dataset_type=None, aggfunc="mean", show_control_table=False
):
    """
    Creates a pivot table with the average values of a given metric based on row and column values.

    Additional functionality:
    - add_to_row_values: Adds additional row categories.
    - add_to_col_values: Iterates over a column and adds additional column categories.

    :return: DataFrame as a pivot table with average values of the selected metric.
    """
    results = []
    
    for i, df in enumerate(df_list):
        # Duplicate rows where 'forecast_model' is "AutoETS" or "AutoARIMA" (to create level rows)
        df_to_duplicate = df[df['forecast_model'].isin(['AutoETS', 'AutoARIMA'])].copy()
        df_to_duplicate['forecast_method'] = 'level'
        df = pd.concat([df, df_to_duplicate], ignore_index=True)

        # Filter by the provided parameters
        if forecast_method is not None:
            df = df[df['forecast_method'] == forecast_method]

        if forecast_model is not None:
            df = df[df['forecast_model'] == forecast_model]

        if dataset_type is not None:
            df = df[df['dataset_type'] == dataset_type]

        if metric is not None:
            df = df[df['metric'] == metric]

        if method is not None:
            df = df[df['method'] == method]

        if time_limit is not None:
            df = df[df['time_limit'] == time_limit]
            
        if 'Model' in df.columns:
            df = df.drop(columns=['Model'])

        # Add a new column for add_to_row_values
        if add_to_row_values is not None and add_to_row_values in df.columns:
            df["methodNew"] = df['method']
            df.loc[df['method'] == 'weighted_pred', 'methodNew'] = df['method'] + '_' + df['optim_method']

        results.append(df)

    # Combine all results
    df = pd.concat(results)

    # Filtering condition
    condition = (
        ((df['method'] != 'weighted_pred')) &
        (df['optim_method'] != 'ensemble_selection')
    )
    all_results = df[~condition].reset_index(drop=True)
    all_results.drop("method", axis=1, inplace=True)
    all_results = all_results.rename(columns={'methodNew': 'method'})

    # Replace method names
    replacements = {
        "base": "Base Forecast",
        "BottomUp": "Bottom-up",
        "MinTrace_method-ols": "Ordinary Least Squares",
        "MinTrace_method-wls_struct": "Structural Scaling",
        "equal_weights_pred": "WHC (equal weights)",
        "weighted_pred_differential_evolution": "WHC (differential evolution)",
        "weighted_pred_ensemble_selection": "WHC (ensemble selection)",
        "weighted_pred_optimize_nnls": "WHC (nnls)"
    }
    all_results["method"] = all_results["method"].replace(replacements)

    method_order = [
        "Base Forecast", "Bottom-up", "Ordinary Least Squares", "Structural Scaling",
        "WHC (equal weights)", "WHC (differential evolution)", "WHC (ensemble selection)", "WHC (nnls)"
    ]
    all_results["method"] = pd.Categorical(
        all_results["method"], categories=method_order, ordered=True
    )
    all_results = all_results.sort_values("method").reset_index(drop=True)
    
    # If add_to_col_values is specified
    if add_to_col_values:
        all_pivots = []
        for col_value in df[add_to_col_values].unique():
            filtered_df = all_results[all_results[add_to_col_values] == col_value]
            if add_to_col_values == "forecast_method":
                col_value = col_value.replace("level", "single-level").replace("global", "multi-level")
            
            pivot_table = filtered_df.pivot_table(
                index=row_values, columns=column_values, values="value", aggfunc=aggfunc
            )

            if show_control_table:
                print("The entries in the table must match the number of datasets.")
                control_table = filtered_df.pivot_table(
                    index=row_values, columns=column_values, values="value", aggfunc="count"
                )
                print(control_table)

            pivot_table.columns = [f"{col} ({col_value})" for col in pivot_table.columns]
            all_pivots.append(pivot_table)

        # Merge all pivots
        final_pivot = all_pivots[0]
        for additional_pivot in all_pivots[1:]:
            final_pivot = final_pivot.merge(
                additional_pivot, left_index=True, right_index=True, how="outer"
            )
    else:
        if show_control_table:
            print("The entries in the table must match the number of datasets.")
            control_table = all_results.pivot_table(
                index=row_values, columns=column_values, values="value", aggfunc="count"
            )
            print(control_table)
        final_pivot = all_results.pivot_table(
            index=row_values, columns=column_values, values="value", aggfunc=aggfunc
        )

    if row_values == "method":
        # Remove specific labels from column names
        final_pivot.columns = [
            col.replace(" (global)", "").replace(" (single-level)", "")
            if "AutoETS" in col or "AutoARIMA" in col else col
            for col in final_pivot.columns
        ]
        
        # Logic for removing columns
        columns_to_drop = [
            col for col in final_pivot.columns
            if ("AutoETS" in col or "AutoARIMA" in col) and
              ("multi-level" in col or "single-level" in col)
        ]

        final_pivot = final_pivot.drop(columns=columns_to_drop)

    for col in final_pivot.select_dtypes(include=["float", "int"]).columns:
        # Round and ensure exactly 4 decimal places
        final_pivot[col] = final_pivot[col].apply(lambda x: format(x, ".4f"))

    # Define the desired column order
    custom_column_order = [
        col for col in final_pivot.columns if all(x not in col for x in ["AutoARIMA", "AutoETS", "single-level", "multi-level"])
    ] + [
        col for col in final_pivot.columns if "AutoARIMA" in col
    ] + [
        col for col in final_pivot.columns if "AutoETS" in col
    ] + [
        col for col in final_pivot.columns if "single-level" in col
    ] + [
        col for col in final_pivot.columns if "multi-level" in col
    ]

    # Sort the columns according to the custom order
    final_pivot = final_pivot[custom_column_order]

    return final_pivot.reset_index()
