import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def custom_sum(x):
    """
    Benutzerdefinierte Aggregationsfunktion, die NaN-Werte ignoriert und NaN zurückgibt,
    wenn alle Werte in der Gruppe NaN sind.
    """
    if x.isna().all():
        return np.nan
    else:
        return x.sum()

def plot_timeseries_from_dict(dic, date_col, actuals_col, pred_col, variables, plot_individual=False, 
                              legend_below=False, top_n=None):
    """
    Plotte Zeitreihen für ausgewählte Variablen aus einem Dictionary von DataFrames.

    Args:
        dic (dict): Dictionary, wo die Keys Tupel von Variablen sind und die Values DataFrames.
        date_col (str): Name der Datenspalte im DataFrame.
        actuals_col (str): Name der Spalte mit den tatsächlichen Werten.
        pred_col (str): Name der Spalte mit den Vorhersagen.
        variables (list): Liste von Variablen, die im Tupel der Keys des dic vorkommen sollen.
        plot_individual (bool): Wenn True, werden die Zeitreihen einzeln geplottet, ansonsten in einem Plot.
        legend_below (bool): Wenn True, wird die Legende unter dem Plot angezeigt.
        top_n (int): Anzahl der Top-Zeitreihen, die basierend auf dem größten absoluten Wert der Vorhersagen oder tatsächlichen Werte angezeigt werden sollen.

    Returns:
        None
    """
    # Suche nach einem DataFrame im Dic, der genau die übergebenen Variablen enthält
    matching_key = None
    for key in dic.keys():
        if set(variables) == set(key):  # Exakt gleiche Variablen im Tupel
            matching_key = key
            break
    
    if matching_key is None:
        print(f"Kein passender DataFrame gefunden für Variablen: {variables}")
        return
    
    # Den passenden DataFrame holen
    df = dic[matching_key]
    
    # Gruppiere nach den Variablen und der Datenspalte, benutze custom_sum
    grouped = df.groupby(variables + [date_col]).agg({actuals_col: custom_sum, pred_col: custom_sum}).reset_index()

    # Optional: Wähle die Top "n" Zeitreihen basierend auf den absolut größten Werten
    if top_n is not None:
        # Berechne den maximalen absoluten Wert für jede Gruppe (Actuals und Predictions)
        grouped['max_abs_value'] = grouped[[actuals_col, pred_col]].abs().max(axis=1)
        
        # Gruppiere nach den Variablen und finde die Gruppen mit den größten absoluten Werten
        max_values_per_group = grouped.groupby(variables)['max_abs_value'].max().reset_index()
        
        # Sortiere nach den maximalen Werten und wähle die obersten 'top_n'
        top_groups = max_values_per_group.nlargest(top_n, 'max_abs_value')[variables]
        
        # Filtere den DataFrame für die Top-Gruppen
        grouped = grouped[grouped[variables].apply(tuple, axis=1).isin(top_groups.apply(tuple, axis=1))]
    
    if plot_individual:
        # Jede Gruppe separat plotten
        unique_groups = grouped[variables].drop_duplicates()
        
        for _, group_vals in unique_groups.iterrows():
            group_df = grouped[(grouped[variables] == group_vals.values).all(axis=1)]
            
            plt.figure(figsize=(10, 5))
            # Filtere NaN-Werte vor dem Plotten heraus
            actuals_filtered = group_df[[date_col, actuals_col]].dropna()
            preds_filtered = group_df[[date_col, pred_col]].dropna()
            
            # Plot Actuals und Predictions in derselben Farbe
            if not actuals_filtered.empty or not preds_filtered.empty:
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                
                # Plot Actuals, wenn sie nicht leer sind
                if not actuals_filtered.empty:
                    plt.plot(actuals_filtered[date_col], actuals_filtered[actuals_col], label='Actuals', color=color)
                
                # Plot Predictions, wenn sie nicht leer sind
                if not preds_filtered.empty:
                    plt.plot(preds_filtered[date_col], preds_filtered[pred_col], label='Predictions', linestyle='--', color=color)
            
            plt.title(f'Zeitreihe für Gruppe: {group_vals.to_dict()}')
            plt.xlabel('Datum')
            plt.ylabel('Wert')

            if legend_below:
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Legende unterhalb des Plots
            else:
                plt.legend()

            plt.show()
    else:
        # Alle Gruppen in einem Plot
        plt.figure(figsize=(10, 5))
        # Define color map to use the same color for actuals and predictions
        color_map = {}
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(grouped.groupby(variables))))
        for (name, group), color in zip(grouped.groupby(variables), color_cycle):
            # Filtere NaN-Werte vor dem Plotten heraus
            actuals_filtered = group[[date_col, actuals_col]].dropna()
            preds_filtered = group[[date_col, pred_col]].dropna()
            
            color_map[name] = color  # Store color for consistent coloring
            
            # Plot Actuals, wenn sie nicht leer sind
            if not actuals_filtered.empty:
                plt.plot(actuals_filtered[date_col], actuals_filtered[actuals_col], label=f'Actuals: {name}', color=color)
            
            # Plot Predictions, wenn sie nicht leer sind
            if not preds_filtered.empty:
                plt.plot(preds_filtered[date_col], preds_filtered[pred_col], linestyle='--', label=f'Predictions: {name}', color=color)
        
        plt.title('Zeitreihen')
        plt.xlabel('Datum')
        plt.ylabel('Wert')
        plt.grid()

        if legend_below:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)  # Legende unterhalb des Plots
        else:
            plt.legend()

        plt.show()