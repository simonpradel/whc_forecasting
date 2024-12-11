def apply_hierarchical_mapping(df, source_column, mapping):
    """
    Map values in a source column to hierarchical levels based on a predefined mapping, starting from the deepest level.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    source_column (str): The name of the column where the values are being checked.
    mapping (list of lists): A list of lists, where each list contains the source value and its hierarchical levels.

    Returns:
    pd.DataFrame: DataFrame with new columns for LVL1, LVL2, LVL3, LVL4, etc.
    """
    # Finde die maximale Anzahl an Hierarchie-Ebenen
    max_levels = max(len(item) for item in mapping) 
    for i in range(1, max_levels + 1):
        df[f'LVL{i}'] = None  # Erstelle leere Spalten für LVL1, LVL2, LVL3, LVL4, etc.

    # Mapping anwenden
    for row in mapping:
        pl_line_value = row[0]
        levels = row[0:]  # Hierarchie-Ebenen, die auf LVL1, LVL2, LVL3, LVL4 aufgeteilt werden
        
        # Finde die Zeilen, die den PL_LINE-Wert enthalten
        matching_rows = df[source_column] == pl_line_value
        
        # Wir iterieren rückwärts, also letzte Ebene zuerst (letzter Eintrag = LVL1)
        for i, level_value in enumerate(reversed(levels), start=1):
            df.loc[matching_rows, f'LVL{i}'] = level_value

    return df