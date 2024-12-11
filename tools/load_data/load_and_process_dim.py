from databricks.sdk.runtime import *
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import Window

def load_dimension_table(dim_table_name, password_system, shared_members=None):
    """
    Lädt eine Dimensionstabelle aus SQL Server in ein Spark DataFrame und konvertiert es in ein Pandas DataFrame.
    Wenn shared_members = 1 ist, werden bei doppelten 'Child'-Werten die Zeilen mit SharedMember = 1 bevorzugt.
    Wenn shared_members = None, wird die Spalte 'SharedMember' nicht berücksichtigt.

    :param dim_table_name: Name der Dimensionstabelle (z.B. 'vDIM_SEGMENT')
    :param password_system: Passwort für die Verbindung zum SQL Server.
    :param shared_members: Optionaler Integer-Wert, um nach dem SharedMember zu filtern (z.B. 0, 1).
                           Wenn None, wird die Spalte 'SharedMember' nicht berücksichtigt.
    :return: Pandas DataFrame, das die geladenen Daten enthält.
    """
    
    # Benutzername für die Verbindung
    username_system = r'''Databricks_READ'''
    scope_name = 'sqlserver'
    
    # Verbindungs-URL
    url = f"jdbc:sqlserver://sqlserver.telefonica;databaseName=CO_PL;encrypt=true;trustServerCertificate=true;"
    
    # SQL-Abfrage zum Laden der Daten
    if shared_members is None:
        # Abfrage ohne Berücksichtigung von 'SharedMember'
        query = f'''
        SELECT 
            {dim_table_name}.Parent,
            {dim_table_name}.Child,
            {dim_table_name}.Alias,
            {dim_table_name}.Hier_Level
        FROM 
            [CO_PL].[Cube].[{dim_table_name}]
        '''
    else:
        # Abfrage mit 'SharedMember' Spalte
        query = f'''
        SELECT 
            {dim_table_name}.Parent,
            {dim_table_name}.Child,
            {dim_table_name}.Alias,
            {dim_table_name}.Hier_Level,
            {dim_table_name}.SharedMember
        FROM 
            [CO_PL].[Cube].[{dim_table_name}]
        '''
    
    # Daten laden
    df_segment = spark.read \
        .format("jdbc") \
        .option("reliabilityLevel", "BEST_EFFORT") \
        .option("url", url) \
        .option("query", query) \
        .option("user", username_system) \
        .option("password", password_system) \
        .load()
    
    # Anwendung der Logik basierend auf shared_members, falls vorhanden
    if shared_members == 0:
        # Filter nur für SharedMember = 0
        df_segment = df_segment.filter(F.col("SharedMember") == 0)
        
    elif shared_members == 1:
        # Bevorzugt SharedMember = 1 bei Duplikaten in 'Child'
        window_spec = Window.partitionBy("Child").orderBy(F.desc("SharedMember"))
        df_segment = df_segment.withColumn("rank", F.row_number().over(window_spec))
        df_segment = df_segment.filter(F.col("rank") == 1).drop("rank")
    
    # DataFrame in Pandas DataFrame konvertieren
    df_segment = df_segment.toPandas()
    
    return df_segment


def process_hierarchy(df, target_level, direct_alias=False):
    """
    Funktion zur Erstellung eines Dictionaries für die Hierarchie und zur Bestimmung der Eltern auf einer spezifischen Hierarchieebene.
    Wenn Hier_Level NaN ist oder direct_alias=True, wird der Alias direkt verwendet. Wenn das target_level größer ist als die maximale
    Hierarchieebene, wird der Alias des höchsten vorhandenen Levels zurückgegeben.

    :param df: DataFrame mit den Hierarchieinformationen. Muss die Spalten 'Child', 'Parent', 'Hier_Level' und 'Alias' enthalten.
    :param target_level: Ziel-Hierarchieebene, auf der der Parent gefunden werden soll.
    :param direct_alias: Boolescher Wert, der bestimmt, ob der Alias direkt übernommen werden soll.
    :return: DataFrame mit einer neuen Spalte 'Alias_New', die die Aliase der Eltern auf der Ziel-Hierarchieebene enthält.
    """
    
    # Schritt 1: Mapping der Parent-Child-Beziehungen erstellen
    hierarchy_dict = df.set_index('Child')['Parent'].to_dict()
    
    # Schritt 2: Mapping der Aliase erstellen
    alias_dict = df.set_index('Child')['Alias'].to_dict()

    # Schritt 3: Mapping der Hierarchieebenen erstellen
    level_dict = df.set_index('Child')['Hier_Level'].to_dict()

    # Funktion zum Finden des Parents auf Hierarchieebene X und Rückgabe des Aliases
    def find_parent_lvl_X(child, target_level):
        current_child = child
        max_level_reached = -1  # Um das höchste erreichte Level zu verfolgen
        while True:
            # Überprüfen, ob der Alias direkt übernommen werden soll
            if direct_alias:
                return alias_dict.get(current_child, None)

            # Finde die Hierarchieebene des aktuellen Kindes
            current_hierarchy = level_dict.get(current_child, None)
            if current_hierarchy is None:
                return None
            
            # Verfolge das höchste Level
            max_level_reached = max(max_level_reached, current_hierarchy)

            # Wenn Hier_Level NaN ist, gib den Alias des Kindes zurück
            if pd.isna(current_hierarchy):
                return alias_dict.get(current_child, None)

            # Wenn das aktuelle Level das Ziellevel ist, gib den Alias des Kindes zurück
            if current_hierarchy == target_level:
                return alias_dict.get(current_child, None)

            # Wenn das aktuelle Level kleiner als das Ziellevel ist, gibt es keinen weiteren Parent auf dem Ziellevel
            if current_hierarchy < target_level:
                # Wenn das Ziellevel größer ist als das maximale Level, gib den Alias des höchsten Levels zurück
                return alias_dict.get(current_child, None)

            # Gehe zum übergeordneten Parent auf der aktuellen Ebene
            current_child = hierarchy_dict.get(current_child, None)
            if current_child is None:
                # Wenn wir keinen weiteren Parent finden, gib den Alias des höchsten Levels zurück
                return alias_dict.get(current_child, alias_dict.get(child))

    # Wende die Funktion auf die 'Child'-Spalte an, um die 'Alias_New'-Spalte zu erstellen
    df['Alias_New'] = df['Child'].apply(lambda x: find_parent_lvl_X(x, target_level))

    # Ergebnis als Spark DataFrame erstellen
    df = spark.createDataFrame(df)
    
    return df

def filter_by_top_pl_line(df, top_pl_line):
    """
    Filters the rows of the DataFrame based on the condition that the Parent at some point has the alias 'top_pl_line'
    or the alias of the Child is directly 'top_pl_line'. Also adds a column indicating whether the Child is the
    lowest hierarchy level, as well as a column with the alias of the lowest level.
    
    :param df: DataFrame with hierarchy information. Must contain the columns 'Child', 'Parent', and 'Alias'.
    :param top_pl_line: The alias value to filter on (e.g., 'Mobile Service Revenue (PL_MSR)').
    :return: DataFrame containing only the rows whose Parent hierarchy eventually reaches the alias 'top_pl_line'
             or whose alias is directly 'top_pl_line', as well as a column indicating whether the Child is the lowest level
             and a column with the alias of the lowest level.
    """
    
    # Step: Create mapping of Parent-Child relationships
    hierarchy_dict = df.set_index('Child')['Parent'].to_dict()

    # Step: Create mapping of aliases
    alias_dict = df.set_index('Child')['Alias'].to_dict()

    # Step: Determine if a Child is the lowest level (does not appear as a Parent)
    lowest_level_set = set(df['Child']) - set(df['Parent'])
    
    # Function to find the top Parent where the alias matches
    def find_top_parent_by_alias(child):
        current_child = child
        while current_child is not None:
            # If the Parent alias is the top_pl_line, return True
            parent_alias = alias_dict.get(hierarchy_dict.get(current_child))
            if parent_alias == top_pl_line:
                return True
            
            # Move to the next Parent
            current_child = hierarchy_dict.get(current_child, None)
        
        return False

    # Apply the function to the 'Child' column to check if the Parent alias eventually points to top_pl_line
    df['top_pl_line'] = df['Child'].apply(
        lambda x: top_pl_line if find_top_parent_by_alias(x) or alias_dict.get(x) == top_pl_line else None
    )
    
    # Add the column indicating whether this row is the lowest level
    df['is_lowest_level'] = df['Child'].apply(lambda x: x in lowest_level_set)
    
    # Filter the rows where top_pl_line is not null
    filtered_df = df[df['top_pl_line'].notna()]
    
    # Function to find the lowest level alias
    def find_lowest_level_alias(child):
        current_child = child
        
        # Loop to find the lowest level
        while current_child in filtered_df['Parent'].unique():
            # Find the row where the Parent matches current_child
            parent_row = filtered_df[filtered_df['Parent'] == current_child].iloc[0]
            # Update current_child with the Child from this row
            current_child = parent_row['Child']
        
        # If no more Parent is present, return the alias of the last current_child
        return alias_dict.get(current_child, None)
    
    # Apply the function to each Child to find the lowest level alias
    filtered_df['find_lowest_level_alias'] = filtered_df['Child'].apply(find_lowest_level_alias)
    
    # Create result as Spark DataFrame
    filtered_df = spark.createDataFrame(filtered_df)
    
    return filtered_df



