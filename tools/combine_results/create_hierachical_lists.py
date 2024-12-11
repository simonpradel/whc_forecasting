# Annahme: mapping_df ist dein DataFrame, PL_lines ist die Liste
# last_element ist der festgelegte Wert, bei dem die Iteration gestoppt werden soll
from pyspark.sql.functions import col

def create_hierachical_lists(mapping_df, PL_lines_list, last_element = None):
    """
    Diese Funktion iteriert durch die Listenelemente von PL_lines und extrahiert Alias-Ketten basierend auf 
    der Mapping-Tabelle, bis entweder ein festgelegtes letztes Element oder ein NULL-Wert erreicht ist.
    
    Parameters:
    mapping_df (DataFrame): PySpark DataFrame mit den Spalten 'Alias', 'Parent', 'child'
    PL_lines_list (list): Liste der Elemente, für die die Ketten extrahiert werden sollen
    last_element (str): Festgelegtes letztes Element, bei dem die Iteration gestoppt wird
    
    Returns:
    list: Eine Liste von Listen, in denen jede innere Liste die Alias-Kette für ein Element von PL_lines_list enthält
    """
    
    # Leere Liste, um die Ergebnisse zu speichern
    result_list = []

    # Iteration durch die Listenelemente von PL_lines_list
    for line in PL_lines_list:
        # Erstelle eine leere Liste für das aktuelle Element
        alias_list = []

        # Setze den Startwert für den Alias auf das aktuelle Listenelement
        current_alias = line

        # Schleife, um Alias und Parent zu verfolgen
        while current_alias is not None:
            # Filtere die Zeile basierend auf dem aktuellen Alias
            row = mapping_df.filter(col("Alias") == current_alias).first()

            if row is not None:
                # Füge den aktuellen Alias zur Liste hinzu
                alias_list.append(row['Alias'])

                # Überprüfe, ob der aktuelle Alias das festgelegte letzte Element ist
                if row['Alias'] == last_element:
                    break  # Beende die Schleife, wenn der letzte Alias erreicht ist

                # Setze den neuen Alias auf den Parent-Wert der aktuellen Zeile
                current_parent = row['Parent']

                # Wenn der Parent nicht None ist, suche ihn in der 'child'-Spalte
                if current_parent is not None:
                    row = mapping_df.filter(col("child") == current_parent).first()
                    if row is not None:
                        current_alias = row['Alias']  # Wechsle zu dem Alias, der zum Parent gehört
                    else:
                        current_alias = None
                else:
                    current_alias = None
            else:
                current_alias = None

        # Füge die Liste der Aliase zur Ergebnisliste hinzu
        result_list.append(alias_list)

    return result_list