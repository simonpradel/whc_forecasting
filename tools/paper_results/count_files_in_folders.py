import os
import warnings

def count_files_in_folders(datasets, current_dir, check, max_files=None, return_datasets_below_threshold=False):
    """
    Zählt die Anzahl der Dateien in jedem Dataset-Ordner und gibt optional eine Liste der Dataset-Namen zurück,
    bei denen die Anzahl an Dateien unter dem gegebenen Schwellenwert max_files liegt.

    Args:
        datasets (list): Liste der Dataset-Namen.
        base_dir (str): Basisverzeichnis, in dem die Dataset-Ordner liegen.
        max_files (int, optional): Schwellenwert für die maximale Anzahl an Dateien pro Ordner.
        return_datasets_below_threshold (bool, optional): Wenn True, gibt die Funktion eine Liste der Dataset-Namen
                                                         zurück, bei denen die Dateianzahl unter max_files liegt.

    Returns:
        dict: Ein Dictionary mit Dataset-Namen als Schlüsseln und der Anzahl an Dateien im jeweiligen Ordner.
        list (optional): Eine Liste der Dataset-Namen, bei denen die Dateianzahl unter max_files liegt.
    """
    file_counts = {}
    below_threshold_datasets = []

    for dataset in datasets:
        dataset_dir = os.path.join(current_dir, dataset, check)
        try:
            # Zähle die Anzahl der Dateien im Verzeichnis
            file_count = len([f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))])
            file_counts[dataset] = file_count
            
            # Prüfe, ob die Anzahl der Dateien unter dem Schwellenwert liegt
            if max_files is not None and file_count < max_files:
                below_threshold_datasets.append(dataset)
                
        except FileNotFoundError:
            warnings.warn(f"Verzeichnis für Dataset '{dataset}' nicht gefunden.")
    
    if return_datasets_below_threshold:
        return file_counts, below_threshold_datasets
    return file_counts