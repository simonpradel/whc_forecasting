import pandas as pd
import matplotlib.pyplot as plt

def plot_methods_comparison(df, date_column='ds', y_cols=None, title='', figsize=(10, 6)):
    """
    Plotte ein Liniendiagramm, wobei die y_cols Spalten im Vordergrund liegen.
    Wenn keine Y-Spalten angegeben werden, werden alle Spalten außer der Datumsspalte geplottet.

    :param df: DataFrame, der die Daten enthält
    :param date_column: Name der Datumsspalte
    :param y_cols: Liste der Namen der Spalten, die im Vordergrund geplottet werden sollen (optional)
    :param title: Titel des Plots (optional)
    :param figsize: Größe des Plots (optional)
    """
    # Sicherstellen, dass die Datumsspalte als Index verwendet wird
    if df.index.name != date_column:  # Falls die Datumsspalte nicht bereits der Index ist
        df = df.reset_index() if df.index.name is not None else df  # Falls ein anderer Index vorhanden ist, zurücksetzen
        df[date_column] = pd.to_datetime(df[date_column])  # Datumsspalte in ein datetime-Format konvertieren
        df.set_index(date_column, inplace=True)  # Datumsspalte als Index setzen

    # Wenn keine y_cols angegeben sind, plottet man alle Spalten außer der Datumsspalte
    if y_cols is None:
        y_cols = df.columns.tolist()  # Liste aller Spalten
        if date_column in y_cols:  # Check if date_column exists in the DataFrame
            y_cols.remove(date_column)  # Datumsspalte entfernen

    # Bestimme die restlichen Spalten
    other_cols = [col for col in df.columns if col not in "y"]

    # Plotten der y_cols Spalten zuerst
    ax = df["y"].plot(figsize=figsize, linestyle='-', marker=None)

    # Danach die anderen Spalten plotten
    df[other_cols].plot(ax=ax, linestyle='-', marker=None, alpha=1)  # Alpha für Transparenz der anderen Spalten

    # Plot-Einstellungen
    ax.set_title(title)
    ax.set_xlabel('Datum')
    ax.set_ylabel('Wert')
    ax.grid(True)
    ax.legend(title='methods')

    # Zeige den Plot
    plt.show()