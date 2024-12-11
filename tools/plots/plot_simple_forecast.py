import matplotlib.pyplot as plt
import pandas as pd

def plot_simple_forecast(results, date = 'date', actual_label = "total", pred_label = "opt_method", title = "'Tatsächliche Werte vs. Gewichtete Prognosen'"):
    """
    Plottet die rollierenden Prognosen gegen die tatsächlichen Werte.
    
    Parameters:
    results (dict): Dictionary mit den rollierenden Prognosen für den Testbereich.
    """
    # Plotten der tatsächlichen Werte und der gewichteten Prognosen
    plt.figure(figsize=(14, 7))
    plt.plot(results[date], results[actual_label], label='Tatsächliche Werte', color='blue')
    plt.plot(results[date], results[pred_label], label='Gewichtete Prognosen', color='orange')
    plt.xlabel('Datum')
    plt.ylabel('Wert')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

