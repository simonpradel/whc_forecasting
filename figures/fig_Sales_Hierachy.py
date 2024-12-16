# This notebook will recreat figure 4 and 5 in the master thesis

# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt

# Create timestamps
dates = pd.date_range(start='2024-02-01', periods=5, freq='M')

# Create time series data for the example
data = {
    'Total Sales': [100, 120, 115, 130, 125],
    'Electronics - Online': [40, 46, 45, 48, 47],
    'Electronics - Physical': [20, 24, 23, 26, 25],
    'Clothing - Online': [15, 17, 16, 18, 17],
    'Clothing - Physical': [25, 33, 30, 38, 35]
}

# Create a DataFrame
df = pd.DataFrame(data, index=dates)

# Aggregation by Group 1 (Category)
df['Electronics'] = df['Electronics - Online'] + df['Electronics - Physical']
df['Clothing'] = df['Clothing - Online'] + df['Clothing - Physical']

# Aggregation by Group 2 (Channel)
df['Online'] = df['Electronics - Online'] + df['Clothing - Online']
df['Physical'] = df['Electronics - Physical'] + df['Clothing - Physical']

# Determine y-axis limits for all plots
y_min = min(df.min()) - 10
y_max = max(df.max()) + 5

# Function to add a black border
def add_black_box(ax):
    """Adds a black border to the given axis object."""
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

# Plot 1: Aggregation by Group 1 (Category)
plt.figure(figsize=(6, 4))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax1.plot(df.index, df['Electronics'], label='Electronics', linestyle='solid', color='#1f77b4')  # Blue
ax1.plot(df.index, df['Clothing'], label='Clothing', linestyle='solid', color='#2ca02c')  # Green
ax1.set_title('Aggregation by Category')
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_xticks([])  # Remove x-axis ticks
ax1.set_xticklabels([])  # Remove x-axis labels
ax1.set_ylim([y_min, y_max])  # Set same y-axis limits
add_black_box(ax1)  # Add a black border
plt.show()

# Plot 2: Aggregation by Group 2 (Channel)
plt.figure(figsize=(6, 4))
ax2 = plt.subplot(1, 1, 1)
ax2.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax2.plot(df.index, df['Online'], label='Online', linestyle='--', color='grey')
ax2.plot(df.index, df['Physical'], label='Physical', linestyle='-', color='grey')
ax2.set_title('Aggregation by Channel')
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_xticks([])  # Remove x-axis ticks
ax2.set_xticklabels([])  # Remove x-axis labels
ax2.set_ylim([y_min, y_max])  # Set same y-axis limits
add_black_box(ax2)  # Add a black border
plt.show()

# Plot 3: Aggregation by Group 1 and Group 2
plt.figure(figsize=(6, 4))
ax3 = plt.subplot(1, 1, 1)
ax3.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax3.plot(df.index, df['Electronics - Online'], label='Electronics - Online', linestyle='--', color='#1f77b4')  # Blue
ax3.plot(df.index, df['Electronics - Physical'], label='Electronics - Physical', linestyle='-', color='#1f77b4')  # Blue
ax3.plot(df.index, df['Clothing - Online'], label='Clothing - Online', linestyle='--', color='#2ca02c')  # Green
ax3.plot(df.index, df['Clothing - Physical'], label='Clothing - Physical', linestyle='-', color='#2ca02c')  # Green
ax3.set_title('Aggregation by Category and Channel')
ax3.legend(loc='upper left')
ax3.grid(True)
ax3.set_xticks([])  # Remove x-axis ticks
ax3.set_xticklabels([])  # Remove x-axis labels
ax3.set_ylim([y_min, y_max])  # Set same y-axis limits
add_black_box(ax3)  # Add a black border
plt.show()

# COMMAND ----------

# DBTITLE 1,Forecast
import pandas as pd
import matplotlib.pyplot as plt

# Create timestamps for Actuals and Forecast
dates_actual = pd.date_range(start='2024-02-01', periods=5, freq='M')
dates_forecast = pd.date_range(start='2024-06-01', periods=3, freq='M')

# Create time series for the example (Actuals)
data_actual = {
    'Total Sales': [100, 120, 115, 130, 125],
    'Electronics - Online': [40, 46, 45, 48, 47],
    'Electronics - Physical': [20, 24, 23, 26, 25],
    'Electronics': [60, 68, 68, 74, 72],
    'Clothing': [40, 50, 46, 56, 52],
    'Online': [55, 63, 61, 66, 67],
    'Physical': [45, 57, 53, 64, 60],
    'Clothing - Online': [15, 17, 16, 18, 17],
    'Clothing - Physical': [25, 33, 30, 38, 35]
}

# Forecast data
forecast_data = {
    'Electronics': [72, 76, 75],
    'Clothing': [52, 54, 58],
    'Online': [67, 67, 69],
    'Physical': [60, 67, 65],
    'Electronics - Online': [47, 45, 48],
    'Electronics - Physical': [25, 26, 27],
    'Clothing - Online': [17, 18, 19],
    'Clothing - Physical': [35, 35, 37],
    'Aggregated Forecast by Channel': [125, 134, 134],
    'Aggregated Forecast by Category': [125, 130, 133],
    'Aggregated Forecast by Channel + Category': [125, 134, 131]
}

# Create DataFrames
df_actual = pd.DataFrame(data_actual, index=dates_actual)
df_forecast = pd.DataFrame(forecast_data, index=dates_forecast)
df_combined = pd.concat([df_actual, df_forecast])

# Determine y-axis limits
y_min = min(df_combined.min()) - 10
y_max = max(df_combined.max()) + 15

# Function for black border
def add_black_box(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

# Mark forecast start
forecast_start = dates_forecast[0]

# Plot 1: Aggregation by Category
plt.figure(figsize=(6, 4))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(df_combined.index, df_combined['Total Sales'], label='Total Sales', color='black')
ax1.plot(df_combined.index, df_combined['Electronics'], label='Electronics', color='#1f77b4')
ax1.plot(df_combined.index, df_combined['Clothing'], label='Clothing', color='#2ca02c')
ax1.plot(df_combined.index, df_combined['Aggregated Forecast by Category'], label='Aggregated Forecast', linestyle='--', color='black')
ax1.axvline(forecast_start, color='#b22222', linestyle='-', label='Forecast Start')

ax1.set_title('Aggregated Forecast by Category')
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_ylim([y_min, y_max])
add_black_box(ax1)
ax1.set_xticks([])
plt.show()

# Plot 2: Aggregation by Channel
plt.figure(figsize=(6, 4))
ax2 = plt.subplot(1, 1, 1)
ax2.plot(df_combined.index, df_combined['Total Sales'], label='Total Sales', color='black')
ax2.plot(df_combined.index, df_combined['Online'], label='Online', linestyle='--', color='grey')
ax2.plot(df_combined.index, df_combined['Physical'], label='Physical', color='grey')
ax2.plot(df_combined.index, df_combined['Aggregated Forecast by Channel'], label='Aggregated Forecast', linestyle=':', color='black')
ax2.axvline(forecast_start, color='#b22222', linestyle='-', label='Forecast Start')

ax2.set_title('Aggregated Forecast by Channel')
ax2.legend(loc='upper left')
ax2.grid(True)
ax2.set_ylim([y_min, y_max])
add_black_box(ax2)
ax2.set_xticks([])
plt.show()

# Plot 3: Aggregation by Category and Channel
plt.figure(figsize=(6, 4))
ax3 = plt.subplot(1, 1, 1)
ax3.plot(df_combined.index, df_combined['Total Sales'], label='Total Sales', color='black')
ax3.plot(df_combined.index, df_combined['Electronics - Online'], label='Electronics - Online', linestyle='--', color='#1f77b4')
ax3.plot(df_combined.index, df_combined['Electronics - Physical'], label='Electronics - Physical', color='#1f77b4')
ax3.plot(df_combined.index, df_combined['Clothing - Online'], label='Clothing - Online', linestyle='--', color='#2ca02c')
ax3.plot(df_combined.index, df_combined['Clothing - Physical'], label='Clothing - Physical', color='#2ca02c')
ax3.plot(df_combined.index, df_combined['Aggregated Forecast by Channel + Category'], label='Aggregated Forecast', linestyle='-.', color='black')
ax3.axvline(forecast_start, color='#b22222', linestyle='-', label='Forecast Start')

ax3.set_title('Aggregated Forecast by Channel + Category')
ax3.legend(loc='upper left')
ax3.grid(True)
ax3.set_ylim([y_min, y_max])
add_black_box(ax3)
ax3.set_xticks([])
plt.show()


############################################################
############################################################
############################################################

# Plot 4: Aggregation by Category and Channel with Forecast
plt.figure(figsize=(6, 4))
ax4 = plt.subplot(1, 1, 1)
ax4.plot(df_combined.index, df_combined['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax4.plot(df_combined.index, df_combined['Aggregated Forecast by Channel'], label='Aggregated Forecast by Channel', linestyle=':', color='black')
ax4.plot(df_combined.index, df_combined['Aggregated Forecast by Category'], label='Aggregated Forecast by Category', linestyle='--', color='black')
ax4.plot(df_combined.index, df_combined['Aggregated Forecast by Channel + Category'], label='Aggregated Forecast by Channel + Category', linestyle='-.', color='black')
ax4.set_xticks([])  # Remove x-axis ticks
ax4.set_xticklabels([])  # Remove x-axis labels

# Vertical line for forecast start
ax4.axvline(forecast_start, color='#b22222', linestyle='solid', label='Forecast Start')

# Legend and formatting
ax4.set_title('Aggregated Forecasts')
ax4.legend(loc='lower left')
ax4.grid(True)
ax4.set_ylim([y_min, y_max])  # Set same y-axis limits
add_black_box(ax4)  # Add a black border
plt.show()

# COMMAND ----------

# DBTITLE 1,Zusammen
import pandas as pd
import matplotlib.pyplot as plt

# Create timestamps
dates = pd.date_range(start='2024-02-01', periods=5, freq='M')

# Create time series for the example
data = {
    'Total Sales': [100, 120, 115, 130, 125],
    'Electronics - Online': [40, 46, 45, 48, 47],
    'Electronics - Physical': [20, 24, 23, 26, 25],
    'Clothing - Online': [15, 17, 16, 18, 17],
    'Clothing - Physical': [25, 33, 30, 38, 35]
}

# Create a DataFrame
df = pd.DataFrame(data, index=dates)

# Aggregation by Group 1 (Category)
df['Electronics'] = df['Electronics - Online'] + df['Electronics - Physical']
df['Clothing'] = df['Clothing - Online'] + df['Clothing - Physical']

# Aggregation by Group 2 (Channel)
df['Online'] = df['Electronics - Online'] + df['Clothing - Online']
df['Physical'] = df['Electronics - Physical'] + df['Clothing - Physical']

# Aggregation by Group 1 and Group 2
df['Electronics - Online'] = df['Electronics - Online']
df['Electronics - Physical'] = df['Electronics - Physical']
df['Clothing - Online'] = df['Clothing - Online']
df['Clothing - Physical'] = df['Clothing - Physical']

# Determine y-axis limits for all plots
y_min = min(df.min())-10
y_max = max(df.max())+5

# Function to add a black border
def add_black_box(ax):
    """Adds a black border to the given axis object."""
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

# Create the plot figure
plt.figure(figsize=(10, 12))

# Plot 1: Aggregation by Group 1 (Category)
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax1.plot(df.index, df['Electronics'], label='Electronics', linestyle='solid', color='#1f77b4')  # Blue
ax1.plot(df.index, df['Clothing'], label='Clothing', linestyle='solid', color='#2ca02c')  # Green
ax1.set_title('Aggregation by Category')
#ax1.set_ylabel('Sales')
ax1.legend(loc='upper right')
ax1.grid(True)
ax1.set_ylim([y_min, y_max])  # Set same y-axis limits
ax1.set_xticks([])  # Remove x-axis ticks
ax1.set_xticklabels([])  # Remove x-axis labels
add_black_box(ax1)  # Add a black border

# Plot 2: Aggregation by Group 2 (Channel)
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax2.plot(df.index, df['Online'], label='Online', linestyle='--', color='grey')  
ax2.plot(df.index, df['Physical'], label='Physical', linestyle='-', color='grey')  
ax2.set_title('Aggregation by Channel')
#ax2.set_ylabel('Sales')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_ylim([y_min, y_max])  # Set same y-axis limits
ax2.set_xticks([])  # Remove x-axis ticks
ax2.set_xticklabels([])  # Remove x-axis labels
add_black_box(ax2)  # Add a black border

# Plot 3: Aggregation by Group 1 and Group 2
ax3 = plt.subplot(3, 1, 3)
ax3.plot(df.index, df['Total Sales'], label='Total Sales', linestyle='solid', color='black')
ax3.plot(df.index, df['Electronics - Online'], label='Electronics - Online', linestyle='--', color='#1f77b4')  # Blue
ax3.plot(df.index, df['Electronics - Physical'], label='Electronics - Physical', linestyle='-', color='#1f77b4')  # Blue
ax3.plot(df.index, df['Clothing - Online'], label='Clothing - Online', linestyle='--', color='#2ca02c')  # Green
ax3.plot(df.index, df['Clothing - Physical'], label='Clothing - Physical', linestyle='-', color='#2ca02c')  # Green
ax3.set_title('Aggregation by Category and Channel')
#ax3.set_ylabel('Sales')
ax3.legend(loc='upper right')
ax3.grid(True)
ax3.set_ylim([y_min, y_max])  # Set same y-axis limits
ax3.set_xticks([])  # Remove x-axis ticks
ax3.set_xticklabels([])  # Remove x-axis labels
add_black_box(ax3)  # Add a black border

plt.tight_layout()
plt.show()
