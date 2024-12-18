# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------
###############################################################################
# Load Data
###############################################################################
from pyspark.sql.functions import dense_rank, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import expr
from pyspark.sql import SparkSession
from itertools import chain, combinations

spark = SparkSession.builder.getOrCreate()

# Load the tourism dataset
tourism = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`tourism_original`
""")

tourism.display()
print(tourism.select("`Region`", "`State`", "`Purpose`").distinct().count())

# COMMAND ----------
###############################################################################
# Overview of the original data
###############################################################################

columns = ["`Region`", "`State`", "`Purpose`"] #  "`Country`", is a constant
df_original = tourism

# Function to get all subsets of a list
def all_subsets(lst):
    return chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1))

# List to store results
results = []

# Count distinct values for each subset of columns
for subset in all_subsets(columns):
    subset_list = list(subset)
    distinct_count = df_original.select(*subset_list).distinct().count()
    results.append((str(subset_list), distinct_count, "Subset"))

# Convert results into a DataFrame
results_df = spark.createDataFrame(results, ["Column(s)", "Distinct Count", "Type"])

# Count distinct values for the entire dataset
distinct_count_sum = results_df.select("Distinct Count").groupBy().sum("Distinct Count").collect()[0][0]

# Convert results into a DataFrame and display
results_df.display()
print(distinct_count_sum)

# COMMAND ----------
###############################################################################
# Preparation of the data
###############################################################################

df = tourism 
# Rename columns
df = df.withColumnRenamed("Quarter", "date").withColumnRenamed("trips", "total")

# Subtract one day from each `date` column
df = df.withColumn("date", expr("date_sub(date, 1)"))

# Define a window to group by Region, State, and Purpose
window_spec = Window.partitionBy("Region", "State", "Purpose").orderBy("Region", "State", "Purpose")

# Assign a unique ID to each combination
df = df.withColumn("id_temp", dense_rank().over(window_spec))

# Get the first occurrence of each unique combination and assign a row number to it
unique_combinations = df.select("Region", "State", "Purpose", "id_temp").distinct()
unique_combinations = unique_combinations.withColumn("ts_id", row_number().over(Window.orderBy("id_temp"))).drop("id_temp")

# Join back to get the new unique ID for each row
df = df.join(unique_combinations, on=["Region", "State", "Purpose"], how="left").drop("id_temp")

# Select the required columns
df = df.select("ts_id", "date", "total", "Region", "State", "Purpose").orderBy("ts_id", "date")

# Display the result
df.display()

# COMMAND ----------
###############################################################################
# Overview of the pre-processed data
###############################################################################

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Number of time series
unique_ts_id_count = df.select("ts_id").distinct().count()
print(f"Number of time series (unique ts_id): {unique_ts_id_count}")

# Length of time series
time_series_length = dfP.groupby("ts_id")["date"].nunique().max()  # Number of unique dates per ts_id
print(f"Length of a time series (number of observations): {time_series_length}")

# COMMAND ----------
###############################################################################
# Save the final DataFrame back to the same database and table
###############################################################################
# Important: Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables

# DELETE THE OLD TABLE FIRST
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`tourism`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`tourism`")
# Redefine the DataFrame to ensure it matches the latest schema

# Convert the Spark DataFrame to a Pandas DataFrame
pandas_df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
pandas_df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/tourism.csv', index=False)
