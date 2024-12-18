# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------
###############################################################################
# Load Data
###############################################################################
from itertools import chain, combinations
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import last_day
from pyspark.sql.functions import concat, lit

spark = SparkSession.builder.getOrCreate()

# Load the M5 dataset
store_item_demand = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`store_item_demand_original`
""")

# Convert the Spark DataFrame to a Pandas DataFrame
store_item_demand_original = store_item_demand.toPandas()

# Display the DataFrame
store_item_demand.display()
print(store_item_demand.select("`store`", "`item`").distinct().count())

# COMMAND ----------
###############################################################################
# Overview of the original data
###############################################################################

columns = ["`store`", "`item`"] 
df_original = store_item_demand

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

# Convert "Order Date" column to the last day of the month
df = store_item_demand.withColumnRenamed("sales", "total")

# The file would be too large, therefore make monthly data out of daily data
df = df.withColumn("date", last_day("date"))

# Grouping and aggregation step to calculate "total_sales"
df = df.groupBy("store", "item", "date") \
                       .agg(F.sum("total").alias("total"))

# Window definition for calculating the continuous ts_id based on the grouping variables
window_spec = Window.orderBy("store", "item")

# Calculate the continuous ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", "store", "item")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Convert the columns 'brand' and 'item' to string data type
df = df.withColumn("item", col("item").cast("string"))
df = df.withColumn("store", col("store").cast("string"))


# Add an 'S' to the 'store' column and convert to string
df = df.withColumn("store", concat(lit("S"), col("store").cast("string")))
df = df.withColumn("item", concat(lit("I"), col("item").cast("string")))

# Output the DataFrame
df.display()

print(df.count())
print(df.select("ts_id").distinct().count())
print(df.select("date").distinct().count())
df.groupBy('ts_id').count().show()

# Examine the number of unique values in the string columns
string_columns = ['store', 'item']

for col in string_columns:
    distinct_count = df.select(col).distinct().count()
    print(f"Column '{col}' has {distinct_count} unique values.")

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
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`store_item_demand`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`store_item_demand`")
# Redefine the DataFrame to ensure it matches the latest schema

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/store_item_demand.csv', index=False)
