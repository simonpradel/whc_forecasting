# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, monotonically_increasing_id, dense_rank, row_number
from pyspark.sql.window import Window

# Load the M5 dataset
website_traffic = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`website_traffic_original`
""")

# Convert the Spark DataFrame to a Pandas DataFrame
website_traffic_original = website_traffic.toPandas()

# Save the Pandas DataFrame as a CSV file
# website_traffic_original.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/original_datasets/website_traffic_original.csv', index=False)

website_traffic.display()

print(website_traffic.select("`Device Category`", "`Browser`").distinct().count())


# COMMAND ----------

columns = ["`Device Category`", "`Browser`"] #  "`Country`", is a constant
df_original = website_traffic

from itertools import chain, combinations

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

from pyspark.sql import functions as F
from pyspark.sql import Window

# Convert "Order Date" column to the last day of the month
df = website_traffic.withColumnRenamed("Date", "date")
df = df.withColumnRenamed("Device Category", "DeviceCategory")
df = df.withColumnRenamed("# of Visitors", "total")

# Grouping and aggregation step to calculate "total_sales"
df = df.groupBy("DeviceCategory", "Browser", "date") \
                       .agg(F.sum("total").alias("total"))

# Window definition for calculating the continuous ts_id based on the grouping variables
window_spec = Window.orderBy("DeviceCategory", "Browser")

# Calculate the continuous ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", 
                               "DeviceCategory", "Browser")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Get ts_ids with count < 1032
small_ts_ids = df.groupBy('ts_id').count().filter('count < 1000').select('ts_id').rdd.flatMap(lambda x: x).collect()

# Define a function to update column values based on the condition
from pyspark.sql.functions import when

# Apply the changes to "DeviceCategory" and "Browser" for ts_id with less than 1000 observations
df = df.withColumn(
    "DeviceCategory",
    when(df["ts_id"].isin(small_ts_ids), "DeviceCategoryOther").otherwise(df["DeviceCategory"])
).withColumn(
    "Browser",
    when(df["ts_id"].isin(small_ts_ids), "BrowserOther").otherwise(df["Browser"])
)

# Grouping and aggregation step to calculate "total_sales"
df = df.groupBy("DeviceCategory", "Browser", "date") \
                       .agg(F.sum("total").alias("total"))

# Calculate the continuous ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", 
                               "DeviceCategory", "Browser")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Output the DataFrame
df.display()

print(df.count())
print(df.select("ts_id").distinct().count())
print(df.select("date").distinct().count())
df.groupBy('ts_id').count().show()

# COMMAND ----------

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Number of time series
unique_ts_id_count = df.select("ts_id").distinct().count()
print(f"Number of time series (unique ts_id): {unique_ts_id_count}")

# Length of time series
time_series_length = dfP.groupby("ts_id")["date"].nunique().max()  # Number of unique dates per ts_id
print(f"Length of a time series (number of observations): {time_series_length}")

columns = ["DeviceCategory", "Browser"] #  "`Country`", is a constant
df_original = df

from itertools import chain, combinations

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

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Number of time series
unique_ts_id_count = df.select("ts_id").distinct().count()
print(f"Number of time series (unique ts_id): {unique_ts_id_count}")

# Length of time series
time_series_length = dfP.groupby("ts_id")["date"].nunique().max()  # Number of unique dates per ts_id
print(f"Length of a time series (number of observations): {time_series_length}")

# COMMAND ----------

df.display()

# COMMAND ----------

################################################## Important #####################################################
# Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables
################################################## Important #####################################################
from pyspark.sql.functions import col
from pyspark.sql.functions import to_date
from datetime import datetime
df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
df.display()



# COMMAND ----------

# Save the final DataFrame back to the same database and table
# DELETE THE OLD TABLE FIRST
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`website_traffic`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`website_traffic`")


# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/website_traffic.csv', index=False)
