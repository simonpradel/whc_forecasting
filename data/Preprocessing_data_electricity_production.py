# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, monotonically_increasing_id, dense_rank, row_number
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
    
# Load the M5 dataset
global_electricity_production = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`global_electricity_production_original`
""")

# Convert the Spark DataFrame to a Pandas DataFrame
global_electricity_production_original = global_electricity_production.toPandas()

global_electricity_production.display()
print(global_electricity_production.select("`country_name`", "`parameter`", "`product`", "unit").distinct().count())

# COMMAND ----------

columns = ["`country_name`", "`parameter`", "`product`"] # "unit" = constant 
df_original = global_electricity_production

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
from pyspark.sql import types as T
from pyspark.sql import Window

# Helper function to format the date string
def format_date_column(date_str):
    parts = date_str.split("/")
    
    # Month and day must be two digits
    if len(parts[0]) == 1:
        parts[0] = "0" + parts[0]  # Month
    if len(parts[1]) == 1:
        parts[1] = "0" + parts[1]  # Day
        
    # Assemble the formatted date in the format MM/dd/yyyy
    return "/".join(parts)

# User-defined function (UDF) for applying to the column
format_date_udf = F.udf(format_date_column, T.StringType())

# First format the date column by applying the UDF
df = global_electricity_production.withColumn(
    "formatted_date", 
    format_date_udf(F.col("date"))
)

# Then: Convert the "formatted_date" column to the correct date format
df = df.withColumn(
    "date",
    F.to_date(F.col("formatted_date"), "MM/dd/yyyy")
)

# Ensure that the last day of the month is always taken
df = df.withColumn(
    "date",
    F.last_day(F.col("date"))
)

# Delete the temporary "formatted_date" column
df = df.drop("formatted_date")

# Convert the "Order Date" column to the last day of the month
df = df.withColumnRenamed("value", "total")

# Grouping and aggregation step to calculate "total_sales"
df = df.groupBy("parameter", "product", "country_name", "date") \
                       .agg(F.sum("total").alias("total"))

# Window definition for calculating the sequential ts_id based on the grouping variables
window_spec = Window.orderBy("parameter", "product", "country_name")

# Calculation of the sequential ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", 
                              "parameter", "product", "country_name")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Output the DataFrame
df.display()

print(df.select("ts_id").distinct().count())
print(df.select("date").distinct().count())
df.groupBy('ts_id').count().show()

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

row_count = df.count()
print(f"Number of rows in the DataFrame: {row_count}")

# COMMAND ----------

# DBTITLE 1,Info for Paper
dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Number of time series
unique_ts_id_count = df.select("ts_id").distinct().count()
print(f"Number of time series (unique ts_id): {unique_ts_id_count}")

# Length of time series
time_series_length = dfP.groupby("ts_id")["date"].nunique().max()  # Number of unique dates per ts_id
print(f"Length of a time series (number of observations): {time_series_length}")

# COMMAND ----------

################################################## Important #####################################################
# Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables
################################################## Important #####################################################
df.display()

# COMMAND ----------

# Save the final DataFrame back to the same database and table
# DELETE THE OLD TABLE FIRST
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`global_electricity_production`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`global_electricity_production`")
# Redefine the DataFrame to ensure it matches the latest schema

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/global_electricity_production.csv', index=False)
