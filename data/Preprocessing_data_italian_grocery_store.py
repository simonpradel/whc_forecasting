# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------
###############################################################################
# Load Data
###############################################################################

from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit
from pyspark.sql import functions as F
from pyspark.sql import types as T
import re
from itertools import chain, combinations

spark = SparkSession.builder.getOrCreate()

# Load the M5 dataset
italian_grocery_store = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`italian_grocery_store_original`
""")

# Convert the Spark DataFrame to a Pandas DataFrame
italian_grocery_store_original = italian_grocery_store.toPandas()

# Display the Spark DataFrame
italian_grocery_store.display()

# COMMAND ----------
###############################################################################
# Preparation of the data
###############################################################################

df = italian_grocery_store

# Get all QTY column names
qty_columns = [col for col in df.columns if col.startswith('QTY_')]

# Create a list of StructType containing brand and item information
def extract_brand_item(col_name):
    match = re.match(r"QTY_B(\d+)_(\d+)", col_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return (None, None)

# UDF to apply on column names
extract_brand_item_udf = F.udf(lambda col_name: extract_brand_item(col_name), T.StructType([
    T.StructField("brand", T.IntegerType(), True),
    T.StructField("item", T.IntegerType(), True)
]))

# Long format transformation
df = df.select(
    F.col("date"), 
    F.explode(F.array([F.struct(F.lit(c).alias("col_name"), F.col(c).alias("total")) for c in qty_columns])).alias("exploded")
).select(
    "date",
    extract_brand_item_udf(F.col("exploded.col_name")).getItem("brand").alias("brand"),
    extract_brand_item_udf(F.col("exploded.col_name")).getItem("item").alias("item"),
    F.col("exploded.total").alias("total")
)

# Window definition for calculating the continuous ts_id based on grouping variables
window_spec = Window.orderBy("brand", "item")

# Calculate the continuous ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", "brand", "item")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Convert 'brand' and 'item' columns to string data type
df = df.withColumn("brand", col("brand").cast("string"))
df = df.withColumn("item", col("item").cast("string"))


# Add 'B' to the 'brand' column and convert to string
df = df.withColumn("brand", concat(lit("B"), col("brand").cast("string")))
# Add 'I' to the 'item' column and convert to string
df = df.withColumn("item", concat(lit("I"), col("item").cast("string")))

# Output the DataFrame
df.display()

print(df.count())
print(df.select("ts_id").distinct().count())
print(df.select("date").distinct().count())
print(df.select("brand").distinct().count())
df.groupBy('ts_id').count().show()

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Examine the number of unique values in the string columns
string_columns = ['brand', 'item']

for col in string_columns:
    distinct_count = df.select(col).distinct().count()
    print(f"Column '{col}' has {distinct_count} unique values.")

# COMMAND ----------
###############################################################################
# Overview of the pre-processed data
###############################################################################

columns = ["`brand`", "`item`"] # "unit" = constant 
df_original = df

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
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`italian_grocery_store`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`italian_grocery_store`")
# Redefine the DataFrame to ensure it matches the latest schema

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/italian_grocery_store.csv', index=False)
