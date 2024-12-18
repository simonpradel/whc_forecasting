# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------
###############################################################################
# Load Data
###############################################################################

from pyspark.sql.functions import col, expr, lit, array, date_add, sum as spark_sum, row_number, dense_rank
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from itertools import chain, combinations

spark = SparkSession.builder.getOrCreate()

# Load the M5 dataset
m_5_sales = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`m_5_sales_train_evaluation`
""")

m_5_sell_prices = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`m_5_sell_prices`
""")

m_5_calendar = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`m_5_calendar`
""")

m_5_sales = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`m_5_sales_train_evaluation`
""")

print(m_5_sales.select("`item_id`", "`dept_id`", "`cat_id`", "`store_id`", "`state_id`").distinct().count())

# COMMAND ----------
###############################################################################
# Preparation of the data
###############################################################################

# Convert the Spark DataFrame to a Pandas DataFrame
m_5_sales_original = m_5_sales.toPandas()
m_5_sell_prices_original = m_5_sell_prices.toPandas()
m_5_calendar_original = m_5_calendar.toPandas()

# List of day columns
days_columns = [f"d_{i}" for i in range(1, 1942)]

# Create an array of day columns
m_5_sales = m_5_sales.withColumn("days_array", array([col(c).cast(IntegerType()) for c in days_columns]))

# List of other columns
other_columns = [col for col in m_5_sales.columns if col not in days_columns and col != "id"]

# Explode the array and duplicate the other columns
df = m_5_sales.select(
    *other_columns,
    expr("posexplode(days_array)").alias("day_index", "sales")
)

# Create the date
start_date = "2011-01-29"
df = df.withColumn("date", date_add(lit(start_date), col("day_index")))

# Select the required columns and sort by ID and date
df = df.select(
    "date",
    "sales",
    *other_columns
).orderBy("id", "date")

df = df.drop("days_array") 
other_columns.remove('days_array')

# Join df with m_5_calendar to get wm_yr_wk
df = df.join(m_5_calendar.select("date", "wm_yr_wk"), on="date", how="left")

# Join df with m_5_sell_prices to get sell_price
df = df.join(
    m_5_sell_prices.select("store_id", "item_id", "wm_yr_wk", "sell_price"),
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

# Calculate the revenue
df = df.withColumn("sales", col("sales").cast("double") * col("sell_price"))

df.display()

# Select the required columns
other_columns.remove('item_id')
df = df.select("date", "sales", *other_columns)

df.display()

# DBTITLE 1,Aggregate sales
# Sum the sales by id and date
df = df.groupBy("state_id", "store_id", "dept_id", "cat_id", "date").agg(
    spark_sum("sales").alias("total")
)

# Generate a time series ID
# Window spec to partition by unique combination and order by state_id, store_id, dept_id, cat_id
window_spec = Window.partitionBy("state_id", "store_id", "dept_id", "cat_id").orderBy("state_id", "store_id", "dept_id", "cat_id")

# Assign a unique ID to each combination
df = df.withColumn("id_temp", dense_rank().over(window_spec))

# Retrieve the first occurrence of each unique combination and assign it a row number
unique_combinations = df.select("state_id", "store_id", "dept_id", "cat_id", "id_temp").distinct()
unique_combinations = unique_combinations.withColumn("ts_id", row_number().over(Window.orderBy("id_temp"))).drop("id_temp")

# Join back to get the new unique ID for each row
df = df.join(unique_combinations, on=["state_id", "store_id", "dept_id", "cat_id"], how="left").drop("id_temp")

# Select the required columns
df = df.select("ts_id", "date", "total", "state_id", "store_id", "cat_id", "dept_id").orderBy("ts_id", "date")

# Display the result
df.display()

# COMMAND ----------
###############################################################################
# Overview of the pre-processed data
###############################################################################

columns = ["`state_id`", "`store_id`", "`dept_id`", "`cat_id`", "`item_id`"] # "unit" = constant 
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

columns = ["state_id", "store_id", "dept_id", "cat_id"] #  "`Country`", is a constant

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
# Save the final DataFrame back to the same database and table
###############################################################################
# Important: Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables

# DELETE THE OLD TABLE FIRST
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`m_5_sales_long`")

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/m5_sales_long.csv', index=False)