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
from pyspark.sql.functions import count, when
from itertools import chain, combinations
from pyspark.sql.functions import last_day, to_date
from pyspark.sql import functions as F
from pyspark.sql.functions import regexp_replace

spark = SparkSession.builder.getOrCreate()

# Load the M5 dataset
monthly_food_retail_prices = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`monthly_food_retail_prices_original`
""")

# Number of rows in the DataFrame
row_count = monthly_food_retail_prices.count()
print("Number of rows:", row_count)

monthly_food_retail_prices.display()
print(monthly_food_retail_prices.select("`State`", "`Centre`", "`Commodity`", "`Variety`", "`Unit`", "`Category`").distinct().count())


# Calculate the number of missing values in the "Retail Price" column
missing_values_count = monthly_food_retail_prices.filter(col("Retail Price").isNull()).count()

# Output the number of missing values
print("Number of missing values in Retail Price:", missing_values_count)

# COMMAND ----------
###############################################################################
# Overview of the original data
###############################################################################

columns = ["`State`", "`Centre`", "`Commodity`", "`Variety`", "`Unit`"] # "['`Category`']" = constant 
df_original = monthly_food_retail_prices


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

# Check distributions: calculate the number of entries and missing values for each combination of State, Centre, and Category
aggregated_data = (
    monthly_food_retail_prices
    .groupBy("State")
    .agg(
        count("*").alias("total_count"),
        count(when(col("Retail Price").isNull(), 1)).alias("missing_count")
    )
)

print(aggregated_data)
# COMMAND ----------
###############################################################################
# Preparation of the data
###############################################################################

# Delete all rows where the value in the "Retail Price" column is null
df = monthly_food_retail_prices.filter(col("Retail Price").isNotNull())

df = df.withColumnRenamed("Date", "date")

# Replace all other null values in the remaining columns with 'others'
df = df.withColumn("Variety", when(col("Variety").isNull(), "othersVarietys").otherwise(col("Variety")))

# Replace null values in 'Commodity'
df = df.withColumn("Commodity", when(col("Commodity").isNull(), "OtherCommoditys").otherwise(col("Commodity")))

# Replace null values in 'State'
df = df.withColumn("State", when(col("State").isNull(), "OtherStates").otherwise(col("State")))

# Delete the columns "Food" and "Unit"
df = df.drop("Category", "Unit")

# Convert the 'Date' column to a date format and set the date to the last day of the month
df = df.withColumn("date", last_day(to_date(col("date"), "MMM-yyyy")))

df = df.withColumnRenamed("Retail Price", "total")

df = df.groupBy("State", "Commodity", "date") \
                       .agg(F.sum("total").alias("total"))

# Window definition for calculating the running ts_id, based on the grouping variables
window_spec = Window.orderBy("State", "Commodity")

# Calculate the running ts_id for each unique combination of the grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", "State", "Commodity")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

row_count = df.count()
print(f"Number of rows in the DataFrame: {row_count}")

# Examine the number of unique values in the string columns
string_columns = ['State', 'Commodity']

for col in string_columns:
    distinct_count = df.select(col).distinct().count()
    print(f"Column '{col}' has {distinct_count} unique values.")

# Replace "/" with "_" in a specific column (e.g., 'ColumnName')
df = df.withColumn('Commodity', regexp_replace('Commodity', '/', '_'))
df = df.withColumn('State', regexp_replace('State', '/', '_'))

# Filter by date
df = df.filter(col("date") > "2011-01-01")

# Display the result
df.display()

# COMMAND ----------
###############################################################################
# Overview of the pre-processed data
###############################################################################

columns = ["State", "Commodity"] #  "`Country`", is a constant
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

# Filter DataFrame by "west"
filtered_df = df.filter(df.State.isin(["West Bengal"]))  # Example for western states
filtered_df = filtered_df.filter(filtered_df.date.isin(["2020-10-31"]))  # Example for western states

filtered_df.display()

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
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`retail_prices`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`retail_prices`")
# Redefine the DataFrame to ensure it matches the latest schema

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/retail_prices.csv', index=False)
