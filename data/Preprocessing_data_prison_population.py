# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the M5 dataset
prison_population = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`prison_population`
""")

# Convert the Spark DataFrame to a Pandas DataFrame
prison_population_original = prison_population.toPandas()

prison_population.display()
print(prison_population.select("`State`", "`Gender`", "`Legal`", "`Indigenous`").distinct().count())

# COMMAND ----------

columns = ["`State`", "`Gender`", "`Legal`", "`Indigenous`"] # "['`product`']" = constant 
df_original = prison_population

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
df = prison_population.withColumnRenamed("Date", "date")
df = df.withColumnRenamed("Count", "total")

# Grouping and aggregation step to calculate "total_sales"
df = df.groupBy("State", "Gender", "Legal", "Indigenous", "date") \
                       .agg(F.sum("total").alias("total"))

# Window definition for calculating the running ts_id based on the grouping variables
window_spec = Window.orderBy("State", "Gender", "Legal", "Indigenous")

# Calculate the running ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", 
                               "State", "Gender", "Legal", "Indigenous")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

from pyspark.sql.functions import last_day

# Set the date to the end of the month
df = df.withColumn('date', last_day(df['date']))

# Display the updated DataFrame
df.display()

print(df.count())
print(df.select("ts_id").distinct().count())
print(df.select("date").distinct().count())
df.groupBy('ts_id').count().show()

dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

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

################################################## Important #####################################################
# Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables
################################################## Important #####################################################
df.display()

# COMMAND ----------

# Save the final DataFrame back to the same database and table
# DELETE THE OLD TABLE FIRST
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`prison_population`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`prison_population`")
# Redefine the DataFrame to ensure it matches the latest schema

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/prison_population.csv', index=False)
