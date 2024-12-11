# Databricks notebook source
# MAGIC %md
# MAGIC Data preparation: The original dataset is modified, and the result is saved in the dataset folder. Therefore, the following steps are not required to reproduce the results.

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, monotonically_increasing_id, dense_rank, row_number
from pyspark.sql.window import Window

# Load M5
australian_labour_market = spark.sql("""
    SELECT * 
    FROM `analytics`.`p&l_prediction`.`australian_labour_market_original`
""")

australian_labour_market.display()
print(australian_labour_market.select("`Sex`", "`State and territory (STT): ASGS (2011)`", "`Occupation of main job: ANZSCO (2013) v1.2`").distinct().count())

# COMMAND ----------

# DBTITLE 1,Overview dataset
columns = ["`Sex`", "`State and territory (STT): ASGS (2011)`", "`Occupation of main job: ANZSCO (2013) v1.2`"]
df_original = australian_labour_market

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

from pyspark.sql.functions import to_date, concat_ws, last_day, regexp_extract, lpad, lit
from pyspark.sql.functions import regexp_extract
from pyspark.sql import functions as F

df = australian_labour_market

# Extract the month and year from the "Mid-quarter month" column
df = df.withColumn("month", regexp_extract(df["Mid-quarter month"], r"([A-Za-z]+)", 1))
df = df.withColumn("year", regexp_extract(df["Mid-quarter month"], r"(\d+)", 0))

# Convert the month to a numerical representation (e.g., "Jan" -> "01")
months = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}
df = df.replace(months, subset=["month"])

# Combine year and month into a date string in the format "YYYY-MM-01"
df = df.withColumn("date_temp", concat_ws("-", df["year"], df["month"], lpad(lit("01"), 2, "0")))

# Convert the resulting column to a date format and calculate the last day of the month
df = df.withColumn("date", last_day(to_date(df["date_temp"], "yyyy-MM-dd")))

# Rename the column
df = df.withColumnRenamed("Occupation of main job: ANZSCO (2013) v1.2", "ANZSCO")

# Extract the first 4, 3, 2, and 1 digits from the "ANZSCO" column
from pyspark.sql.functions import regexp_extract, col, regexp_replace
from pyspark.sql.types import FloatType

# Rename the column and convert it to a numeric type (e.g., Float) and multiply by 1000
df = df.withColumnRenamed("Employed total ('000)", "total")
df = df.withColumn("total", regexp_replace(col("total"), ",", "."))
df = df.withColumn("total", (col("total").cast(FloatType()) * 1000))

# Extract the first 4, 3, 2, and 1 digits from the ANZSCO column as a string
df = df.withColumn(
    "hier_4",
    F.concat(
        F.regexp_extract(df["ANZSCO"], r"^(\d{4})", 1),
        F.lit("D")  # Add the desired string here
    )
)
df = df.withColumn(
    "hier_3",
    F.concat(
        F.regexp_extract(df["ANZSCO"], r"^(\d{3})", 1),
        F.lit("C")  # Add the desired string here
    )
)
df = df.withColumn(
    "hier_2",
    F.concat(
        F.regexp_extract(df["ANZSCO"], r"^(\d{2})", 1),
        F.lit("B")  # Add the desired string here
    )
)
df = df.withColumn(
    "hier_1",
    F.concat(
        F.regexp_extract(df["ANZSCO"], r"^(\d{1})", 1),
        F.lit("A")  # Add the desired string here
    )
)

# Remove the temporary columns
df = df.drop("date_temp", "month", "year", "Number of hours actually worked in all jobs ('000 Hours)", "Mid-quarter month", "ANZSCO")

# Rename
df = df.withColumnRenamed("State and territory (STT): ASGS (2011)", "state_and_territory")

df = df.groupBy("Sex", "state_and_territory", "hier_1", "date") \
                       .agg(F.sum("total").alias("total"))

from pyspark.sql import functions as F

# Assuming the column is named 'date' and is in the format 'yyyy-MM-dd'
df = df.withColumn("date", F.expr("last_day(add_months(date, 1))"))

# Define window specification for calculating the continuous ts_id based on the grouping variables
window_spec = Window.orderBy("Sex", "state_and_territory", "hier_1")

# Calculate the continuous ts_id for each unique combination of grouping variables
df = df.withColumn("ts_id", F.dense_rank().over(window_spec))

# Adjust column order: "ts_id", "date", "total" and then the grouping variables
df = df.select("ts_id", "date", "total", 
                              "Sex", "state_and_territory", "hier_1")

# Sort by ts_id and date
df = df.orderBy("ts_id", "date")

# Display the result
print(df)

# COMMAND ----------

# DBTITLE 1,Info for paper
print(df.select("state_and_territory").distinct().count())
dfP = df.toPandas()
print(f"Date range: {dfP['date'].min()} to {dfP['date'].max()}")

# Number of time series
unique_ts_id_count = df.select("ts_id").distinct().count()
print(f"Number of time series (unique ts_id): {unique_ts_id_count}")

# Length of time series
time_series_length = dfP.groupby("ts_id")["date"].nunique().max()  # Number of unique dates per ts_id
print(f"Length of a time series (number of observations): {time_series_length}")

columns = ["`Sex`", "`state_and_territory`", "`hier_1`"] #  "`Country`", is a constant
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

################################################## Important #####################################################
# Note that in the current Version all Variables beside ts_id, date and total will be used as grouping variables
################################################## Important #####################################################
df.display()

# COMMAND ----------

# Save the final DataFrame back to the same database and table
# DELETE THE OLD TABLE FIRST
spark.sql("DROP TABLE IF EXISTS `analytics`.`p&l_prediction`.`australian_labour_market`")
df.write.mode("overwrite").saveAsTable("`analytics`.`p&l_prediction`.`australian_labour_market`")
# Redefine the DataFrame to ensure it matches the latest schema

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
df = df.toPandas()

# Save the Pandas DataFrame as a CSV file
df.to_csv('/Workspace/Users/simon.pradel@telefonicatgt.es/p-l_prediction/P&L_Prediction/data/datasets/australian_labour_market.csv', index=False)
