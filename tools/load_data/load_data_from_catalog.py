import os
import sys
from pyspark.sql.functions import lit
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType



def load_data_from_catalog(data, datasetName=None, maindir = None):
    print("hi")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    if maindir == None:
        maindir = os.getcwd()
        print(maindir)

    if data == "M5":
        current_dir = os.path.join(maindir, "data/datasets/m5_sales_long.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'m_5_sales_long': df}

        freq = "D"

        # Target column name (assumes the last column is the target)
        target_column_name = "total"

    elif data == "australian_labour_market":
        current_dir = os.path.join(maindir, "data/datasets/australian_labour_market.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'australian_labour_market': df}

        freq = "QE"

        # Target column name (assumes the last column is the target)
        target_column_name = "total"

    elif data == "prison_population":
        current_dir = os.path.join(maindir, "data/datasets/prison_population.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'prison_population': df}

        freq = "QE"

        # Target column name (assumes the last column is the target)
        target_column_name = "total"

    elif data == "retail_prices":
        current_dir = os.path.join(maindir, "data/datasets/retail_prices.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'retail_prices': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "total"

    elif data == "natural_gas_usage":
        current_dir = os.path.join(maindir, "data/datasets/natural_gas_usage.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'natural_gas_usage': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "total"
        
    elif data == "tourism":
        current_dir = os.path.join(maindir, "data/datasets/tourism.csv")
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        # Add the new column with constant value 'M5'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'tourism': df}

        target_column_name = "total"

        freq = "QE"

    elif data == "global_electricity_production":

        # CSV-Datei aus dem Pfad "data/" laden
        current_dir = os.path.join(maindir, "data/datasets/global_electricity_production.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        # Add the new column with constant value 'superstore'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'global_electricity_production': df}

        target_column_name = "total"
        
        freq = "ME"

    elif data == "superstore":

        # CSV-Datei aus dem Pfad "data/" laden
        current_dir = os.path.join(maindir, "data/datasets/superstore.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)
        


        # Add the new column with constant value 'superstore'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'superstore': df}

        target_column_name = "total"
        
        freq = "ME"

    elif data == "italian_grocery_store":

        # CSV-Datei aus dem Pfad "data/" laden
        current_dir = os.path.join(maindir, "data/datasets/italian_grocery_store.csv")
        df = pd.read_csv(current_dir)


        df = spark.createDataFrame(df)

        # Add the new column with constant value 'superstore'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'italian_grocery_store': df}

        target_column_name = "total"
        
        freq = "D"

    elif data == "store_item_demand":

        current_dir = os.path.join(maindir, "data/datasets/store_item_demand.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        # Add the new column with constant value 'superstore'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'store_item_demand': df}

        target_column_name = "total"
        
        freq = "ME"
        
    elif data == "website_traffic":

        current_dir = os.path.join(maindir, "data/datasets/website_traffic.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        # Add the new column with constant value 'superstore'
        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'website_traffic': df}

        target_column_name = "total"
        
        freq = "D"

    elif data == "test":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`test`
        """)
        test = "Telefonica - mobile_service_revenue"
        df = df.withColumn("dataset", lit(test))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'test': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - mobile_service_revenue":
  
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`mobile_service_revenue`
        """)

        df = df.withColumn("dataset", lit(data))
 
        # Dictionary of datasets with their corresponding names
        datasets = {'mobile_service_revenue': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - hardware_revenue":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`hardware_revenue`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'hardware_revenue': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - fbb_fixed_other_revenue":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`fbb_fixed_other_revenue`
        """)

        df = df.withColumn("dataset", lit(data))
        
        # Dictionary of datasets with their corresponding names
        datasets = {'fbb_fixed_other_revenue': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - cos":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`cos`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'cos': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - commercial_costs":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`commercial_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'commercial_costs': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - non_commercial_costs":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`non_commercial_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'non_commercial_costs': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

    elif data == "Telefonica - non_recurrent_income_cost":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`non_recurrent_income_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'non_recurrent_income_cost': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

        # Add the new column with constant value 'M5'
        df = df.withColumn("dataset", lit(data))


    elif data == "Telefonica - bad_debt":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`bad_debt`
        """)

        df = df.withColumn("dataset", lit(data))

        # Dictionary of datasets with their corresponding names
        datasets = {'bad_debt': df}

        freq = "ME"

        # Target column name (assumes the last column is the target)
        target_column_name = "Amount"

        # Add the new column with constant value 'M5'
        df = df.withColumn("dataset", lit(data))

    else:
        raise ValueError(f"Unknown data value: {data}")

    # Use the specified datasetName or default to the first dataset
    if datasetName and datasetName in datasets:
        selected_dataset = datasets[datasetName]
        selected_name = datasetName
    else:
        selected_name = list(datasets.keys())[0]
        selected_dataset = datasets[selected_name]



    # Funktion zur Konvertierung der Datentypen
    def convert_column_types(df):
        if isinstance(df, pd.DataFrame):
            # Pandas DataFrame
            if 'ts_id' in df.columns:
                df['ts_id'] = df['ts_id'].astype(int)

            if 'total' in df.columns:
                df['total'] = df['total'].astype(float)

            if 'Amount' in df.columns:
                df['Amount'] = df['Amount'].astype(float)


        elif isinstance(df, SparkDataFrame):
            # PySpark DataFrame
            if 'ts_id' in df.columns:
                df = df.withColumn('ts_id', df['ts_id'].cast(IntegerType()))

            if 'total' in df.columns:
                df = df.withColumn('total', df['total'].cast(FloatType()))

            if 'Amount' in df.columns:
                df = df.withColumn('Amount', df['Amount'].cast(FloatType()))


        return df
    
    # Konvertiere die Datentypen der relevanten Spalten
    selected_dataset = convert_column_types(selected_dataset)

    # Extract grouping variables (columns excluding 'date' and 'total')
    grouping_variables = [col for col in selected_dataset.columns if col not in ['ts_id', 'date', 'total', "Year", "Period", "Amount"]]

    # General dataset name
    datasetNameGeneral = data

    # Return the dictionary with the relevant information
    return {
        'datasetNameGeneral': datasetNameGeneral,
        'dataframe_name': selected_name,
        'target_column_name': target_column_name,
        'grouping_variables': grouping_variables,
        'freq': freq,
        'original_dataset': selected_dataset
    }


