import os
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import SparkSession

def load_data_from_catalog(data, datasetName=None, maindir=None):
    """
    Load a dataset from a predefined catalog based on the specified `data` parameter.

    Parameters:
        data (str): Identifier for the dataset to load. Valid values include:
            "M5", "australian_labour_market", "prison_population", "retail_prices",
            "natural_gas_usage", "tourism", "global_electricity_production",
            "superstore", "italian_grocery_store", "store_item_demand", "website_traffic",
            "test", "Telefonica - mobile_service_revenue", "Telefonica - hardware_revenue",
            "Telefonica - fbb_fixed_other_revenue", "Telefonica - cos",
            "Telefonica - commercial_costs", "Telefonica - non_commercial_costs",
            "Telefonica - non_recurrent_income_cost", "Telefonica - bad_debt".

        datasetName (str, optional): Specific name of the dataset to select from the loaded data.
            Defaults to the first dataset in the dictionary.

        maindir (str, optional): Base directory for locating dataset files.
            Defaults to the current working directory if not specified.

    Returns:
        dict: A dictionary containing the following keys:
            - 'datasetNameGeneral' (str): General identifier for the dataset.
            - 'dataframe_name' (str): Name of the selected dataset.
            - 'target_column_name' (str): Name of the target column in the dataset.
            - 'grouping_variables' (list): Columns to be used for grouping (excluding date and target columns).
            - 'freq' (str): Frequency of the data (e.g., "D", "QE", "ME").
            - 'original_dataset' (SparkDataFrame or pd.DataFrame): The loaded dataset.
    """
    spark = SparkSession.builder.getOrCreate()
    if maindir == None:
        maindir = os.getcwd()
        print(maindir)

    if data == "M5":
        current_dir = os.path.join(maindir, "data/datasets/m5_sales_long.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        datasets = {'m_5_sales_long': df}

        freq = "D"

        target_column_name = "total"

    elif data == "australian_labour_market":
        current_dir = os.path.join(maindir, "data/datasets/australian_labour_market.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'australian_labour_market': df}

        freq = "QE"

        target_column_name = "total"

    elif data == "prison_population":
        current_dir = os.path.join(maindir, "data/datasets/prison_population.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'prison_population': df}

        freq = "QE"

        target_column_name = "total"

    elif data == "retail_prices":
        current_dir = os.path.join(maindir, "data/datasets/retail_prices.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'retail_prices': df}

        freq = "ME"

        target_column_name = "total"

    elif data == "natural_gas_usage":
        current_dir = os.path.join(maindir, "data/datasets/natural_gas_usage.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))
        
        datasets = {'natural_gas_usage': df}

        freq = "ME"

        target_column_name = "total"
        
    elif data == "tourism":
        current_dir = os.path.join(maindir, "data/datasets/tourism.csv")
        
        df = pd.read_csv(current_dir)
        
        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'tourism': df}

        target_column_name = "total"

        freq = "QE"

    elif data == "global_electricity_production":
        current_dir = os.path.join(maindir, "data/datasets/global_electricity_production.csv")
        
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'global_electricity_production': df}

        target_column_name = "total"
        
        freq = "ME"

    elif data == "superstore":
        current_dir = os.path.join(maindir, "data/datasets/superstore.csv")
        
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'superstore': df}

        target_column_name = "total"
        
        freq = "ME"

    elif data == "italian_grocery_store":
        current_dir = os.path.join(maindir, "data/datasets/italian_grocery_store.csv")
        
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'italian_grocery_store': df}

        target_column_name = "total"
        
        freq = "D"

    elif data == "store_item_demand":
        current_dir = os.path.join(maindir, "data/datasets/store_item_demand.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'store_item_demand': df}

        target_column_name = "total"
        
        freq = "ME"
        
    elif data == "website_traffic":
        current_dir = os.path.join(maindir, "data/datasets/website_traffic.csv")
        df = pd.read_csv(current_dir)

        df = spark.createDataFrame(df)

        df = df.withColumn("dataset", lit(data))

        datasets = {'website_traffic': df}

        target_column_name = "total"
        
        freq = "D"

    elif data == "Telefonica - mobile_service_revenue":
  
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`mobile_service_revenue`
        """)

        df = df.withColumn("dataset", lit(data))
 
        datasets = {'mobile_service_revenue': df}

        freq = "ME"

        target_column_name = "Amount"

    elif data == "Telefonica - hardware_revenue":

        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`hardware_revenue`
        """)

        df = df.withColumn("dataset", lit(data))

        datasets = {'hardware_revenue': df}

        freq = "ME"

        target_column_name = "Amount"

    elif data == "Telefonica - fbb_fixed_other_revenue":

        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`fbb_fixed_other_revenue`
        """)

        df = df.withColumn("dataset", lit(data))
        
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

        datasets = {'cos': df}

        freq = "ME"

        target_column_name = "Amount"

    elif data == "Telefonica - commercial_costs":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`commercial_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        datasets = {'commercial_costs': df}

        freq = "ME"

        target_column_name = "Amount"

    elif data == "Telefonica - non_commercial_costs":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`non_commercial_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        datasets = {'non_commercial_costs': df}

        freq = "ME"

        target_column_name = "Amount"

    elif data == "Telefonica - non_recurrent_income_cost":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`non_recurrent_income_cost`
        """)

        df = df.withColumn("dataset", lit(data))

        datasets = {'non_recurrent_income_cost': df}

        freq = "ME"

        target_column_name = "Amount"

        df = df.withColumn("dataset", lit(data))

    elif data == "Telefonica - bad_debt":
        df = spark.sql("""
            SELECT * 
            FROM `analytics`.`p&l_prediction`.`bad_debt`
        """)

        df = df.withColumn("dataset", lit(data))

        datasets = {'bad_debt': df}

        freq = "ME"

        target_column_name = "Amount"

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


    # Convert column types
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


