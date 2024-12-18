################################ EDIT HERE ###################################
# Define the root directory (e.g., name of the project). This should be updated
# based on the user's directory structure.
main_dir = "C:/Users/simon/whc_forecasting"
################################ EDIT HERE ###################################

# Import necessary packages
import os
import sys
import random

# Load custom functions
from tools.paper_results.run_model_aggregation import run_model_aggregation
from tools.methods.filter_datasets import filter_datasets

# Add the project directory to the system path and set it as the current working directory
sys.path.append(os.path.abspath(main_dir))
os.chdir(main_dir)

# Set the random seed for reproducibility of any operations that depend on randomness
random.seed(123)

##############################################################################
# SETTINGS
##############################################################################
# Define the directory to save results
current_dir = os.path.join(main_dir, "results")

# Main settings for the forecasting pipeline
forecast_methods = ["global", "level"]  # Forecasting methods to use
time_limit = [60 * 10, 60 * 30]  # Time limit for each method (in seconds)
models = ["AutoARIMA", "AutoETS", "AutoTS", "AutoGluon"]  # List of models to use
models = ["AutoGluon"]  # Overwrite with a single model for focused testing
optim_method = ["ensemble_selection", "optimize_nnls", "differential_evolution"]  # Optimization methods
used_period_for_cv = 0.45  # Fraction of data used for cross-validation

# Additional settings for computational efficiency and debugging
reduceCompTime = True  # Whether to reduce computation time by simplifying operations
use_best_model = False  # Whether to automatically select the best model
verbosity = 4  # Level of logging detail: higher values provide more debugging info
includeModels = None  # Models to include (if not None, only these models are used)
includeModels = ["SeasonalAverage"]
excludeModels = None  # Models to exclude (if not None, these models are skipped)
remove_groups = [False]  # Whether to remove grouping structures during processing
use_test_data = True  # Whether to include test data in the evaluation
delete_weights_folder = False  # Whether to delete intermediate weight files after use
delete_forecast_folder = False  # Whether to delete forecast files after use
RERUN_calculate_weights = True  # If True, recompute weights even if they exist
RERUN_calculate_forecast = True  # If True, recompute forecasts even if they exist

# Dataset filtering settings
exclude_datasets = None  # Datasets to explicitly exclude
run_Telefonica_data = False  # Whether to include Telefonica-specific datasets
run_other_data = True  # Whether to include non-Telefonica datasets
run_dataset = None  # Specify datasets to include; if None, include all datasets
#run_dataset = ['prison_population']  # Example: include only a specific dataset

# Specific settings for Telefonica datasets
test_period = 1  # Length of the test period
fold_length = 6  # Length of each fold in cross-validation
future_periods = 12  # Number of future periods for forecasting
include_groups = None  # Specific groups to include (if any)

# Define the datasets with their properties
datasets = [
    {"Telefoncia_data": False, "name": "website_traffic", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "store_item_demand", "test_period": 6, "fold_length": 6, "future_periods": 6, "include_groups": None},
    {"Telefoncia_data": False, "name": "retail_prices", "test_period": 12, "fold_length": 12, "future_periods": 12, "include_groups": None},
    {"Telefoncia_data": False, "name": "prison_population", "test_period": 4, "fold_length": 4, "future_periods": 4, "include_groups": None},
    {"Telefoncia_data": False, "name": "natural_gas_usage", "test_period": 12, "fold_length": 12, "future_periods": 12, "include_groups": None},
    {"Telefoncia_data": False, "name": "italian_grocery_store", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "global_electricity_production", "test_period": 15, "fold_length": 15, "future_periods": 15, "include_groups": None},
    {"Telefoncia_data": False, "name": "australian_labour_market", "test_period": 14, "fold_length": 14, "future_periods": 14, "include_groups": None},
    {"Telefoncia_data": False, "name": "M5", "test_period": 180, "fold_length": 180, "future_periods": 180, "include_groups": None},
    {"Telefoncia_data": False, "name": "tourism", "test_period": 7, "fold_length": 7, "future_periods": 7, "include_groups": None},
    {"Telefoncia_data": False, "name": "superstore", "test_period": 4, "fold_length": 4, "future_periods": 4, "include_groups": None},
    {"Telefoncia_data": True, "name": "Telefonica - bad_debt", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - commercial_costs", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - cos", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - fbb_fixed_other_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - hardware_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - mobile_service_revenue", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - non_commercial_costs", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups},
    {"Telefoncia_data": True, "name": "Telefonica - non_recurrent_income_cost", "test_period": test_period, "fold_length": fold_length, "future_periods": future_periods, "include_groups": include_groups}
]

##############################################################################
# Filter datasets
##############################################################################
# Apply the filter_datasets function to select datasets based on the filtering criteria
datasets = filter_datasets(datasets, run_Telefonica_data, run_other_data, run_dataset, exclude_datasets)

# Print the filtered datasets for verification
print("Filtered datasets:")
for dataset in datasets:
    print(dataset)

################################################################################################################
# RUN
################################################################################################################
# Iterate over each filtered dataset and run the model aggregation process
for dataset in datasets:
    try:
        dataset_name = dataset["name"]  # Extract the name of the current dataset
        print(f"Running model aggregation for dataset: {dataset_name}")
        
        # Call the run_model_aggregation function with the relevant parameters
        run_model_aggregation(
            save_intermediate_results_path=os.path.join(current_dir, dataset_name), 
            save_final_results_path=None, 
            dataset_name=dataset_name, 
            model=models, 
            forecast_method=forecast_methods, 
            use_best_model=use_best_model, 
            time_limit=time_limit, 
            verbosity=verbosity, 
            test_period=dataset["test_period"], 
            includeModels=includeModels, 
            excludeModels=excludeModels, 
            fold_length=dataset["fold_length"], 
            used_period_for_cv=used_period_for_cv, 
            include_groups=dataset["include_groups"], 
            optim_method=optim_method, 
            remove_groups=remove_groups, 
            future_periods=dataset["future_periods"], 
            use_test_data=use_test_data, 
            reduceCompTime=reduceCompTime,
            delete_weights_folder=delete_weights_folder,
            delete_forecast_folder=delete_forecast_folder,
            RERUN_calculate_weights=RERUN_calculate_weights,
            RERUN_calculate_forecast=RERUN_calculate_forecast
        )
        
        # Indicate successful completion for the current dataset
        print(f"Finished model aggregation for dataset: {dataset_name}")
        
    except Exception as e:
        # Handle and log any errors that occur during processing
        print(f"Error in model aggregation for dataset: {dataset_name} - {e}")
