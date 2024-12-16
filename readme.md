# Time Series Forecasting for Predictive Planning

This repository contains the code and instructions to reproduce the publicly available results of my master's thesis, which introduces the Weighted Hierarchical Combination method, a new approach to achieve high-accuracy top-level forecasts within hierarchical time series structures.

## Reproducing Results

To reproduce the results of the thesis, follow these steps:

1. Clone this repository and install the required packages from the `requirements.txt` file using:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script `modified_method_weighted_aggregation.py` located in the `results` folder to generate forecasts for the **11 pre-processed public datasets** using the specified models and configurations. Read further instructions on top of the file before running.

3. Run the script `paper_tables.py` in the `results` folder to create the tables shown in the Results section of the thesis. Read further instructions on top of the file before running.

## Repository Structure

- **datasets/**: Contains the pre-processed datasets and the corresponding notebooks used to prepare the data.
  
- **figures/**: Includes the scripts used to generate Figures 4 and 5 in the thesis.

- **tools/**: Contains functions for:

  - Data preprocessing
  - Training and forecasting
  - Recreating Results
  
  For a detailed description of these functions, refer to the `tools.txt` file.

## Notes on Data Preparation

The pre-processing scripts cannot be executed without further customisation, as the preparation was done in Databricks, with the original datasets stored in a Hive metastore. Therefore, the preprocessing scripts only serve to trace the preprocessing steps back to the processed datasets used in this thesis.

