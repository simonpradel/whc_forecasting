There are 3 major things to do to generate an output

1) Load data. The data can be loaded from the sql-database. To ensure consistency the data needs to be saved in the catalog. The notebook used to do so is the "Preprocessing_data_Telefonica" notebook. It does the following
- It load and combines the table vFACT_REVENUE_COS, vFACT_OPEX and Adjusting_Table
- The Adjusting_table contains one-offs and structural adjustments. structural adjustments needs to transfered into one-offs to use them
- Dimensionstables for each variable used needs prepration: It is possible to decide which observations wants to be used -> less grouping variables possible
- The p&L Lines are splitten in eight different categories: mobile sales, hardware sales, fbb/fixed/other sales, cos, commercial costs, non-commercical costs, bad debt and non-recurrent income/costs. Each of these Lines needs to be extracted from the data, aggregated to the needed dimension and saved. 
2) Train and Make Forecasts. Here the data are trained and the forecast in the future is done
- Important: Set parameters: "test_period", "future_period"
