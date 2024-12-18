def split_data(data, period, unique_id, format="dictionary", set_index=False):
    """
    Splits the data into training and testing datasets based on the given period and chosen format.

    Parameters:
    data (dict or DataFrame): 
        - If a dictionary, it should contain one or more DataFrames. It is expected to have a key 'top_level' for the top-level DataFrame and other keys for lower-level DataFrames.
        - If a single DataFrame, it should be in a 'long' format with data organized by a 'unique_id' column.
    period (int): 
        The number of most recent rows per 'unique_id' to be used for testing. These rows are taken from the end of each group of 'unique_id'.
    unique_id (str): 
        The column name used as the 'unique_id' to identify groups within the data. The data will be grouped by this column.
    format (str): 
        The format in which the data is split. 
        - 'dictionary' (default): The data is returned as dictionaries, each containing a training and test DataFrame.
        - 'long': The data is returned as individual DataFrames for the training and testing data.
    set_index (bool): 
        If True, the 'unique_id' column will be set as the index for the resulting DataFrames.

    Returns:
    dict or tuple: 
        - If 'format' is 'dictionary', a tuple containing two dictionaries (train_data, test_data) will be returned.
          Each dictionary contains the training and testing DataFrames for the corresponding key in the input data.
        - If 'format' is 'long', a tuple containing two DataFrames (train_df, test_df) will be returned for the top-level DataFrame.

    Raises:
    ValueError: 
        If the 'format' is not either 'dictionary' or 'long'.
    """

    if format == "dictionary":
        # Initialize dictionaries for training and testing data
        train_data = {}
        test_data = {}

        # Split the data based on the period count for each 'unique_id'
        for key, df in data.items():
            # Select the last 'period' entries per unique_id as test data
            test_data[key] = df.groupby(unique_id).tail(period)
            train_data[key] = df.drop(test_data[key].index)
            
            # Set the index to 'unique_id' if required
            if set_index:
                train_data[key] = train_data[key].set_index(unique_id)
                test_data[key] = test_data[key].set_index(unique_id)
        
        return train_data, test_data
    
    elif format == "long":
        # Extract the DataFrame corresponding to the top-level data ('Y_df')
        data = data['Y_df']

        # Split the data into training and testing datasets
        test_df = data.groupby(unique_id).tail(period)
        train_df = data.drop(test_df.index)

        # Set the index to 'unique_id' if required
        if set_index:
            test_df = test_df.set_index(unique_id)
            train_df = train_df.set_index(unique_id)

        return train_df, test_df

    else:
        raise ValueError("The format must be either 'dictionary' or 'long'.")
