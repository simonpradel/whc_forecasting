# Databricks notebook source
def pivot_table_w_subtotals(df, values, indices, columns, aggfunc, fill_value):
    '''
    Adds tabulated subtotals to pandas pivot tables with multiple hierarchical indices.
    
    Args:
    - df - dataframe used in pivot table
    - values - values used to aggregrate
    - indices - ordered list of indices to aggregrate by
    - columns - columns to aggregrate by
    - aggfunc - function used to aggregrate (np.max, np.mean, np.sum, etc)
    - fill_value - value used to in place of empty cells
    
    Returns:
    -flat table with data aggregrated and tabulated
    
    '''
    listOfTable = []
    for indexNumber in range(len(indices)):
        n = indexNumber+1
        if n == 1:
            table = pd.pivot_table(df,values=values,index=indices[:n],columns=columns,aggfunc=aggfunc,fill_value=fill_value,margins=True)
        else:
            table = pd.pivot_table(df,values=values,index=indices[:n],columns=columns,aggfunc=aggfunc,fill_value=fill_value)
        table = table.reset_index()
        for column in indices[n:]:
            table[column] = ''
        listOfTable.append(table)
    concatTable = pd.concat(listOfTable).sort_index()
    concatTable = concatTable.set_index(keys=indices)
    return concatTable.sort_index(axis=0,ascending=True)
pivot_table_w_subtotals(df=df,values='Value',indices=['Store','Department','Type'],columns=[],aggfunc='sum',fill_value='')
