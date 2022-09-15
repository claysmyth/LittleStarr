import numpy as np

def get_proportion_Nans(df, channel):
    """
    Returns the proportion of TD channel that is Nan
    :param df: pandas dataframe of RC+S data as outputted by Analysis-rcs-data/combinedDataTable.m
    :param channel: TD channel (e.g. 'TD_key3')
    :return: returns proportion of Nan values (double between 0 and 1)
    """
    per_nans = np.sum(np.isnan(df[channel].values)) / np.size(df[channel].values)
    return per_nans