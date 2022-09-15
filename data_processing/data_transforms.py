import dask.dataframe as dd
import polars as pl
from multipledispatch import dispatch


@dispatch(dd.core.DataFrame, list)
def get_sleep_stage_dict(df, cols):
    """
    returns dictionary with key-value pair as sleep stage (int) : n x size(cols) dataframe corresponding to that sleep stage
    :param df: Either a pandas, Dask, or polars dataframe
    :param cols: columns to keep in sleep_stage_dict
    :return: dict of each sleep stage paired with raw data corresponding to that stage
    """
    sleep_stages = df['SleepStage'].unique().compute().values
    df_gb = df.groupby('SleepStage')
    sleep_stage_data_dict = {stage: df_gb[cols].get_group(stage) for stage in sleep_stages}
    return sleep_stage_data_dict


@dispatch(pl.internals.dataframe.frame.DataFrame, list)
def get_sleep_stage_dict(df, cols):
    """
    returns dictionary with key-value pair as sleep stage (int) : n x size(cols) dataframe corresponding to that sleep stage
    :param df: Either a pandas, Dask, or polars dataframe
    :param cols: columns to keep in sleep_stage_dict
    :return: dict of each sleep stage paired with raw data corresponding to that stage
    """
    sleep_stages = df.select('SleepStage').unique().to_numpy().squeeze()
    sleep_stage_data_dict = {stage: df.select(pl.col(cols)).filter(pl.col('SleepStage') == stage) for stage in sleep_stages}
    return sleep_stage_data_dict


