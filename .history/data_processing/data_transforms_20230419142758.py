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


def chunk_df_by_timesegment(df, interval='1s', period='2s', sample_rate=500, align_with_PB_outputs=False, td_columns=['TD_BG', 'TD_key2', 'TD_key3']):
    """
    Chunk a dataframe  based on a time interval and period.
    The period is the length of the time segment, the interval is the time between the start of each time segment.
    
    Parameters:
    df (DataFrame): The dataframe to be chunked
    interval (str): The time interval between the start of each time segment. Default is '1s'
    period (str): The length of each time segment. Default is '2s'
    align_with_PB_outputs (bool): If True, the time segments will be aligned with the Power Band outputs. Default is False.
    """
    td_cols = [col for col in df.columns if col in td_columns]
    # TODO: Remove hardcoding of 'SleepStage', should refer to it as a variable
    if align_with_PB_outputs:
        df_pb_count = df.join(
            df.filter(pl.col('Power_Band8').is_not_null()).select(
                'DerivedTime').with_row_count(),
            on='DerivedTime', how='left').with_columns(pl.col('row_nr').fill_null(strategy='backward')).rename({'row_nr': 'PB_count'})

        df_pb_count = df_pb_count.with_columns([
            pl.when( (pl.col('PB_count') % 2) == 0).then(pl.lit(None)).otherwise(pl.col('PB_count')).fill_null(strategy='backward').alias('PB_count_odd'),
            pl.when( (pl.col('PB_count') % 2) == 1).then(pl.lit(None)).otherwise(pl.col('PB_count')).fill_null(strategy='backward').alias('PB_count_even')
        ])

        df_pb_count = df_pb_count.groupby(['SleepStage', 'PB_count_even']).agg(
            [
                pl.col('DerivedTime'),
                pl.col('^Power_Band.*$').drop_nulls().first(),
                pl.col('^TD_.*$'),
                pl.col(td_cols[0]).count().alias('TD_count')
            ]).rename({'PB_count_even': 'PB_ind'}).vstack(
                df_pb_count.groupby(['SleepStage', 'PB_count_odd']).agg(
                    [
                        pl.col('DerivedTime'),
                        pl.col('^Power_Band.*$').drop_nulls().first(),
                        pl.col('^TD_.*$'),
                pl.col(td_cols[0]).count().alias('TD_count')
                    ]).rename({'PB_count_odd': 'PB_ind'})
        ).select(pl.all().shrink_dtype()).rechunk()

        df_chunked = df_pb_count
    else:
        df_grouped = df.sort('localTime').groupby_dynamic('localTime', every=interval, period=period, by=['SessionIdentity', 'SleepStage']).agg([
            pl.col('DerivedTime'),
            pl.col('^Power_Band.*$').drop_nulls().first(),
            pl.col('^TD_.*$'),
                pl.col(td_cols[0]).count().alias('TD_count')]).select(pl.all().shrink_dtype())

        df_grouped = df_grouped.with_columns(
                    pl.col(td_cols[0]).arr.eval(pl.element().is_null().any()).alias('TD_null')
                ).filter((pl.col('TD_count') == int(period[0]) * sample_rate ) &
                        (pl.col('TD_null').arr.contains(False))
                        )
        df_chunked = df_grouped
    return df_chunked