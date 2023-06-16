import numpy as np
import polars as pl
import pandas as pd

def get_device_as_pl_df(device, db_con, lazy=False, time_zone='America/Los_Angeles'):
    """
    Accesses duckdb database and returns columns of interest, labeled reasonably with session identifiers cast as categoricals.
    :param device: device name (str) (e.g. '02L')
    :param db_con: duckdb connection object
    :return: polars Dataframe
    """
    db_con.sql(f"SET TIMEZONE = '{time_zone}'")
    if lazy:
        return db_con.sql(f"select DerivedTime, columns('localTime'), columns('^Session|TD_|Power_Band'), SleepStage from overnight.r{device}").pl().lazy().with_columns(
            pl.col('^Session.*$').cast(pl.Categorical)
        ).sort('localTime').select(pl.all().shrink_dtype())
    else:
        return db_con.sql(f"select DerivedTime, columns('localTime'), columns('^Session|TD_|Power_Band'), SleepStage from overnight.r{device}").pl().with_columns(
            pl.col('^Session.*$').cast(pl.Categorical)
        ).sort('localTime').select(pl.all().shrink_dtype())

def get_device_as_pd_df(device, db_con):
    """
    Accesses duckdb database and returns columns of interest, labeled reasonably with session identifiers cast as categoricals.
    :param device: device name (str) (e.g. '02L')
    :param db_con: duckdb connection object
    :return: polars Dataframe
    """
    return db_con.execute(f"select DerivedTime, columns('localTime'), columns('^Session|TD_|Power_Band'), SleepStage from overnight.r{device}").df()


def get_gains_from_settings(sessions, identity_col, device, db_con) -> np.ndarray:
    """

    :param settings:
    :param sessions:
    :param identity_col:
    :param device:
    :param db_con:
    :return:
    """
    td_settings = db_con.execute(f"Select columns('^chan'), columns('^gain') from r{device}.TDSettings where {identity_col} in {*sessions,}").df().drop_duplicates()
    if len(td_settings) > 1:
        print(f"WARNING: MULTIPLE TIME DOMAIN SETTINGS FOUND for {device} sessions {sessions}. Using the first unique values.")
    gains = []
    for i in range(1,len(td_settings.columns)+1):
        if (f"chan{i}" in td_settings.columns) and not ('Disabled' in td_settings[f"chan{i}"][0]):
            gains.append(td_settings[f"gain_{i}"][0])
    return np.array(gains)



def get_settings_for_pb_calcs(device, db_con, session_nums, identity_col):
    """
    Get the relevant settings to run rcs_sim package functions:
        rcs.create_hann_window
        rcs.transform_mv_to_rcs
        rcs.td_to_fft
        rcs.fft_to_pb

    :param device:
    :param db_con:
    :param session_nums: list of SessionIdentity values to pull settings from
    :return:
    """
    # TODO: FIX THIS FUNCTION TO ALLOW FOR identity_col = 'Session#'

    td_settings = db_con.execute(f"Select {identity_col}, samplingRate, columns('^gain') from r{device}.TDSettings where {identity_col} in {*session_nums,}").df()
    fft_settings = db_con.execute(f"Select {identity_col}, fft_bandFormationConfig, fft_interval, fft_size, fft_windowLoad, fft_numBins, fft_binWidth, columns('^Power_Band') from r{device}.FftAndPowerSettings where SessionIdentity in {*session_nums,}").df()
    settings_df = pd.concat([td_settings, fft_settings], axis=1, join='inner')
    settings_df['fft_bandFormationConfig'] = settings_df['fft_bandFormationConfig'].str.extract('(\d+)').astype(int)
    if not (np.sum(np.unique(settings_df.loc[:, settings_df.columns!=identity_col].nunique().values)) == 1):
        print("WARNING: Session numbers provided have different settings.")
        print('Settings dataframe:')
        print(settings_df)

    return settings_df.loc[:, settings_df.columns!=identity_col].drop_duplicates().to_dict('list')
