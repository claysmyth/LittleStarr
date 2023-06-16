import polars as pl
import numpy as np
import pandas as pd
from rcssim import rcs_sim as rcs

def rcssim_wrapper(td_data, times, settings, gain) -> tuple[np.ndarray, int]:
    """
    This function takes in polars lists of times (DerivedTimes) and timedomain data (td_data), and returns an estimate of the embedded RCS fft vector.
    params:
    times: (pl.list of floats: unix timestamps) 1xn vector of DerivedTimes
    td_data: (pl.list of floats: millivolts) 1xn vector of time domain data
    settings: (dict) settings for device
    returns: data_fft_pl: (pl.list of floats) Estimate of embedded FFT vector(s) (size: mxn) corresponding to td_data
             t_pb: (float: unix timestamp) DerivedTime unix timestamp of FFT vector
    """
    hann_win = rcs.create_hann_window(settings['fft_size'][0], percent=100)
    # times_np = times.to_numpy(zero_copy_only=True)

    # td_np = rcs.transform_mv_to_rcs(td_data.to_numpy(zero_copy_only=True), gain)
    td_np = rcs.transform_mv_to_rcs(td_data, gain)

    data_fft, t_pb = rcs.td_to_fft(td_np, times,
                                   settings['samplingRate'][0],
                                   settings['fft_size'][0], settings['fft_interval'][0],
                                   hann_win, interp_drops=False, output_in_mv=False, shift_timestamps_up_by_one_ind=True)
    data_fft_out = rcs.fft_to_pb(data_fft, settings['samplingRate'][0], settings['fft_size'][0],
                                 settings['fft_bandFormationConfig'][0],
                                 input_is_mv=False)
    return data_fft_out, t_pb


def add_simulated_ffts(df_chunked, settings, gains, shift=None, td_columns=['TD_BG', 'TD_key2', 'TD_key3']):    
    """
    Add simulated FFTs to a dataframe of time segments. This function calls the rcs_sim package to simulate RC+S FFT outputs from time domain data,
    and adds the FFT outputs to the dataframe as additional columns.
    
    parameters:
    df_chunked (pl.DataFrame): The dataframe of time segments, as output by data_transforms.chunk_df_by_timesegment
    settings (dict): The settings dictionary for the simulation (usually replicating the settings used to generate the original FFTs from an RCS session)
    gains (list): The amp gains to use for the simulation
    fft_subset_inds (list): The indices of the FFTs to use for the simulation. Default is [2,120]

    returns:
    df_chunked (pl.DataFrame): The dataframe of time segments with simulated FFTs added as additional columns. Each time domain channel has its own set of simulated FFTs.
    """

    # Select the time domain channels and timestamps to use for the simulating FFTs
    td_np = df_chunked.select([
        pl.col('DerivedTime'),
        pl.col('^TD_.*$')
    ]).collect().to_numpy()

    td_cols = [col for col in df_chunked.columns if col in td_columns]

    fft_arr = np.zeros((len(td_cols), td_np.shape[0], settings['fft_numBins'][0]))

    sim_settings = settings.copy()
    if shift:
        sim_settings['fft_bandFormationConfig'] = [shift]


    # Could try to use numba to speed up the loop, or use 'pl.apply' on the dataframe
    # Simulate FFTs for each time segment, one FFT period at a time
    # def fft_sim_row(i):
    #     for j in range(len(td_cols)):
    #         fft_arr[j,i], _ = rcssim_wrapper(td_np[i,j+1], td_np[i,0], sim_settings, gains[j])


    #@njit(parallel=True)
    for i in range(td_np.shape[0]):
        #fft_sim_row(i)
        for j in range(len(td_cols)):
            fft_arr[j,i], _ = rcssim_wrapper(td_np[i,j+1], td_np[i,0], sim_settings, gains[j])

    # Join the simulated FFTs to the dataframe
    df_chunked = df_chunked.with_row_count().join(
                            pl.LazyFrame({f'fft_of_{td_cols[i]}': fft_arr[i] for i in range(len(td_cols))}).with_row_count(), 
                            on='row_nr', how='left')

    return df_chunked