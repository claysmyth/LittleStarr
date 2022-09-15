import sys
sys.path.append(BASE_PATH)
import numpy as np
import scipy.signal as signal


def get_PSD_as_array(data, sample_rate, epoch_length, window_length, overlap, column_format=False):
    """
    Takes time series data of arbitrary complexity, and reshapes the time-series portion into a (n x epoch_length) array, then applies the welch method of PSD estimation on each row in (n).
    Assumes the last 2 dimensions are (num channels x num time samples). If last 2 dimensions are (num time samples x num channels), then column_format should be toggled to True.
    :param data: Raw time series data, where last two dimensions should be (num channels x num time samples) or (num samples x num time channels).
    :param sample_rate: sample rate of time series samples [Hz]
    :param epoch_length: number of samples in the resulting row, on which PSD is calculated [int]
    :param window_length: number of samples for welch window [int]
    :param overlap: number of samples to overlap welch windows [int]
    :param column_format: Set to true if last 2 dimensions of input data is (num time samples x num channels)
    :return:
        f: output array f of the signal.welch
        Pxx_den: an (n x f) array denoting the resulting power spectral estimation for each epoch.
    """
    if column_format:
        data = np.transpose(data, np.shape(data)[:-2] + (-1, -2))

    # Ignore tail end of data that does not fit into welch epoch
    num_chunks = (np.shape(data)[-1] - epoch_length) // epoch_length

    # Chunk the time series portion of the data into an (n x epoch_length) array.
    # The last dimension will be a continuous section of data of epoch length, and second to last will be contiguous epochs.
    data_epoch_blocked = np.reshape(data[..., :num_chunks * epoch_length],
                                    (np.shape(data)[:-1]) + (num_chunks, epoch_length))

    # Run PSD calculation on each epoch of data (the last dimension)
    f, Pxx_den = signal.welch(data_epoch_blocked, fs=sample_rate, nperseg=window_length, noverlap=overlap)

    return f, Pxx_den


def get_PSD_dict(data_dict, cols, value_type, **kwargs):
    """
    Wrapper function for get_PSD_as_array(...). Processes data values in data_dict into PSD_arrays
    :param data_dict: dictionary pairs of key - dataframe [either dask, pandas, or polars]
    :param cols: [list of strings] cols to extract from dataframe values. Are ultimately transformed into numpy arrays
    :param value_type: [string] indicates what type of dataframe are stored as values in data_dict
    :param **kwargs: parameter values for get_PSD_as_array(...)
    :return:
        f: output array f of the signal.welch
        psd_dictionary: keys of data_dict pairs with Pxx_den, as outputted from get_PSD_as_array(...)
    """
    if value_type == 'polars':
        psd_dict = {key: get_PSD_as_array(data=value.select(cols).fill_null(0).to_numpy().T, **kwargs)
                    for key, value in data_dict.items()}
    elif value_type == 'dask':
        psd_dict = {key: get_PSD_as_array(data=value[cols].fillna(0).compute().values.T, **kwargs)
                    for key, value in data_dict.items()}
    elif value_type == 'pandas':
        psd_dict = {key: get_PSD_as_array(data=value[cols].fillna(0).values.T, **kwargs)
                    for key, value in data_dict.items()}
    else:
        print('Not a recognized value_type. Allowed types are dask, pandas, or polars')
        return
    keys = list(psd_dict.keys())
    f = psd_dict[keys[0]][0]
    return f, {key: value[1] for key, value in psd_dict.items()}