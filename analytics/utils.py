import numpy as np
import scipy.stats as stats

def remove_epochs_with_value():
    # TODO: Create function that removes rows of last dimension (i.e second to last dimension) which contain a
    #  specific value. (e.g. epochs with 0's in PSD_array)
    return None

def process_PSDs_for_channel(psd_dict, channel):
    """
    Processes PSD arrays for plotting. Processes PSD arrays by taking the log of the psd array for each channel, and calculating the average and sem of the log(data) for each channel
    :param psd_dict: key-value pairing, where each value is an (num channels x n x psd length) array of power spectral density estimates
    :param channel: [tuple, e.g. (1,2) or (1) or just 1] index on which to process the psd array
    :return: average [1 x psd length] and standard error measure [1 x psd length] of the input psd array
    """
    ave = {}
    sem = {}
    for key, arr in psd_dict.items():
        arr_pruned = arr[channel][np.where(~(arr[channel] == 0).any(axis=-1)),:].squeeze()
        arr_log = np.log10(arr_pruned)
        ave[key] = np.average(arr_log, axis=-2)
        sem[key] = stats.sem(arr_log, axis=-2)
    return ave, sem
