import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from matplotlib import cm

def get_hours_and_indices(rcs_df, sample_rate):
    """
    Helper function returns the indices and string for each hour of an overnight session (e.g. '01:00' for 1 am)
    :param rcs_df: pandas dataframe of the RC+S session, as created by Analysic-rcs-data/combinedDataTable.m
    :param sample_rate: time domain sample rate in Hz
    :return:
        top_of_hour_indices: row index of each calendar hour
        hours: strings that depict each hour that occurs (e.g. '01:00' for 1 am)
    """
    samples_per_hour = sample_rate*60*60
    first_hour = rcs_df['DerivedTime'].values[0:samples_per_hour]
    top_of_first_hour = np.argmin(first_hour%(60*60*1000))
    num_hours = (len(rcs_df) - top_of_first_hour) // samples_per_hour
    top_of_hour_indices = np.arange(num_hours+1) * samples_per_hour + top_of_first_hour
    hours = rcs_df['localTime'].iloc[top_of_hour_indices].dt.strftime('%H:%M').values
    return top_of_hour_indices, hours

def plot_spec_hypno_overlay(rcs_df, channel, sample_rate, window_length, title, overlap=0, freq_range = np.s_[1:101], max=None, cmap='bwr', sleep_list=["N3", "N2", "N1", "REM", "Awake"]):
    """
    Plots the Spectogram and an overlaid hypnogram of an overnight RC+S session
    :param rcs_df: pandas dataframe of the RC+S session, as created by Analysic-rcs-data/combinedDataTable.m - must have SleepStage column
    :param channel: TD channel to use (e.g. 'TD_key3') (str)
    :param sample_rate: TD sample rate in Hz (int)
    :param window_length: length of window for FFT in number of samples (int)
    :param title: title for plot (str)
    :param overlap: overlap for FFT windows in number of samples (int)
    :param freq_range: spectrogram y-axis range to be depicted on figure (numpy.s_ object)
    :param max: Max value for colorbar (double or float)
    :param cmap: colorbar colorscheme (str)
    :param sleep_list: list of sleep stages, reflecting order of SleepStage labels
    :return: None
    """
    f, t, Sxx = signal.spectrogram(rcs_df[channel].fillna(value=0).values,
                                   sample_rate, nperseg=window_length, noverlap=overlap)

    sleep_stages_parsed = rcs_df["SleepStage"].values
    sleep_stages_overlay = sleep_stages_parsed[window_length-1::(window_length-overlap)]

    fig1 = plt.figure(figsize=(20, 5))
    plt.title(title, size=18)

    tmp = Sxx[freq_range]
    f = f[freq_range]
    tmp[np.where(tmp < 1e-8)] = 1e-8

    if max is None:
        max=tmp.max()

    plt.rcParams['axes.grid'] = False

    mesh = fig1.gca().pcolormesh(t, f, tmp, shading='gouraud', norm=colors.LogNorm(vmin=tmp.min(), vmax=max), cmap=cmap)
    fig1.gca().set_ylabel('Frequency [Hz]', size=18)
    fig1.gca().tick_params(axis='y', labelsize=15)

    fig1.gca().set_xlabel('Time [hour of day]', size=18)
    top_of_hour_indices, hours = get_hours_and_indices(rcs_df, sample_rate)
    hour_indices_adjusted = (top_of_hour_indices // (window_length - overlap)) // (sample_rate / (window_length - overlap))
    fig1.gca().set_xticks(hour_indices_adjusted, size=3)
    fig1.gca().set_xticklabels(hours, size=15)

    fig1.colorbar(mesh, pad=0.1)

    ax2 = fig1.gca().twinx()
    line = ax2.plot(t, (sleep_stages_overlay * 10) - 10, color='k')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylim(0, 50)
    ax2.set_yticks(np.arange(10, 60, 10))
    ax2.set_yticklabels(sleep_list, size=15)

    plt.subplots_adjust(top=2.0)
    plt.gcf().patch.set_facecolor('white')
    plt.show()


def plot_PSD(psd_array, labels, title, f, freq_range=np.s_[1:101], plot_SEM=False, SEM=[], ax=None, legend=False, yo=-1):
    """
    NOTE: Best to take the log of the PSD array prior to function call.
    Plots elements, or values, of psd_array with corresponding labels.
    :param psd_array: [list-like or dict] of psd row vectors that are ultimately plotted.
    :param labels: [list-like or dict] of strings. Must have same order or matching keys to psd_array
    :param title: [str]
    :param freq_range: [np.s_[]] slice of frequencies to plot on x-axis
    :param plot_SEM: [bool] flag to plot error-bars on psds
    :param SEM: [list-like or dict] standard error measure. Same dimension as psd_array
    :return: None
    """
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        #plt.gcf().patch.set_facecolor('white')
    # plt.xlabel('Frequency [Hz]', size=15)
    # plt.ylabel('PSD [log(mV^2/Hz)]', size=15)
    ax.set_facecolor('white')
    ax.set_xlabel('Frequency ($Hz$)', size=18)
    ax.set_ylabel('$log_{10}(mV^2/Hz)$', size=18)
    c = cm.get_cmap('Set1')(np.linspace(0,1,len(labels)))
    if type(psd_array) == dict:
        for j, i in enumerate(labels.keys()):
            plt.plot(f[freq_range], psd_array[i][freq_range], label=labels[i], color=c[j], linewidth=2)
            if plot_SEM:
                sem = SEM[i][freq_range]
                plt.fill_between(f[freq_range], psd_array[i][freq_range]-sem, psd_array[i][freq_range]+sem, alpha=0.2, color=c[j])
    else:
        for i in range(len(labels)):
            plt.plot(f[freq_range], psd_array[i][freq_range], label=labels[i], color=c[j], linewidth=2)
            if plot_SEM:
                sem = SEM[i][freq_range]
                plt.fill_between(f[freq_range], psd_array[i][freq_range]-sem, psd_array[i][freq_range]+sem, alpha=0.2, color=c[j])

    if legend:
        plt.legend(prop={'size': 20}, loc='upper right')
    plt.title(title, size=20)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ax.grid(False)
    plt.ylim([-7.4, -6.4])
    plt.xticks(size=15, ticks=np.arange(55, 76, 5))
    plt.show()

    return None