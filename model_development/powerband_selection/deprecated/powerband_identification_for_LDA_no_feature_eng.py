from powerband_identification_for_LDA_pipeline_funcs import *
from sklearn.feature_selection import SequentialFeatureSelector

def convert_list_to_freq_dict(x, td_cols, binWidth, fft_numBins):
    if x:
        freq_dict = {'Channel': td_cols[np.unique(np.array(x) // fft_numBins)[0]], 
                       'freq_range': [x[0] * binWidth - binWidth/2, x[1] * binWidth + binWidth/2] }
    else:
        freq_dict = {'Channel': None, 'freq_range': None}
    return freq_dict

def convert_pb_indices_to_frequencies(df, td_cols, binWidth, fft_numBins):
    df.with_columns([
        pl.col(pb).apply(
            lambda x: convert_list_to_freq_dict(x, td_cols, binWidth, fft_numBins), skip_nulls=False
        ).alias(pb[:3]) 
        for pb in df.columns if 'PB' in pb
    ])
    return df


def convert_index_columns_to_freq_using_binWidth(td_cols, x, fft_length, bin_width):
    '''
    Converts the index column of the feature selection dataframe to a dictionary of frequencies.
    Parameters:
    td_cols (list): The list of TD channel names.
    x (list): List of indices of Powerband Edges.
    fft_length (int): The length of the simulated fft vector for each TD channel (e.g. 512 for an FFT window of 1024 on a 500 TD sampling rate).
    bin_width (float): The width of each FFT bin in Hz.
    '''
    powerbands_freq = []
    print(x)
    if np.array(x).ndim == 3:
        # Iterate through each powerband combination
        for combination in x:
            combination_freqs_with_channels = []
            # Iterate through each powerband in the powerband combination. Could be of length 1, as this function allows single powerbands to be converted
            for powerband in combination:
                index = np.unique(np.array(powerband) // fft_length)
                assert np.size(index == 1) # make sure that the index is only one value, otherwise this powerband spans multiple channels.
                index = index[0]
                combination_freqs_with_channels.append({'Channel': td_cols[ index ], 'freq_range': [round(bin_width * powerband[0] - bin_width/2, 2), 
                                                                                                    round(bin_width * powerband[1] + bin_width/2, 2)]})
            powerbands_freq.append(combination_freqs_with_channels)
    elif np.array(x).ndim == 2:
        for powerband in x:
            combination_freqs_with_channels = []
            index = np.unique(np.array(powerband) // fft_length)
            assert np.size(index == 1) # make sure that the index is only one value, otherwise this powerband spans multiple channels.
            index = index[0]
            combination_freqs_with_channels.append({'Channel': td_cols[ index ], 'freq_range': [round(bin_width * powerband[0] - bin_width/2, 2), 
                                                                                                round(bin_width * powerband[1] + bin_width/2, 2)]})
            powerbands_freq.append(combination_freqs_with_channels)
    else:
        raise ValueError('The list of powerband indices must be of dimension 2 or 3.')
    return powerbands_freq

def convert_index_column_to_freq_using_binWidth(td_cols, x, fft_length, bin_width):
    '''
    SIMILAR TO convert_index_columns_to_freq_using_binWidth, but for a single powerband index column.
    Converts the index column of the feature selection dataframe to a dictionary of frequencies.
    Parameters:
    td_cols (list): The list of TD channel names.
    x (list): List of indices of Powerband Edges.
    fft_length (int): The length of the simulated fft vector for each TD channel (e.g. 512 for an FFT window of 1024 on a 500 TD sampling rate).
    bin_width (float): The width of each FFT bin in Hz.
    '''
    if type(x) is list:
        x = np.array(x)

    assert x.ndim == 2, "The powerband index column must be of dimension 2."
    td_indexes = x // fft_length
    powerbands_freq = (x % fft_length) * bin_width
    powerbands_freq[:, 0] = powerbands_freq[:, 0] - bin_width/2
    powerbands_freq[:, 1] = powerbands_freq[:, 1] + bin_width/2
    powerbands_freq = np.around(powerbands_freq, 2)
    
    powerband_dicts = []
    for i in range(np.shape(td_indexes)[0]):
        if any(np.isnan(x[i])):
            powerband_dicts.append({'Channel': 'None', 'freq_range': [-1.0, -1.0]})
        else:
            powerband_index = np.unique(td_indexes[i]).astype(int)
            assert np.size(powerband_index) == 1 # make sure that the index is only one value, otherwise this powerband spans multiple channels.
            powerband_dicts.append({'Channel': td_cols[powerband_index[0]], 'freq_range': powerbands_freq[i, :].tolist()})

    return powerband_dicts
        
# TODO: Log SFS outputs (clean up method of saving PB indices as frequencies). Save pb combos and scores to parquet file. Run on Cortical and subcortical data.

# The below function is intended to identify potential powerband combinations for maximizing sleep stage classification via LDA (or alternative models). Executes the above functions in order. 

def hyperparameter_search_pipeline(device, parameters, sleep_stage_mapping, out_file_path, db_path):
    """
    Executes the hyperparameter search pipeline for a given device. The pipeline is as follows:
    1. Get the device's fft dataframe.
    2. Correlate FFT bins with the binarized sleep stage labels.
    3. Remove FFT bins with low correlation (below the threshold)
    4. Use SequentialForwardFeature selection to get the n best remaining FFT bins for sleep stage classification.
    5. Cluster the n best FFT bins into 2 to k clusters.
    5. Get the powerband combinations for each cluster amount.
    6. Use the powerbands combinations and update rate to procure training data for LDAs.
    7. Run an LDA cross validation on each powerband combination.
    8. Get the cross-validated LDA model's scores for each powerband combination.
    9. Save the powerband combinations and corresponding scores to a parquet file.

    parameters:
    device (str): The device to execute the pipeline on.
    parameters (dict): The parameters dictionary. It must contain the following keys:
        'TD_Columns' (list): The TD channels to use for the pipeline.
        'fft_subset_inds' (list): The start and end indices of the fft subset that is used for feature selection. In other words, pre-limit the FFT bins desired to be used for feature selection. 
        'update_rate' (int): The update rate, of which to average the powerbands over.
        'correlation_threshold' (float): The correlation threshold for removing FFT bins with low correlation (below the threshold).
        'num_features_to_select' (int): The number of features to select in the SequentialForwardFeature selection.
        'max_clusters' (int): The maximum number of clusters to use for the k-means clustering.
        'model' (sklearn model): The sklearn model to use for the cross-validated LDA. (OPTIONAL, defaults to LDA)
    sleep_stage_mapping (dict): The sleep stage mapping dictionary. This is used to binarize the sleep stages into 0 and 1.
    out_file_path (str): The path to save the parquet file to.
    """
    # Get the device's fft dataframe
    con = duckdb.connect(db_path, read_only=True)
    df_chunked = con.sql(f'select * from r{device}.overnight_simulated_FFTs_train;').pl()
    settings = con.sql(f'select * from r{device}.overnight_simulated_FFTs_settings;').pl().to_dict(as_series=False)
    # TODO: NEED TO SELECT ONLY DESIRED TD CHANNELS
    
    # Remap sleep stages to binary classes
    df_chunked = df_chunked.with_columns(pl.col('SleepStage').map_dict(sleep_stage_mapping).alias('SleepStageBinary'))

    # Correlate FFT bins with sleep stage labels
    df_fft_subset, fft_corrs = get_fft_corrs(df_chunked, parameters['fft_subset_inds'], label_col='SleepStageBinary', td_columns=parameters['TD_Columns'], corr_type=parameters['corr_type'])

    # Save correlation chart
    fft_subset_length = parameters['fft_subset_inds'][1] - parameters['fft_subset_inds'][0]
    corr_df = pd.DataFrame({'Hz': settings['fft_binWidth'][0]*(np.arange(fft_subset_length) + parameters['fft_subset_inds'][0]), **{key: fft_corrs[fft_subset_length*ind:fft_subset_length*(ind+1)] for ind, key in enumerate(parameters['TD_Columns'])}}).melt(id_vars='Hz', value_vars=parameters['TD_Columns'], var_name='key', value_name='corr')
    # corr_chart = alt.Chart(corr_df).mark_line().encode(x='Hz', y='corr', color='key')
    # corr_chart.save(os.path.splitext(out_file_path)[0] + '_corr_chart.png')
    ax = sns.lineplot(data=corr_df, x='Hz', y='corr', hue='key')
    plt.savefig(os.path.splitext(out_file_path)[0] + '_corr_chart.png')

    # TODO: Add option to use KL divergence instead of correlation
    first_feature_group, X, y = get_train_data_by_corr_threshold(df_fft_subset, fft_corrs, parameters['corr_threshold'], label_col='SleepStageBinary')
    del(df_fft_subset)
    
    model = parameters['model']
    print(f'Running Sequential Feature Selector... with {model}')
    sfs = SequentialFeatureSelector(model, n_features_to_select=parameters['num_features_to_select'], direction='forward', n_jobs=parameters['cross_val']+1, cv=parameters['cross_val'], scoring=parameters['sfs_scoring'])
    sfs.fit(X, y)

    # Cluster and combine powerbands in combinations (with various update rates)
    # TODO: Use functions from KL divergence pipeline
    second_feature_group = first_feature_group[sfs.get_support(indices=True)]
    second_feature_group_original_inds = np.array(recover_original_feature_inds(second_feature_group, parameters['fft_subset_inds'], settings['fft_numBins'][0])).squeeze()
    pb_combos, pb_chart = get_PB_combinations(second_feature_group_original_inds, max_clusters=parameters['max_clusters'])
    df_pbs_corrected = get_df_from_pb_combos(pb_combos)

    df_pbs_corrected = df_pbs_corrected.join(pl.DataFrame({'UpdateRate': [2, 5, 10, 15, 30]}), how='cross')

    fft_cols = [col for col in df_chunked.columns if 'fft' in col]

    df_chunked = df_chunked.select([
        pl.col('SleepStage'),
        pl.col('SleepStageBinary'),
        pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).alias('fft_vec')
    ])
    # try:
    #     assert fft_length == df_chunked.select(
    #             pl.col('fft_vec').arr.lengths()
    #         ).unique().item() / NUM_TD_CHANNELS
    # except AssertionError:
    #     print('fft_length is not equal to the length of the fft_vec column.')
    #     print('fft_length: ', fft_length)
    #     print('length of fft_vec column: ', df_chunked.select(
    #             pl.col('fft_vec').arr.lengths()
    #         ).unique().item() / NUM_TD_CHANNELS)


    print('Searching over powerband combos...')
    scores = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    scores_stds = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    for i in range(df_pbs_corrected.height):
        pbs = [value for value in df_pbs_corrected.select(pl.exclude('UpdateRate'))[i].to_dicts()[0].values()]
        X, y = get_training_data_for_model(df_chunked, pbs, df_pbs_corrected[i,'UpdateRate'], label_col='SleepStageBinary')

        # NOTE: 'roc_auc' gets label predictions with clf.predict_proba(X)[:, 1], which allows more thresholds to be tested. 
        # clf.predict_proba(X)[:, 1] is the probability of the positive class (1). clf.predict_proba(X)[:, 0] is the probability of the negative class (0)
        score_dict = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy', 'recall': 'recall', 'precision': 'precision'}

        cv_results = cross_validate(model, X, y, cv=parameters['cross_val'],
                        scoring=score_dict, n_jobs=parameters['cross_val']+1)
        
        [scores[k].extend([np.mean(v)]) for (k, v) in cv_results.items() if k in scores.keys()]
        [scores_stds[k].extend([np.std(v)]) for (k, v) in cv_results.items() if k in scores_stds.keys()]

        tnr = np.array(cv_results['test_balanced_accuracy']) * 2 - cv_results['test_recall']
        scores['test_tnr'].extend([np.mean(tnr)])
        scores_stds['test_tnr'].extend([np.std(tnr)])
        
    #print(settings['fft_numBins'], settings['samplingRate'], settings['fft_size'])
    # Need to double check how to pass
    # df_pbs_corrected = df_pbs_corrected.with_columns([
    #     pl.col(pb).apply(
    #         lambda x: convert_index_column_to_freq_dict(parameters['TD_Columns'], np.array(x.to_list()), settings['fft_numBins'][0], settings['samplingRate'][0], settings['fft_size'][0])
    #     ).alias(pb[:3]) 
    #     for pb in df_pbs_corrected.columns if 'PB' in pb
    # ])

    # TODO: Use KL divergence func methods for converting index columns to frequencies
    df_pbs_corrected = df_pbs_corrected.with_columns([
        pl.col(pb).map(
            lambda x: pl.Series(convert_index_column_to_freq_using_binWidth(parameters['TD_Columns'], np.stack(x.to_numpy()), settings['fft_numBins'][0], settings['fft_binWidth'][0]))
        ).alias(f"{pb[:3]}_channel_freq_range") 
        for pb in df_pbs_corrected.columns if 'PB' in pb
    ])

    # TODO: Check that pbs are saved in legible manner

    df_hyperparams = pl.concat([df_pbs_corrected, pl.DataFrame(scores).rename(
        {'test_accuracy': 'Acc', 'test_roc_auc': 'AUC', 'test_balanced_accuracy': 'BalAcc', 'test_recall': 'TPR', 'test_precision': 'Precision', 'test_tnr': 'TNR'}),
        pl.DataFrame(scores_stds).rename({'test_accuracy': 'Acc_std', 'test_roc_auc': 'AUC_std', 'test_balanced_accuracy': 'BalAcc_std', 'test_recall': 'TPR_std', 'test_precision': 'precision_std', 'test_tnr': 'TNR_std'})
        ], how='horizontal')

    # TODO: Run leave-one-session-out validation on top PB combos
    df_hyperparams.write_parquet(out_file_path)
    # corr_chart.save(BASE_PATH + 'sleepstage_corr.png')

    # return (device, df_hyperparams, corr_chart.properties(title=f'{device}'), pb_chart.properties(title=f'{device}'))
    return (device, df_hyperparams, pb_chart.properties(title=f'{device}'))