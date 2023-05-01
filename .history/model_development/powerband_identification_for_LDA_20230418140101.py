import duckdb
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from rcssim import rcs_sim as rcs
from itertools import combinations
import sys
sys.path.append('../databasing')
from database_calls import get_device_as_pl_df, get_gains_from_settings_dict, get_settings_for_pb_calcs


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans

DATABASE_PATH = '/media/shortterm_ssd/Clay/databases/duckdb/rcs-db.duckdb'
ALL_TD_COLUMNS = ['TD_BG', 'TD_key2', 'TD_key3']

# TODO: Use function from dataprocessing/data_transforms instead of this one
def chunk_df_by_timesegment(df, interval='1s', period='2s', sample_rate=500, align_with_PB_outputs=False, td_columns=['TD_BG', 'TD_key2', 'TD_key3']):
    """
    Chunk a dataframe into smaller dataframes based on a time interval and period.
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


def add_simulated_ffts(df_chunked, settings, gains, shift=None, td_columns=['TD_BG', 'TD_key2', 'TD_key3']):    
    """
    Add simulated FFTs to a dataframe of time segments. This function calls the rcs_sim package to simulate RC+S outputs
    
    parameters:
    df_chunked (pl.DataFrame): The dataframe of time segments, as output by chunk_df_by_timesegment
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

    print(td_cols)

    fft_arr = np.zeros((len(td_cols), td_np.shape[0], settings['fft_numBins'][0]))

    sim_settings = settings.copy()
    if shift:
        sim_settings['fft_bandFormationConfig'] = [shift]


    # Could try to use numba to speed up the loop, or use 'pl.apply' on the dataframe
    # Simulate FFTs for each time segment, one FFT period at a time
    def fft_sim_row(i):
        for j in range(len(td_cols)):
            fft_arr[j,i], _ = rcssim_wrapper(td_np[i,j+1], td_np[i,0], sim_settings, gains[j])


    #@njit(parallel=True)
    for i in range(td_np.shape[0]):
        fft_sim_row(i)

    # Join the simulated FFTs to the dataframe
    df_chunked = df_chunked.with_row_count().join(
                            pl.LazyFrame({f'fft_{i}': fft_arr[i] for i in range(len(td_cols))}).with_row_count(), 
                            on='row_nr', how='left')

    return df_chunked
def expand_fft_arrays_with_subset_deprecated(df, fft_ind_start, vec_length):
    """
    Take a subset of each Time Domain Channels FFT simulations, and expand each fft bin into it's own column

    parameters:
    df (pl.DataFrame): The dataframe of time segments with simulated FFTs added as additional columns. Each time domain channel has its own set of simulated FFTs.
    fft_ind_start (int): The index of the first FFT bin to use
    vec_length (int): The number of FFT bins to use when calculating the fft subset

    returns:
    df (pl.DataFrame): The dataframe of time segments with simulated FFTs bins added as individual columns
    """
    return (df.with_columns([
                        pl.col('^fft_.*$').arr.slice(fft_ind_start, vec_length)
                    ])
                    .with_columns(
                        pl.col('fft_BG').arr.concat([pl.col('fft_key2'), pl.col('fft_key3')]).arr.to_struct().alias('fft_vec')
                    )
                    .unnest('fft_vec')
                    )


def expand_fft_arrays_with_subset(df, fft_ind_start, vec_length):
    """
    Take a subset of each Time Domain Channels FFT simulations, and expand each fft bin into it's own column

    parameters:
    df (pl.DataFrame): The dataframe of time segments with simulated FFTs added as additional columns. Each time domain channel has its own set of simulated FFTs.
    fft_ind_start (int): The index of the first FFT bin to use
    vec_length (int): The number of FFT bins to use when calculating the fft subset

    returns:
    df (pl.DataFrame): The dataframe of time segments with simulated FFTs bins added as individual columns
    """
    fft_cols = [col for col in df.columns if 'fft' in col]
    return (df.with_columns([
                        pl.col('^fft_.*$').arr.slice(fft_ind_start, vec_length)
                    ])
                    .with_columns(
                        pl.col(fft_cols[0]).arr.concat(pl.col(fft_cols[1:])).arr.to_struct().alias('fft_vec')
                    )
                    .unnest('fft_vec')
                    )


def get_fft_corrs(df_chunked, fft_subset_inds=[2,120], label_col='SleepStage', td_columns=['TD_BG', 'TD_key2', 'TD_key3']) -> np.ndarray:
    """
    Calculate the correlation between each FFT bin and the label columns (SleepStage, or other)

    parameters:
    df_chunked (pl.DataFrame): The dataframe of time segments with simulated FFTs added as additional columns. Each time domain channel has its own set of simulated FFTs.
    fft_subset_inds (list): The indices of the subset of each FFT output to use for the simulation. Default is [2,120]
    label_col (str): The column to use for the label, for which each frequency bin will be compared to in a pearson correlation measure. Default is 'SleepStage'

    returns:
    df_fft_vec (pl.DataFrame): The dataframe with the desired subset of simulated FFTs bins for each FFT interval added as individual columns
    fft_corrs (np.ndarray): The correlation between each FFT bin and the label column
    """

    fft_ind_start = fft_subset_inds[0]
    vec_length = fft_subset_inds[1] - fft_subset_inds[0]

    df_fft_vec = expand_fft_arrays_with_subset(df_chunked, fft_ind_start=fft_ind_start, vec_length=vec_length)

    # Could convert fft_corrs to be a pl.dataframe with discord trick by @ms. This would allow it to be lazy and save on memory.
    fft_corrs = df_fft_vec.select([
                        pl.corr(f"field_{i}", pl.col(label_col).cast(pl.Float64), method='pearson').alias(f"field_{i}_corr") for i in range(vec_length*len(td_columns))
                    ]).to_numpy().squeeze()
    
    return df_fft_vec, fft_corrs


def get_train_data_by_corr_threshold(df, fft_corrs, corr_threshold, label_col='SleepStageBinary'):
    """
    Reduce features for powerband selection, by selecting the FFT bins that have a correlation above the threshold

    parameters:
    df (pl.DataFrame): The dataframe of time segments (i.e. individual fft windows are single rows) with simulated FFTs added as additional columns.
    corr_threshold (float): The threshold for the correlation between the FFT bins and the label column. Only FFT bins with a correlation above this threshold will kept for future feature selection.
    label_col (str): The column to use for the label, for which each frequency bin will be compared to in a pearson correlation measure. Default is 'SleepStageBinary'
    """
    X = df.select(pl.col("^field_.*$")).to_numpy()
    feature_group = np.argwhere(np.abs(fft_corrs) > corr_threshold)
    X = X[:, np.where(np.abs(fft_corrs) > corr_threshold)[0]]
    y = df.select(pl.col(label_col)).to_numpy().squeeze()
    return feature_group, X, y


def get_PB_combinations_deprecated(sfs, feature_group, max_clusters=8):
    PB_groupings = []
    for i in range(2, max_clusters+1): 
        kmeans = KMeans(n_clusters=i, random_state=0).fit(sfs.get_support(indices=True)[:, np.newaxis])

        pbs = [[feature_group[np.argwhere(kmeans.labels_ == j).squeeze()].min(), 
                feature_group[np.argwhere(kmeans.labels_ == j).squeeze()].max()] for j in range(i)]
        
        cluster_df = pd.DataFrame({'x':sfs.get_support(indices=True), 'y':np.zeros(sfs.get_support(indices=True).shape[0]), 'color':kmeans.labels_})
        if i == 2:
            chart = alt.Chart(cluster_df).mark_point().encode(x='x', y='y', color=alt.Color('color', scale=alt.Scale(scheme='category20b')))
        else:
            chart &= alt.Chart(cluster_df).mark_point().encode(x='x', y='y', color=alt.Color('color', scale=alt.Scale(scheme='category20b')))
        
        PB_groupings.append(pbs)
    
    pb_combos = []
    for pbs in PB_groupings:
        if len(pbs) <= 4:
            pb_combos.append(pbs)
        else:
            pb_combos.append(list(combinations(pbs, 4)))
            
    return pb_combos, chart


def get_PB_combinations(features, max_clusters=8):
    """
    Takes in the features and returns the possible powerband combinations. Features are fft indices as determined by SequentialFeatureSelector.

    parameters:
    features (np.ndarray): The fft indices that were selected by SequentialFeatureSelector. They chould correspond to the indicies within the entire fft vector (i.e. the three channels fft outputs horizontally concatenated), 
        not the subset of the FFTs that was used for feature selection.
    max_clusters (int): The maximum number of possible powerbands to consider for selection. Default is 8.

    returns:
    pb_combos (list): A list of lists of lists. The first list is the combination of powerbands for the corresponding cluster amount, the second list is the individual powerbands, 
    and the third list is the start and end indices of the powerband.
    chart (altair.Chart): An interactive chart of the powerband combinations. The x-axis is the fft indices, the y-axis is irrelevant, and the color is the cluster number.
    """
    PB_groupings = []
    for i in range(2, max_clusters+1): 
        kmeans = KMeans(n_clusters=i, random_state=0).fit(features[:, np.newaxis])

        # Extract first and last index of each cluster, to be used as powerband boundaries
        pbs = [[features[np.argwhere(kmeans.labels_ == j).squeeze()].min(), 
                features[np.argwhere(kmeans.labels_ == j).squeeze()].max()] for j in range(i)]
        
        cluster_df = pd.DataFrame({'x':features, 'y':np.zeros(features.shape[0]), 'color':kmeans.labels_})
        base = alt.Chart(cluster_df).mark_point().encode(x='x', y='y', color=alt.Color('color', scale=alt.Scale(scheme='category20b'))).interactive()
        if i == 2:
            chart = base
        else:
            chart &= base
        
        PB_groupings.append(pbs)
    
    pb_combos = []
    for pbs in PB_groupings:
        if len(pbs) <= 4:
            pb_combos.append(pbs)
        else:
            # Get all combinations of 4 powerbands chosen from the possible powerbands in that cluster (e.g. if num clusters is 8, add all combinations of 4 powerbands from the 8 powerbands)
            pb_combos.append(list(combinations(pbs, 4)))
            
    return pb_combos, chart


def recover_original_feature_inds(feature_list, fft_subset_inds, fft_length):
    """
    Recover the original fft indices (i.e. the indices of the fft bins from the FFT vector that corresponds to the concatenated simulations of each TD channel) from the feature list. This typically refers to putative powerband edges. 
    Recall the features were collected from the fft subset indices.

    parameters:
    feature_list (list): The list of features.
    fft_subset_inds (list): The start and end indices of the fft subset that was used for feature selection.
    fft_length (int): The length of the simulated fft vector for each TD channel (e.g. 512 for an FFT window of 1024 on a 500 TD sampling rate).

    returns:
    list: The list of original fft indices.
    """
    vec_length = fft_subset_inds[1] - fft_subset_inds[0]
    vec_start = fft_subset_inds[0]

    return [(
        (i - (vec_length * (i // vec_length)) ) # remove channel offset of index
        + vec_start # add the start index of the fft vector subset
        + (i // vec_length) * fft_length) # recover channel offset of original fft vector
        for i in feature_list]


def get_df_from_pb_combos(pb_combos):
    """
    Takes in the powerband combinations and returns a polars dataframe with the powerband combinations as columns.
    parameters:
    pb_combos (list): A list of lists of lists. The first list is the combination of powerbands for the corresponding cluster amount, the second list is the individual powerbands,
    and the third list is the start and end indices of the powerband.

    returns:
    pl.DataFrame: A polars dataframe with the powerband combinations as columns.
    """
    tmp = [[list(sub_ele) for sub_ele in ele] for ele in pb_combos]
    tmp2 = []
    for ele in tmp:
        if len(ele) <= 4:
            _ = [ele.append([None,None]) for i in range(4-len(ele))]
            tmp2.append(ele)
        else:
            [tmp2.append(sub_ele) for sub_ele in ele]
    return pl.DataFrame(tmp2, schema=['PB1', 'PB2', 'PB3', 'PB4'])


def get_training_data_for_LDA(df_fft, pbs, update_rate, label_col='SleepStageBinary') -> Tuple(np.ndarray, np.ndarray):
   """
    Takes in the fft dataframe and the powerband combinations and returns a numpy array of the calculated powerbands, averaged over the update rate, and a numpy array of the labels.
    parameters:
    df_fft (pl.DataFrame): The fft dataframe.
    pbs (polards.DataFrame): The powerband combinations dataframe.
    update_rate (int): The update rate, of which to average the powerbands over.
    label_col (str): The label column name. Default is 'SleepStageBinary'.

    returns:
    X (np.ndarray): The calculated powerbands, averaged over the update rate.
    y (np.ndarray): The labels.
   """
   simulated_ffts = (
    df_fft
    .select(
            [
                pl.col('fft_vec').arr.slice(pbs[i][0], (pbs[i][1] - pbs[i][0] + 1)).arr.sum().alias(f'Power_Band{i+1}') 
                for i in range(len(pbs)) if pbs[i][0] is not None
            ] + 
            [pl.col(label_col)]
    )
    .with_row_count()
    .with_columns([
        pl.col('row_nr') // update_rate
    ])
    .groupby(['row_nr'])
    .agg(
        [
            pl.col(f'Power_Band{i+1}').mean() 
            for i in range(len(pbs)) if pbs[i][0] is not None
        ] + 
        [pl.col(label_col).last().alias(label_col)]
      )
   )
   
   X = simulated_ffts.select(pl.col("^Power_Band.*$")).to_numpy()
   y = simulated_ffts.select(pl.col(label_col)).to_numpy().squeeze()
   return X, y
    
# The below function is intended to identify potential powerband combinations for maximizing sleep stage classification via LDA (or alternative models). Executes the above functions in order. 
def hyperparameter_search_pipeline(device, parameters, sleep_stage_mapping, out_file_path, np_random_seed=0):
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
    sleep_stage_mapping (dict): The sleep stage mapping dictionary. This is used to binarize the sleep stages into 0 and 1.
    out_file_path (str): The path to save the parquet file to.
    """
    assert len(parameters['TD_Columns']) > 1, 'Must select at least 2 TD channels to run hyperparameter search pipeline.'

    print('Executing Device: ', device)
    np.random.seed(np_random_seed)
    con = duckdb.connect(database=DATABASE_PATH, read_only=True)
    
    df = get_device_as_pl_df(device, con, lazy=True)

    # Keep only desired TD columns
    df = df.select(pl.all().exclude( list(set(ALL_TD_COLUMNS) - set(parameters['TD_Columns'])) ) )
    print('Analyzing: ' + ', '.join(df.select(pl.col('^TD.*$')).columns))

    sessions = df.select('SessionIdentity').unique().collect().to_dict(
        as_series=False)['SessionIdentity']
    settings = get_settings_for_pb_calcs(device, con, sessions, 'SessionIdentity')
    gains = get_gains_from_settings_dict(
        settings, sessions, 'SessionIdentity', device, con)
    
    gains_tmp = []
    for channel in parameters['TD_Columns']:
        if channel == 'TD_BG':
            # TODO: TD_BG need not be gains[0]. Could be gains[1]... I should use the TDSettings table to get correct gain for subcortical channel
            gains_tmp.append(gains[0])
        if channel == 'TD_key2':
            gains_tmp.append(gains[1])
        if channel == 'TD_key3':
            gains_tmp.append(gains[2])
    gains = gains_tmp
    print(gains)

    print(settings)
    print('ASSUMING SUBCORTICAL CHANNEL IS CHANNEL 0')
    fft_length = settings['fft_numBins'][0]

    df_chunked = chunk_df_by_timesegment(df)
    print(df_chunked.columns) # here
    print(f'Simulating FFTs for TD chunks of size {settings["fft_size"][0] - settings["fft_size"][0]%250} samples')
    df_chunked = add_simulated_ffts(df_chunked, settings, gains)
    df_chunked = df_chunked.with_columns(pl.col('SleepStage').map_dict(sleep_stage_mapping).alias('SleepStageBinary'))

    NUM_TD_CHANNELS = len(parameters['TD_Columns'])

    df_chunked = df_chunked.collect()

    df_fft_subset, fft_corrs = get_fft_corrs(df_chunked, parameters['fft_subset_inds'], label_col='SleepStageBinary', td_columns=parameters['TD_Columns'])

    
    fft_subset_length = parameters['fft_subset_inds'][1] - parameters['fft_subset_inds'][0]
    # corr_df = pd.DataFrame({'Hz': 0.48*(np.arange(fft_subset_length) + parameters['fft_subset_inds'][0]), 'BG': fft_corrs[:fft_subset_length], 'key2': fft_corrs[fft_subset_length:fft_subset_length*2], 
    #                         'key3': fft_corrs[fft_subset_length*2:]}).melt(id_vars='Hz', value_vars=['BG', 'key2', 'key3'], var_name='key', value_name='corr')
    # corr_chart = alt.Chart(corr_df).mark_line().encode(x='Hz', y='corr', color='key').interactive()

    first_feature_group, X, y = get_train_data_by_corr_threshold(df_fft_subset, fft_corrs, parameters['corr_threshold'], label_col='SleepStageBinary')
    del(df_fft_subset)
    model = LinearDiscriminantAnalysis(solver='svd')
    print('Running Sequential Feature Selector...')
    sfs = SequentialFeatureSelector(model, n_features_to_select=parameters['num_features_to_select'], direction='forward', n_jobs=6, cv=5, scoring=parameters['sfs_scoring'])
    sfs.fit(X, y)


    second_feature_group = first_feature_group[sfs.get_support(indices=True)]
    second_feature_group_original_inds = np.array(recover_original_feature_inds(second_feature_group, parameters['fft_subset_inds'], fft_length)).squeeze()
    pb_combos, pb_chart = get_PB_combinations(second_feature_group_original_inds, max_clusters=parameters['max_clusters'])
    df_pbs_corrected = get_df_from_pb_combos(pb_combos)

    df_pbs_corrected = df_pbs_corrected.join(pl.DataFrame({'UpdateRate': [2, 5, 10, 15, 30]}), how='cross')

    fft_cols = [col for col in df_chunked.columns if 'fft' in col]

    df_chunked = df_chunked.select([
        pl.col('SleepStage'),
        pl.col('SleepStageBinary'),
        pl.col(fft_cols[0]).arr.concat(pl.col(fft_cols[1:])).alias('fft_vec')
    ])
    try:
        assert fft_length == df_chunked.select(
                pl.col('fft_vec').arr.lengths()
            ).unique().item() / NUM_TD_CHANNELS
    except AssertionError:
        print('fft_length is not equal to the length of the fft_vec column.')
        print('fft_length: ', fft_length)
        print('length of fft_vec column: ', df_chunked.select(
                pl.col('fft_vec').arr.lengths()
            ).unique().item() / NUM_TD_CHANNELS)


    print('Searching over powerband combos...')
    scores = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    scores_stds = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    for i in range(df_pbs_corrected.height):
        pbs = [value for value in df_pbs_corrected.select(pl.exclude('UpdateRate'))[i].to_dicts()[0].values()]
        X, y = get_training_data_for_LDA(df_chunked, pbs, df_pbs_corrected[i,'UpdateRate'], label_col='SleepStageBinary')

        # NOTE: 'roc_auc' gets label predictions with clf.predict_proba(X)[:, 1], which allows more thresholds to be tested. 
        # clf.predict_proba(X)[:, 1] is the probability of the positive class (1). clf.predict_proba(X)[:, 0] is the probability of the negative class (0)
        score_dict = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy', 'recall': 'recall', 'precision': 'precision'}

        cv_results = cross_validate(model, X, y, cv=5,
                        scoring=score_dict, n_jobs=5)
        
        [scores[k].extend([np.mean(v)]) for (k, v) in cv_results.items() if k in scores.keys()]
        [scores_stds[k].extend([np.std(v)]) for (k, v) in cv_results.items() if k in scores_stds.keys()]

        tnr = np.array(cv_results['test_balanced_accuracy']) * 2 - cv_results['test_recall']
        scores['test_tnr'].extend([np.mean(tnr)])
        scores_stds['test_tnr'].extend([np.std(tnr)])
        


    df_hyperparams = pl.concat([df_pbs_corrected, pl.DataFrame(scores).rename(
        {'test_accuracy': 'Acc', 'test_roc_auc': 'AUC', 'test_balanced_accuracy': 'BalAcc', 'test_recall': 'TPR', 'test_precision': 'Precision', 'test_tnr': 'TNR'}),
        pl.DataFrame(scores_stds).rename({'test_accuracy': 'Acc_std', 'test_roc_auc': 'AUC_std', 'test_balanced_accuracy': 'BalAcc_std', 'test_recall': 'TPR_std', 'test_precision': 'precision_std', 'test_tnr': 'TNR_std'})
        ], how='horizontal')

    df_hyperparams.write_parquet(out_file_path)
    # corr_chart.save(BASE_PATH + 'sleepstage_corr.png')

    # return (device, df_hyperparams, corr_chart.properties(title=f'{device}'), pb_chart.properties(title=f'{device}'))
    return (device, df_hyperparams, pb_chart.properties(title=f'{device}'))


# The below function is intended to identify potential powerband combinations for maximizing sleep stage classification via LDA (or alternative models). Executes the above functions in order. 
def hyperparameter_search_pipeline(device, parameters, sleep_stage_mapping, out_file_path, np_random_seed=0):
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
        'model' (sklearn model): The model to use for the cross validation. Can use custom models from 'em
        'correlation_threshold' (float): The correlation threshold for removing FFT bins with low correlation (below the threshold).
        'num_features_to_select' (int): The number of features to select in the SequentialForwardFeature selection.
        'max_clusters' (int): The maximum number of clusters to use for the k-means clustering.
    sleep_stage_mapping (dict): The sleep stage mapping dictionary. This is used to binarize the sleep stages into 0 and 1.
    out_file_path (str): The path to save the parquet file to.
    """
    assert len(parameters['TD_Columns']) > 1, 'Must select at least 2 TD channels to run hyperparameter search pipeline.'

    print('Executing Device: ', device)
    np.random.seed(np_random_seed)
    con = duckdb.connect(database=DATABASE_PATH, read_only=True)
    
    df = get_device_as_pl_df(device, con, lazy=True)

    # Keep only desired TD columns
    df = df.select(pl.all().exclude( list(set(ALL_TD_COLUMNS) - set(parameters['TD_Columns'])) ) )
    print('Analyzing: ' + ', '.join(df.select(pl.col('^TD.*$')).columns))

    sessions = df.select('SessionIdentity').unique().collect().to_dict(
        as_series=False)['SessionIdentity']
    settings = get_settings_for_pb_calcs(device, con, sessions, 'SessionIdentity')
    gains = get_gains_from_settings_dict(
        settings, sessions, 'SessionIdentity', device, con)
    
    gains_tmp = []
    for channel in parameters['TD_Columns']:
        if channel == 'TD_BG':
            # TODO: TD_BG need not be gains[0]. Could be gains[1]... I should use the TDSettings table to get correct gain for subcortical channel
            gains_tmp.append(gains[0])
        if channel == 'TD_key2':
            gains_tmp.append(gains[1])
        if channel == 'TD_key3':
            gains_tmp.append(gains[2])
    gains = gains_tmp
    print(gains)

    print(settings)
    print('ASSUMING SUBCORTICAL CHANNEL IS CHANNEL 0')
    fft_length = settings['fft_numBins'][0]

    df_chunked = chunk_df_by_timesegment(df)
    print(df_chunked.columns) # here
    print(f'Simulating FFTs for TD chunks of size {settings["fft_size"][0] - settings["fft_size"][0]%250} samples')
    df_chunked = add_simulated_ffts(df_chunked, settings, gains)
    df_chunked = df_chunked.with_columns(pl.col('SleepStage').map_dict(sleep_stage_mapping).alias('SleepStageBinary'))

    NUM_TD_CHANNELS = len(parameters['TD_Columns'])

    df_chunked = df_chunked.collect()

    df_fft_subset, fft_corrs = get_fft_corrs(df_chunked, parameters['fft_subset_inds'], label_col='SleepStageBinary', td_columns=parameters['TD_Columns'])

    
    fft_subset_length = parameters['fft_subset_inds'][1] - parameters['fft_subset_inds'][0]
    # corr_df = pd.DataFrame({'Hz': 0.48*(np.arange(fft_subset_length) + parameters['fft_subset_inds'][0]), 'BG': fft_corrs[:fft_subset_length], 'key2': fft_corrs[fft_subset_length:fft_subset_length*2], 
    #                         'key3': fft_corrs[fft_subset_length*2:]}).melt(id_vars='Hz', value_vars=['BG', 'key2', 'key3'], var_name='key', value_name='corr')
    # corr_chart = alt.Chart(corr_df).mark_line().encode(x='Hz', y='corr', color='key').interactive()

    first_feature_group, X, y = get_train_data_by_corr_threshold(df_fft_subset, fft_corrs, parameters['corr_threshold'], label_col='SleepStageBinary')
    del(df_fft_subset)
    model = LinearDiscriminantAnalysis(solver='svd')
    print('Running Sequential Feature Selector...')
    sfs = SequentialFeatureSelector(model, n_features_to_select=parameters['num_features_to_select'], direction='forward', n_jobs=6, cv=5, scoring=parameters['sfs_scoring'])
    sfs.fit(X, y)


    second_feature_group = first_feature_group[sfs.get_support(indices=True)]
    second_feature_group_original_inds = np.array(recover_original_feature_inds(second_feature_group, parameters['fft_subset_inds'], fft_length)).squeeze()
    pb_combos, pb_chart = get_PB_combinations(second_feature_group_original_inds, max_clusters=parameters['max_clusters'])
    df_pbs_corrected = get_df_from_pb_combos(pb_combos)

    df_pbs_corrected = df_pbs_corrected.join(pl.DataFrame({'UpdateRate': [2, 5, 10, 15, 30]}), how='cross')

    fft_cols = [col for col in df_chunked.columns if 'fft' in col]

    df_chunked = df_chunked.select([
        pl.col('SleepStage'),
        pl.col('SleepStageBinary'),
        pl.col(fft_cols[0]).arr.concat(pl.col(fft_cols[1:])).alias('fft_vec')
    ])
    try:
        assert fft_length == df_chunked.select(
                pl.col('fft_vec').arr.lengths()
            ).unique().item() / NUM_TD_CHANNELS
    except AssertionError:
        print('fft_length is not equal to the length of the fft_vec column.')
        print('fft_length: ', fft_length)
        print('length of fft_vec column: ', df_chunked.select(
                pl.col('fft_vec').arr.lengths()
            ).unique().item() / NUM_TD_CHANNELS)


    print('Searching over powerband combos...')
    scores = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    scores_stds = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    for i in range(df_pbs_corrected.height):
        pbs = [value for value in df_pbs_corrected.select(pl.exclude('UpdateRate'))[i].to_dicts()[0].values()]
        X, y = get_training_data_for_LDA(df_chunked, pbs, df_pbs_corrected[i,'UpdateRate'], label_col='SleepStageBinary')

        # NOTE: 'roc_auc' gets label predictions with clf.predict_proba(X)[:, 1], which allows more thresholds to be tested. 
        # clf.predict_proba(X)[:, 1] is the probability of the positive class (1). clf.predict_proba(X)[:, 0] is the probability of the negative class (0)
        score_dict = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy', 'recall': 'recall', 'precision': 'precision'}

        cv_results = cross_validate(model, X, y, cv=5,
                        scoring=score_dict, n_jobs=5)
        
        [scores[k].extend([np.mean(v)]) for (k, v) in cv_results.items() if k in scores.keys()]
        [scores_stds[k].extend([np.std(v)]) for (k, v) in cv_results.items() if k in scores_stds.keys()]

        tnr = np.array(cv_results['test_balanced_accuracy']) * 2 - cv_results['test_recall']
        scores['test_tnr'].extend([np.mean(tnr)])
        scores_stds['test_tnr'].extend([np.std(tnr)])
        


    df_hyperparams = pl.concat([df_pbs_corrected, pl.DataFrame(scores).rename(
        {'test_accuracy': 'Acc', 'test_roc_auc': 'AUC', 'test_balanced_accuracy': 'BalAcc', 'test_recall': 'TPR', 'test_precision': 'Precision', 'test_tnr': 'TNR'}),
        pl.DataFrame(scores_stds).rename({'test_accuracy': 'Acc_std', 'test_roc_auc': 'AUC_std', 'test_balanced_accuracy': 'BalAcc_std', 'test_recall': 'TPR_std', 'test_precision': 'precision_std', 'test_tnr': 'TNR_std'})
        ], how='horizontal')

    df_hyperparams.write_parquet(out_file_path)
    # corr_chart.save(BASE_PATH + 'sleepstage_corr.png')

    # return (device, df_hyperparams, corr_chart.properties(title=f'{device}'), pb_chart.properties(title=f'{device}'))
    return (device, df_hyperparams, pb_chart.properties(title=f'{device}'))