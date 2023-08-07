import polars as pl
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from scipy.spatial import distance
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import os

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
                        pl.col('^fft_.*$').list.slice(fft_ind_start, vec_length)
                    ])
                    .with_columns(
                        pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).list.to_struct().alias('fft_vec')
                    )
                    .unnest('fft_vec')
                    )


def get_fft_corrs(df_fft_vec, fft_subset_inds=[2,120], label_col='SleepStage', td_columns=['TD_BG', 'TD_key2', 'TD_key3'], corr_type = 'pearson') -> np.ndarray:
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

    #fft_ind_start = fft_subset_inds[0]
    vec_length = fft_subset_inds[1] - fft_subset_inds[0]

    # df_fft_vec = expand_fft_arrays_with_subset(df_chunked, fft_ind_start=fft_ind_start, vec_length=vec_length)

    # Could convert fft_corrs to be a pl.dataframe with discord trick by @ms. This would allow it to be lazy and save on memory.
    fft_corrs = df_fft_vec.select([
                        pl.corr(f"fft_bin_{i}", pl.col(label_col).cast(pl.Float64), method=corr_type).alias(f"field_{i}_corr") for i in range(vec_length*len(td_columns))
                    ]).to_numpy().squeeze()
    
    return fft_corrs


def get_train_data_by_corr_threshold(df, fft_corrs, corr_threshold, label_col='SleepStageBinary'):
    """
    Reduce features for powerband selection, by selecting the FFT bins that have a correlation above the threshold

    parameters:
    df (pl.DataFrame): The dataframe of time segments (i.e. individual fft windows are single rows) with simulated FFTs added as additional columns.
    corr_threshold (float): The threshold for the correlation between the FFT bins and the label column. Only FFT bins with a correlation above this threshold will kept for future feature selection.
    label_col (str): The column to use for the label, for which each frequency bin will be compared to in a pearson correlation measure. Default is 'SleepStageBinary'
    """
    X = df.select(pl.col("^fft_bin_.*$")).to_numpy()
    feature_group = np.argwhere(np.abs(fft_corrs) > corr_threshold)
    X = X[:, np.where(np.abs(fft_corrs) > corr_threshold)[0]]
    y = df.select(pl.col(label_col)).to_numpy().squeeze()
    return feature_group, X, y


def get_PB_combinations(features, max_clusters=8, fft_length=512, num_channels=3):
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
    chart = alt.vconcat().add_selection(alt.selection_interval(bind='scales', encodings=['x']))
    for i in range(2, max_clusters+1): 
        kmeans = KMeans(n_clusters=i).fit(features[:, np.newaxis])

        # Extract first and last index of each cluster, to be used as powerband boundaries
        pbs = [[features[np.argwhere(kmeans.labels_ == j).squeeze()].min(), 
                features[np.argwhere(kmeans.labels_ == j).squeeze()].max()] for j in range(i)]
        
        # TODO: REMOVE pbs THAT CROSS TD CHANNEL BOUNDARIES
        for j in range(1,num_channels):
            pbs = [pb 
                if np.all(~np.isin(np.arange(fft_length*j - 1, fft_length*j + 1), np.arange(pb[0], pb[1]+1))) 
                else [None, None] 
                for pb in pbs]
        
        cluster_df = pd.DataFrame({'x':features, 'y':np.zeros(features.shape[0]), 'color':kmeans.labels_})
        base = alt.Chart(cluster_df).mark_point().encode(x='x', y='y', color=alt.Color('color', scale=alt.Scale(scheme='category20b'))).interactive()
        # if i == 2:
        #     chart = base
        # else:
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


def get_features_by_KL_divergence_threshold(df, top_k, label_col='SleepStageBinary'):
    ## Get KL divergence scores for each FFT bin.
    # First log then standardize each FFT bin
    df = df.with_columns(
        pl.col("^fft_bin_.*$").log(base=10)
    ).with_columns(
        [
            (pl.col("^fft_bin_.*$") - pl.col("^fft_bin_.*$").mean())
            / pl.col("^fft_bin_.*$").std()
        ]
    )

    dists = df.partition_by(label_col)

    # Get the probability distributions for each FFT bin, for each parition
    hist_bins = np.arange(-5, 5.1, 0.1).tolist()

    # Each row is a probability distribution for a given FFT bin
    dists_0 = np.stack(
        [
            dists[0][col]
            .hist(bins=hist_bins)
            .select(pl.col("^.*_count$") / pl.col("^.*_count$").sum())
            .to_numpy()
            .squeeze()
            for col in dists[0].columns
            if "fft_bin" in col
        ]
    )
    dists_1 = np.stack(
        [
            dists[1][col]
            .hist(bins=hist_bins)
            .select(pl.col("^.*_count$") / pl.col("^.*_count$").sum())
            .to_numpy()
            .squeeze()
            for col in dists[1].columns
            if "fft_bin" in col
        ]
    )

    # Calculate the KL divergence scores for each FFT bin (i.e. each row)
    kl_scores = distance.jensenshannon(dists_0, dists_1, axis=1)

    top_kl_scores = (
        pl.DataFrame(
            {
                "fft_bins": [f"fft_bin_{idx}" for idx in range(kl_scores.size)],
                "kl_scores": kl_scores,
            }
        )
        .with_row_count()
        .top_k(top_k, by="kl_scores")
    )
    
    features = top_kl_scores["row_nr"].sort().to_numpy().tolist()
    
    return features


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
    return pl.DataFrame(tmp2, schema=['PB1_index', 'PB2_index', 'PB3_index', 'PB4_index'])


def get_training_data_from_fft_df(df_fft, pbs, label_col="SleepStageBinary"):
    """
    Takes in the fft dataframe and the powerband combinations and returns a numpy array of the calculated powerbands, averaged over the update rate, and a numpy array of the labels.
    parameters:
    df_fft (pl.DataFrame): The fft dataframe. This dataframe must have a column named 'fft_vec' that contains the concatenated fft vectors for each channel.
    pbs (polars.DataFrame): The powerband combinations dataframe.
    label_col (str): The label column name. Default is 'SleepStageBinary'.

    returns:
    X (np.ndarray): The calculated powerbands, averaged over the update rate.
    y (np.ndarray): The labels.
    """
    simulated_pbs = df_fft.select(
        [
            pl.col("fft_vec")
            .list.slice(pbs[i][0], (pbs[i][1] - pbs[i][0] + 1))
            .list.sum()
            .alias(f"Power_Band{i+1}")
            for i in range(len(pbs))
            if pbs[i][0] is not None
        ]
        + [pl.col(label_col)]
    )

    X = simulated_pbs.select(pl.col("^Power_Band.*$")).to_numpy()
    y = simulated_pbs.select(pl.col(label_col)).to_numpy().squeeze()
    return X, y
    
def sfs_cluster_pipeline(df, parameters, settings, out_path):
    """
    Executes the hyperparameter search pipeline for a given device. The pipeline is as follows:
    1. Correlate FFT bins with the binarized sleep stage labels.
    2. Remove FFT bins with low correlation (below the threshold)
    3. Use SequentialForwardFeature selection to get the n best remaining FFT bins for sleep stage classification.
    4. Cluster the n best FFT bins into 2 to k clusters.
    5. Get the powerband combinations for each cluster amount.
    6. Use the powerbands combinations and update rate to procure training data for LDAs.
    7. Run an LDA cross validation on each powerband combination.
    8. Get the cross-validated LDA model's scores for each powerband combination.

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
    out_path (str): The path to out figures and results to.
    """
    
    fft_subset_length = (
        parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    )
    
    if parameters['filtering_method'] == 'corr':
        # TODO: Figure out why multiple TD lines are being added to plot
        fft_corrs = get_fft_corrs(df, parameters['fft_subset_inds'], label_col='SleepStageBinary', td_columns=parameters['TD_Columns'], corr_type=parameters['corr_type'])
        # corr_df = pd.DataFrame({'Hz': settings['fft_binWidth'][0]*(np.arange(fft_subset_length) + parameters['fft_subset_inds'][0]), **{key: fft_corrs[fft_subset_length*ind:fft_subset_length*(ind+1)] for ind, key in enumerate(parameters['TD_Columns'])}}).melt(id_vars='Hz', value_vars=parameters['TD_Columns'], var_name='key', value_name='corr')
        # ax = sns.lineplot(data=corr_df, x='Hz', y='corr', hue='key')
        # plt.savefig(os.path.splitext(out_path)[0] + '_corr_chart.png')
        first_feature_group, X, y = get_train_data_by_corr_threshold(df, fft_corrs, parameters['corr_threshold'], label_col='SleepStageBinary')
    elif parameters['filtering_method'] == 'KL':
        first_feature_group = get_features_by_KL_divergence_threshold(df, parameters['num_features_to_filter'], label_col='SleepStageBinary')
    else:
        raise ValueError(f'Invalid feature filtering method: {parameters["feature_filtering"]}')
    
    model = parameters['model']
    print(f'Running Sequential Feature Selector... with {model}')
    sfs = SequentialFeatureSelector(model, n_features_to_select=parameters['num_features_to_select_sfs'], direction='forward', n_jobs=parameters['cross_val']+1, cv=parameters['cross_val'], scoring=parameters['sfs_scoring'])
    sfs.fit(X, y)

    # Cluster and combine powerbands in combinations
    features = np.stack(first_feature_group).squeeze()[sfs.get_support(indices=True)]
    # second_feature_group_original_inds = np.array(recover_original_feature_inds(second_feature_group, parameters['fft_subset_inds'], settings['fft_numBins'][0])).squeeze()
    pb_combos, pb_chart = get_PB_combinations(features, max_clusters=parameters['max_clusters'], fft_length=settings['fft_numBins'][0], num_channels=len(parameters['TD_Columns']))
    df_pbs = get_df_from_pb_combos(pb_combos)


    fft_cols = [col for col in df.columns if 'fft' in col]

    df = df.select(
        [
            pl.col("SleepStageBinary"),
            pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).alias("fft_vec")
        ]
    )


    print('Searching over powerband combos...')
    cv = StratifiedKFold(
        n_splits=parameters["cross_val"],
        random_state=parameters["random_state"],
        shuffle=True,
    )
    scores = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    scores_stds = {'test_accuracy': [], 'test_roc_auc': [], 'test_balanced_accuracy': [], 'test_recall': [], 'test_precision': [], 'test_tnr': []}
    score_dict = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
    }
    
    for i in range(df_pbs.height):
        if i % 10 == 0:
            print(f"Powerband combo {i+1}/{df_pbs.height}")

        pbs = [value for value in df_pbs[i].to_dicts()[0].values()]
        X, y = get_training_data_from_fft_df(
            df, pbs, label_col="SleepStageBinary"
        )

        cv_results = cross_validate(
            parameters["model"],
            X,
            y,
            cv=cv,
            scoring=score_dict,
            n_jobs=parameters["cross_val"] + 1,
        )

        [
            scores[k].extend([np.mean(v)])
            for (k, v) in cv_results.items()
            if k in scores.keys()
        ]
        [
            scores_stds[k].extend([np.std(v)])
            for (k, v) in cv_results.items()
            if k in scores_stds.keys()
        ]

        tnr = (
            np.array(cv_results["test_balanced_accuracy"]) * 2
            - cv_results["test_recall"]
        )
        scores["test_tnr"].extend([np.mean(tnr)])
        scores_stds["test_tnr"].extend([np.std(tnr)])
        
    df_pbs = pl.concat([df_pbs, pl.DataFrame(scores).rename(
        {'test_accuracy': 'Acc', 'test_roc_auc': 'AUC', 'test_balanced_accuracy': 'BalAcc', 'test_recall': 'TPR', 'test_precision': 'Precision', 'test_tnr': 'TNR'}),
        pl.DataFrame(scores_stds).rename({'test_accuracy': 'Acc_std', 'test_roc_auc': 'AUC_std', 'test_balanced_accuracy': 'BalAcc_std', 'test_recall': 'TPR_std', 'test_precision': 'precision_std', 'test_tnr': 'TNR_std'})
        ], how='horizontal')
    
    return df_pbs