import duckdb
import polars as pl
from sklearn import metrics
import numpy as np
from powerband_identification_sfs_bands import sfs_band_pipeline
from powerband_identification_sfs_cluster import sfs_cluster_pipeline

def load_training_data_from_db(device, db_path, session_validation=False):
    """
    Load data from RCS DB
    """
    con = duckdb.connect(db_path, read_only=True)
    
    # Only collect the entire dataset if session_validation is True, for leave-one-session-out cross-validation
    if session_validation:
        df = con.sql(f"select * from r{device}.overnight_simulated_FFTs;").pl()
    else:
        df = None
    
    # Collect training data and settings    
    df_training = con.sql(
        f"select * from r{device}.overnight_simulated_FFTs_train;"
    ).pl()
    
    settings = (
        con.sql(f"select * from r{device}.overnight_simulated_FFTs_settings;")
        .pl()
        .to_dict(as_series=False)
    )
    
    con.close()
    return df, df_training, settings


def slice_fft_vectors(df, parameters):
    # Retain only desired portions of each FFT
    fft_subset_length = (
        parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    )
    return df.with_columns(
        [
            pl.col("^fft_.*$").list.slice(
                parameters["fft_subset_inds"][0], fft_subset_length
            )
        ]
    )


def implement_update_rate(df, fft_cols, update_rate):
    return (
        df.select(
            [
                pl.col(fft_cols[0])
                .list.concat(pl.col(fft_cols[1:]))
                .list.to_struct(fields=lambda idx: f"fft_bin_{idx}")
                .alias("fft_vec"),
                pl.col("SleepStageBinary"),
            ]
        )
        # Split the desired fft bins into individual columns
        .unnest("fft_vec")
        .with_row_count()
        # Group by the row number and average the FFT bins over the update rate
        .with_columns([pl.col("row_nr") // update_rate])
        .groupby(["row_nr"])
        .agg(
            # pl.col(f"^Power_Band.*").mean(),
            # 'fft_bin_#' columns correspond to the simulated FFT bins
            pl.col("^fft_bin_.*$").mean(),
            pl.col("SleepStageBinary").last().alias("SleepStageBinary"),
        ).select(pl.all().exclude('row_nr'))
    )


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
        if i is not None else None
        for i in feature_list]
    

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


def get_training_data_for_model(df_fft, pbs, update_rate, label_col='SleepStageBinary'):
    """
    Takes in the fft dataframe and the powerband combinations and returns a numpy array of the calculated powerbands, averaged over the update rate, and a numpy array of the labels.
    parameters:
    df_fft (pl.DataFrame): The fft dataframe.
    pbs (polars.DataFrame): The powerband combinations dataframe.
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
                pl.col('fft_vec').list.slice(pbs[i][0], (pbs[i][1] - pbs[i][0] + 1)).list.sum().alias(f'Power_Band{i+1}') 
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

def leave_one_session_out_cross_validation(df, sessions, pbs, parameters):
    # Perform leave-one-session-out cross validation
    session_acc = []
    session_auc = []
    session_precision = []
    session_recall = []
    validation_session = []
    for session in sessions:
        X, y = get_training_data_for_model(
            df.filter(pl.col("SessionIdentity") != session),
            pbs,
            parameters["UpdateRate"],
            label_col="SleepStageBinary",
        )
        parameters["model"].fit(X, y)
        X_val, y_val = get_training_data_for_model(
            df.filter(pl.col("SessionIdentity") == session),
            pbs,
            parameters["UpdateRate"],
            label_col="SleepStageBinary",
        )

        if np.unique(y_val).shape[0] > 1:
            y_pred = parameters["model"].predict(X_val)
            y_pred_prob = parameters["model"].predict_proba(X_val)

            validation_session.append(session)
            session_acc.append(metrics.accuracy_score(y_val, y_pred))
            session_auc.append(metrics.roc_auc_score(y_val, y_pred_prob[:, 1]))
            session_precision.append(metrics.precision_score(y_val, y_pred))
            session_recall.append(metrics.recall_score(y_val, y_pred))

    return {
        "Validation_Session": validation_session,
        "Accuracy": session_acc,
        "AUC": session_auc,
        "Precision": session_precision,
        "Recall": session_recall,
    }
    
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
            if np.size(powerband_index) == 1: # make sure that the index is only one value, otherwise this powerband spans multiple channels.
                powerband_dicts.append({'Channel': td_cols[powerband_index[0]], 'freq_range': powerbands_freq[i, :].tolist()})
            else:
                powerband_dicts.append({'Channel': 'Invalid', 'freq_range': [-1.0, -1.0]})

    return powerband_dicts


def powerband_identification_pipeline(device, parameters, sleep_stage_mapping, out_file_path, db_path, session_validation=False):
    # Prepare training data, labels, and settings 
    np.random.seed(parameters["random_state"])
    df, df_training, settings = load_training_data_from_db(device, db_path, session_validation=session_validation)
    
    # Exclude columns that are not desired (per parameters), and preprocess dataframe
    cols_to_exclude = [
        ele
        for ele in ["TD_BG", "TD_key2", "TD_key3"]
        if ele not in parameters["TD_Columns"]
    ]
    df_training = df_training.select(pl.col("^(SessionIdentity|fft_|Sleep).*$")).select(
        pl.exclude("|".join([f"^.*{col}.*$" for col in cols_to_exclude]))
    )
    
    # Remap sleep stages to binary classes
    df_training = df_training.with_columns(
        pl.col("SleepStage").map_dict(sleep_stage_mapping).alias("SleepStageBinary")
    )
    
    if session_validation:
        # Repeat above steps for the entire dataset
        df = df.select(pl.col("^(SessionIdentity|fft_|Sleep).*$")).select(
            pl.exclude("|".join([f"^.*{col}.*$" for col in cols_to_exclude]))
        )
        df = df.with_columns(
            pl.col("SleepStage").map_dict(sleep_stage_mapping).alias("SleepStageBinary")
        )
    
    # Remove unwanted portions of FFT vectors
    df_training = slice_fft_vectors(df_training, parameters)
    
    # Implement update rate on FFT vectors
    fft_cols = [col for col in df_training.columns if "fft" in col]
    df_training = implement_update_rate(df_training, fft_cols, parameters["UpdateRate"])
    
    # Identify powerbands via desired method
    if parameters['method'] == 'sfs_cluster':
        # if parameters['filtering_method'] == 'corr':
            # Run spearman correlation to filter features...
        df_pbs = sfs_cluster_pipeline(df_training, parameters, settings, out_file_path)
        # elif parameters['filtering_method'] == 'KL_divergence':
        #     # Run KL divergence to filter features...
        #     None
        # else:
        #     raise ValueError(f'Invalid selection method: {parameters["filtering_method"]}')
        
        # TODO: Take only top K PB combinations
        if parameters['sfs_scoring'] == 'roc_auc':
            scoring = 'AUC'
        else:
            scoring = parameters['sfs_scoring'].upper()
            
        df_pbs = df_pbs.top_k(50, by=scoring)
        
    elif parameters['method'] == 'sfs_band':
        # Process data into format for SFS band selection: Unpack fft_bin struct into columns
        # df_training = df_training.rename({f"fft_bin_{i}": f"col_{i}" for i in df_training.columns if "fft_bin" in i})
        df_pbs = sfs_band_pipeline(df_training, parameters, parameters['impose_channel_constraint'])
    else:
        raise ValueError(f'Invalid powerband identification method: {parameters["method"]}')
    

    # second_feature_group_original_inds = np.array(recover_original_feature_inds(second_feature_group, parameters['fft_subset_inds'], settings['fft_numBins'][0])).squeeze()
    df_pbs = df_pbs.with_columns([pl.col(col).apply(lambda x: recover_original_feature_inds(x.to_list(), parameters['fft_subset_inds'], settings['fft_numBins'][0]))
                                for col in df_pbs.columns if 'PB' in col])

    
    # If Leave-One-Session-Out cross-validation: 
    # Repeat processing steps for the entire dataset, then train on all sessions except one, 
    # validating on the remaining session
    if session_validation:
        # Collect session identifiers
        sessions = df["SessionIdentity"].unique().to_list()
        
        # Create dict to store session validation scores
        session_scores = {
            "session_accuracy": [],
            "session_AUC": [],
            "session_recall": [],
            "session_precision": [],
            "validation_session": [],
        }
        
        # Concatenate FFT vectors into single column to simplify powerband calculations
        df = df.with_columns(
                        pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).alias('fft_vec')
                    )
        
        for i in range(df_pbs.height):
            pbs = [value for value in df_pbs.select(pl.col('^PB.*$'))[i].to_dicts()[0].values()]
            # Get the scores for the validation session
            # TODO: Make sure last index in PB is included...
            session_cross_validate = leave_one_session_out_cross_validation(
                df, sessions, pbs, parameters
            )

            session_scores["session_accuracy"].append(
                session_cross_validate["Accuracy"]
            )
            session_scores["session_AUC"].append(
                session_cross_validate["AUC"]
            )
            session_scores["session_precision"].append(
                session_cross_validate["Precision"]
            )
            session_scores["session_recall"].append(
                session_cross_validate["Recall"]
            )
            session_scores["validation_session"].append(
                session_cross_validate["Validation_Session"]
            )
        
        df_pbs = pl.concat(
            [df_pbs, pl.DataFrame(session_scores)],
            how="horizontal",
        )

        df_pbs = df_pbs.with_columns(
            [
                pl.col("^session.*$").list.mean().suffix("_mean"),
                pl.col("^session.*$")
                .apply(lambda x: np.std(x.to_list()))
                .suffix("_std"),
            ]
        )
    
    # Add frequency range columns to df_pbs

    df_pbs = df_pbs.with_columns([
        pl.col(pb).map(
            lambda x: pl.Series(convert_index_column_to_freq_using_binWidth(parameters['TD_Columns'], np.stack(x.to_numpy()), settings['fft_numBins'][0], settings['fft_binWidth'][0]))
        ).alias(f"{pb[:3]}_channel_freq_range") 
        for pb in df_pbs.columns if 'PB' in pb
    ])
    
    df_pbs.write_parquet(out_file_path)