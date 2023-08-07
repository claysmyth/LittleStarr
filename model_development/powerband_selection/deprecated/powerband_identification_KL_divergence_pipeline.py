from powerband_identification_for_LDA_pipeline_funcs import *
from powerband_identification_for_LDA_no_feature_eng import (
    convert_index_column_to_freq_using_binWidth,
)
from scipy.spatial import distance
import plotly.graph_objects as go
from plotly.colors import n_colors
import os
from sklearn import metrics
import numpy as np
from sklearn.model_selection import StratifiedKFold


def plot_feature_distributions(
    df, melt_cols, partition_col, value_name="Value", color_ordering=None
):
    """
    Plots the distributions of each desired column (melt_cols) in df by values in partition_col.
    parameters:
    df: polars dataframe.
    melt_cols: (list of strings) columns to melt. These columns will ultimate be rows in the resulting ridge plot.
    partition_col: (string) column to partition by. A separate distribution of each melt_col for each unique value in the partition_col will be plotted.
    value_name: (string) name of the resulting value column. This is the x-axis label in the ridge plot.
    color_ordering: (list of strings) order of the colors in the ridge plot. The elements in this list should correspond to unique values in the partition_col.
                    If None, the colors will be assigned in the order of the unique values in partition_col.

    returns:
    Plotly Go object.
    """
    data_long = df.melt(
        id_vars=[partition_col],
        value_vars=melt_cols,
        variable_name="Variable",
        value_name=value_name,
    )

    partitioned_data_long = data_long.partition_by(partition_col)

    colors = n_colors(
        "rgb(0, 0, 255)", "rgb(255, 0, 0)", len(partitioned_data_long), colortype="rgb"
    )

    if color_ordering is not None:
        color_map = {ele: colors[i] for i, ele in enumerate(color_ordering)}
        curr_ordering = {
            part[partition_col][0]: i for i, part in enumerate(partitioned_data_long)
        }
        # Reorder partitioned_data_long to match color_ordering
        partitioned_data_long = [
            partitioned_data_long[curr_ordering[ele]] for ele in color_ordering
        ]
    else:
        color_map = {
            ele: colors[i]
            for i, ele in enumerate(
                data_long[partition_col].unique().to_numpy().squeeze()
            )
        }

    fig = go.Figure()

    for i, partition in enumerate(partitioned_data_long):
        partition = partition.sort("Variable")
        x = partition[value_name].to_numpy().squeeze()
        y = partition["Variable"].to_numpy().squeeze()
        partition_name = partition[partition_col][0]
        fig.add_trace(
            go.Violin(
                x=x,
                y=y,
                legendgroup=partition_name,
                scalegroup=partition_name,
                name=partition_name,
                side="positive",
                marker_color=color_map[partition_name],
            )
        )

    fig.update_traces(
        orientation="h", side="positive", width=2, points=False, meanline_visible=True
    )
    fig.update_layout(
        xaxis_showgrid=False, xaxis_zeroline=False, title="Raw Distributions"
    )

    return fig


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


def hyperparameter_search_pipeline(
    device, parameters, sleep_stage_mapping, out_file_path, db_path
):
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
    # TODO: Do not rename the dataframe each time... or use polars LazyFrame
    # TODO: Consider settings aside a portion of the training set for identification of features to avoid data leakage in cross-validation of LDA.
    # Probably unnecessary though, as the LDA for feature selection has separate model weights than the one trained on powerbands.

    # Get the device's fft dataframe
    con = duckdb.connect(db_path, read_only=True)
    df_all = con.sql(f"select * from r{device}.overnight_simulated_FFTs;").pl()
    df_training = con.sql(
        f"select * from r{device}.overnight_simulated_FFTs_train;"
    ).pl()
    settings = (
        con.sql(f"select * from r{device}.overnight_simulated_FFTs_settings;")
        .pl()
        .to_dict(as_series=False)
    )

    # Exclude columns that are not desired (per parameters)
    cols_to_exclude = [
        ele
        for ele in ["TD_BG", "TD_key2", "TD_key3"]
        if ele not in parameters["TD_Columns"]
    ]
    df_training.select([])
    df_training = df_training.select(pl.col("^(SessionIdentity|fft_|Sleep).*$")).select(
        pl.exclude("|".join([f"^.*{col}.*$" for col in cols_to_exclude]))
    )
    df_all = df_all.select(pl.col("^(SessionIdentity|fft_|Sleep).*$")).select(
        pl.exclude("|".join([f"^.*{col}.*$" for col in cols_to_exclude]))
    )

    # Collect session identifiers
    sessions = df_all["SessionIdentity"].unique().to_list()

    # Remap sleep stages to binary classes
    df_training = df_training.with_columns(
        pl.col("SleepStage").map_dict(sleep_stage_mapping).alias("SleepStageBinary")
    )
    df_all = df_all.with_columns(
        pl.col("SleepStage").map_dict(sleep_stage_mapping).alias("SleepStageBinary")
    )

    # Retain only desired portions of each FFT
    fft_subset_length = (
        parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    )
    df_feature_sel = df_training.with_columns(
        [
            pl.col("^fft_.*$").list.slice(
                parameters["fft_subset_inds"][0], fft_subset_length
            )
        ]
    )

    # Average the FFTs over the update rate.
    # Note that the index of the FFT bin will start at the beginning of the slice from the previous step.
    # (i.e. fft_bin_0 will correpsond to the original FFT vector index of parameters['fft_subset_inds'][0])
    fft_cols = [col for col in df_training.columns if "fft" in col]
    df_all = df_all.select(
        [
            pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).alias("fft_vec"),
            pl.col("SessionIdentity"),
            # pl.col("^Power_Band.*$"),
            pl.col("SleepStageBinary"),
        ]
    )

    df_feature_sel = (
        df_feature_sel.select(
            [
                pl.col(fft_cols[0])
                .list.concat(pl.col(fft_cols[1:]))
                .list.to_struct(fields=lambda idx: f"fft_bin_{idx}")
                .alias("fft_vec"),
                # pl.col("^Power_Band.*$"),
                pl.col("SleepStageBinary"),
            ]
        )
        # Split the desired fft bins into individual columns
        .unnest("fft_vec")
        .with_row_count()
        # Group by the row number and average the FFT bins over the update rate
        .with_columns([pl.col("row_nr") // parameters["UpdateRate"]])
        .groupby(["row_nr"])
        .agg(
            # pl.col(f"^Power_Band.*").mean(),
            # 'fft_bin_#' columns correspond to the simulated FFT bins
            pl.col("^fft_bin_.*$").mean(),
            pl.col("SleepStageBinary").last().alias("SleepStageBinary"),
        )
    )

    ## Get KL divergence scores for each FFT bin.
    # First log then standardize each FFT bin
    df_feature_sel = df_feature_sel.with_columns(
        pl.col("^fft_bin_.*$").log(base=10)
    ).with_columns(
        [
            (pl.col("^fft_bin_.*$") - pl.col("^fft_bin_.*$").mean())
            / pl.col("^fft_bin_.*$").std()
        ]
    )

    dists = df_feature_sel.partition_by("SleepStageBinary")

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
        .top_k(parameters["num_features_to_select"], by="kl_scores")
    )
    # Alternative to top_k --> .sort('kl_scores', descending=True)

    # plot the distributions of the top KL scores
    fig = plot_feature_distributions(
        df_feature_sel.select(
            top_kl_scores["fft_bins"].to_list() + ["SleepStageBinary"]
        ),
        top_kl_scores["fft_bins"].to_list(),
        "SleepStageBinary",
    )
    fig.write_html(f"{os.path.dirname(out_file_path)}/feature_ridgeplot.html")

    # Get the original FFT bin indices for the top KL scores
    # TODO: Consider running a corr on the top KL scores to remove redundant features
    features = top_kl_scores["row_nr"].sort().to_numpy().tolist()
    # Select features with low correlation to each other
    # df_feature_sel.select(
    #     top_kl_scores['fft_bins'].to_list()
    # ).corr().select([pl.all().arg_min()]).melt(value_name='index').join(
    #                     df_feature_sel.select(
    #                     top_kl_scores['fft_bins'].to_list()
    #                 ).corr().select([pl.all().min()]).melt(value_name='corr'), how='left', on='variable'
    # )
    fft_bins_original_inds = np.array(
        recover_original_feature_inds(
            features, parameters["fft_subset_inds"], settings["fft_numBins"][0]
        )
    ).squeeze()
    pb_combos, pb_chart = get_PB_combinations(
        fft_bins_original_inds, max_clusters=parameters["max_clusters"]
    )
    df_pbs_corrected = get_df_from_pb_combos(pb_combos)
    # TODO: Make chart interaction uniform across all powerband charts
    pb_chart.save(f"{os.path.dirname(out_file_path)}/powerband_clusters.html")

    # Set up dictionaries to store scores for each powerband
    print("Searching over powerband combos...")
    scores = {
        "test_accuracy": [],
        "test_roc_auc": [],
        "test_balanced_accuracy": [],
        "test_recall": [],
        "test_precision": [],
        "test_tnr": [],
    }
    scores_stds = {
        "test_accuracy": [],
        "test_roc_auc": [],
        "test_balanced_accuracy": [],
        "test_recall": [],
        "test_precision": [],
        "test_tnr": [],
    }
    session_scores = {
        "session_accuracy": [],
        "session_AUC": [],
        "session_recall": [],
        "session_precision": [],
        "validation_session": [],
    }

    score_dict = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
    }

    fft_cols = [col for col in df_training.columns if "fft" in col]
    training_df = df_training.select(
        [
            pl.col("SleepStageBinary"),
            pl.col(fft_cols[0]).list.concat(pl.col(fft_cols[1:])).alias("fft_vec"),
        ]
    )

    # Iterate over each powerband combination and get the scores
    cv = StratifiedKFold(
        n_splits=parameters["cross_val"],
        random_state=parameters["random_state"],
        shuffle=True,
    )
    for i in range(df_pbs_corrected.height):
        if i % 10 == 0:
            print(f"Powerband combo {i+1}/{df_pbs_corrected.height}")

        pbs = [value for value in df_pbs_corrected[i].to_dicts()[0].values()]
        X, y = get_training_data_from_fft_df(
            training_df, pbs, label_col="SleepStageBinary"
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

        if parameters["session_cross_validate"]:
            # Get the scores for the validation session
            session_cross_validate = leave_one_session_out_cross_validation(
                df_all, sessions, pbs, parameters
            )

            session_scores["session_accuracy"].append(
                session_cross_validate["Accuracy"]
            )
            session_scores["session_AUC"].append(session_cross_validate["AUC"])
            session_scores["session_precision"].append(
                session_cross_validate["Precision"]
            )
            session_scores["session_recall"].append(session_cross_validate["Recall"])
            session_scores["validation_session"].append(
                session_cross_validate["Validation_Session"]
            )

    df_pbs_corrected = df_pbs_corrected.with_columns(
        [
            pl.col(pb)
            .map(
                lambda x: pl.Series(
                    convert_index_column_to_freq_using_binWidth(
                        parameters["TD_Columns"],
                        np.stack(x.to_numpy()),
                        settings["fft_numBins"][0],
                        settings["fft_binWidth"][0],
                    )
                )
            )
            .alias(f"{pb[:3]}")
            for pb in df_pbs_corrected.columns
            if "PB" in pb
        ]
    )
    # .with_columns([
    #     pl.col('PB1').struct.rename_fields(['PB1_channel', 'PB1_freq_range']),
    #     pl.col('PB2').struct.rename_fields(['PB2_channel', 'PB2_freq_range']),
    #     pl.col('PB3').struct.rename_fields(['PB3_channel', 'PB3_freq_range']),
    #     pl.col('PB4').struct.rename_fields(['PB4_channel', 'PB4_freq_range'])
    # ])

    df_hyperparams = pl.concat(
        [
            df_pbs_corrected,
            pl.DataFrame(scores).rename(
                {
                    "test_accuracy": "Acc",
                    "test_roc_auc": "AUC",
                    "test_balanced_accuracy": "BalAcc",
                    "test_recall": "TPR",
                    "test_precision": "Precision",
                    "test_tnr": "TNR",
                }
            ),
            pl.DataFrame(scores_stds).rename(
                {
                    "test_accuracy": "Acc_std",
                    "test_roc_auc": "AUC_std",
                    "test_balanced_accuracy": "BalAcc_std",
                    "test_recall": "TPR_std",
                    "test_precision": "precision_std",
                    "test_tnr": "TNR_std",
                }
            ),
        ],
        how="horizontal",
    )

    if parameters["session_cross_validate"]:
        df_hyperparams = pl.concat(
            [df_hyperparams, pl.DataFrame(session_scores)],
            how="horizontal",
        )

        df_hyperparams = df_hyperparams.with_columns(
            [
                pl.col("^session.*$").list.mean().suffix("_mean"),
                pl.col("^session.*$")
                .apply(lambda x: np.std(x.to_list()))
                .suffix("_std"),
            ]
        )

    df_hyperparams.write_parquet(out_file_path)
    # df_hyperparams.write_csv(out_file_path)
