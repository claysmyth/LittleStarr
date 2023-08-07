import polars as pl
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
import numpy as np


def get_data_for_sfs(df, label_col="SleepStageBinary"):
    y = df[label_col].to_numpy().squeeze()
    X = df.select(pl.exclude(label_col))
    return X, y


def create_feature_matrix(X_df, pbs):
    # Potentially deprecated
    cols_in_pbs = []
    [
        cols_in_pbs.extend([f"fft_bin_{i}" for i in range(pb[0], pb[1] + 1)])
        for pb in pbs
    ]

    return X_df.with_columns(
        [
            pl.sum_horizontal([f"fft_bin_{i}" for i in range(pb[0], pb[1] + 1)]).alias(
                f"PB_{pb[0]}_{pb[1]}"
            )
            for pb in pbs
        ]
    ).select(pl.exclude(cols_in_pbs))


def create_feature_matrix_mask(X_df, pbs):
    cols_in_pbs = []
    [
        cols_in_pbs.extend([f"fft_bin_{i}" for i in range(pb[0], pb[1] + 1)])
        for pb in pbs
    ]

    return X_df.with_columns(
        [
            pl.sum_horizontal([f"fft_bin_{i}" for i in range(pb[0], pb[1] + 1)]).alias(
                f"PB_{pb[0]}_{pb[1]}"
            )
            for pb in pbs
        ]
        + [
            (pl.col(col) - pl.col(col) + np.random.normal(size=X_df.height)).alias(col)
            for col in cols_in_pbs
        ]
    )


def sfs_pass(X, y, n_features_to_select, parameters, use_sklearn_sfs=False):
    model = parameters["model"]
    if use_sklearn_sfs:
        sfs = SequentialFeatureSelector(
            estimator=model,
            n_features_to_select=n_features_to_select,
            direction="forward",
            n_jobs=parameters["cross_val"] + 1,
            cv=parameters["cross_val"],
            scoring=parameters["sfs_scoring"],
        )

        sfs.fit(X, y)

        return sfs.get_support(indices=True)
    else:
        # scores = [np.mean(cross_validate(model, X[:,k].reshape(-1,1), y, cv=parameters['cross_val'],
        #             scoring=parameters['sfs_scoring'], n_jobs=parameters['cross_val']+1)['test_score']) for k in range(X.shape[-1])]
        if n_features_to_select > 1:
            previous_pbs = X[:, -n_features_to_select + 1 :]
            features_to_explore = X[:, : -n_features_to_select + 1]
            scores = [
                np.mean(
                    cross_validate(
                        model,
                        np.hstack(
                            (features_to_explore[:, k].reshape(-1, 1), previous_pbs)
                        ),
                        y,
                        cv=parameters["cross_val"],
                        scoring=parameters["sfs_scoring"],
                        n_jobs=parameters["cross_val"] + 1,
                    )["test_score"]
                )
                for k in range(features_to_explore.shape[-1])
            ]
        else:
            scores = [
                np.mean(
                    cross_validate(
                        model,
                        X[:, k].reshape(-1, 1),
                        y,
                        cv=parameters["cross_val"],
                        scoring=parameters["sfs_scoring"],
                        n_jobs=parameters["cross_val"] + 1,
                    )["test_score"]
                )
                for k in range(X.shape[-1])
            ]

        return [np.argsort(scores)[-1]]


def cv_on_powerbands(X_df, y, pbs, parameters, scoring=None):
    # NOTE: pb_inds is INCLUSIVE (i.e. [19,20] means powerband of sum of bins 19 and 20)
    X_tmp = X_df.select(
        [
            pl.sum_horizontal([f"fft_bin_{i}" for i in range(pb[0], pb[1] + 1)]).alias(
                f"PB_{pb[0]}_{pb[1]}"
            )
            for pb in pbs
        ]
    ).to_numpy()

    if scoring is None:
        scoring_method = parameters["sfs_scoring"]
    else:
        scoring_method = scoring

    cv_results = cross_validate(
        parameters["model"],
        X_tmp,
        y,
        cv=parameters["cross_val"],
        scoring=scoring_method,
        n_jobs=parameters["cross_val"] + 1,
    )

    if isinstance(scoring_method, str):
        return cv_results["test_score"]
    else:
        return cv_results


def mask_data_based_on_channel_constraint(X, pbs, parameters):

    MAX_NUMBER_OF_POWERBANDS_PER_CHANNEL = 2

    fft_subset_length = (
        parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    )

    # Only need first index of powerband to check which channel it is from
    pbs = np.array(pbs).flatten()[::2]

    for j in range(len(parameters["TD_Columns"])):
        if (
            np.sum(
                np.isin(
                    pbs, np.arange(j * fft_subset_length, (j + 1) * fft_subset_length)
                )
            )
            == MAX_NUMBER_OF_POWERBANDS_PER_CHANNEL
        ):
            cols = [
                f"fft_bin_{i}"
                for i in range(j * fft_subset_length, (j + 1) * fft_subset_length)
            ]
            X = X.select(pl.exclude(cols))

    return X


def grow_powerband(X_df, y, peak_index, parameters):
    """
    Grow powerband by one bin and see if it improves performance.
    """
    # TODO: Include previous powerbands as features
    # initialize variables
    score_improvement = np.inf
    curr_score = np.mean(
        cv_on_powerbands(X_df, y, [peak_index, peak_index], parameters)
    )

    # Lower indicates the powerband with the lower frequency bin, higher indicates the powerband with the higher frequency bin
    pb_inds_lower = (peak_index - 1, peak_index)
    pb_inds_higher = (peak_index, peak_index + 1)

    max_index = (
        parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    ) * len(parameters["TD_Columns"])
    min_index = 0

    # Check if powerband is at the edge of the frequency range
    if pb_inds_lower[0] < min_index:
        pb_inds_lower = (min_index, pb_inds_lower[1])

    if pb_inds_higher[1] > max_index:
        pb_inds_higher = (pb_inds_higher[0], max_index)

    curr_pb = peak_index

    while score_improvement > parameters["score_improvement_threshold"]:
        score_lower = np.mean(cv_on_powerbands(X_df, y, pb_inds_lower, parameters))
        score_higher = np.mean(cv_on_powerbands(X_df, y, pb_inds_higher, parameters))

        new_score = np.max([score_lower, score_higher])
        score_improvement = new_score - curr_score
        curr_score = new_score

        if score_lower > score_higher:
            curr_pb = pb_inds_lower
            pb_inds_lower = (curr_pb[0] - 1, curr_pb[1])
            pb_inds_higher = (curr_pb[0], curr_pb[1] + 1)
        else:
            curr_pb = pb_inds_higher
            pb_inds_lower = (curr_pb[0] - 1, curr_pb[1])
            pb_inds_higher = (curr_pb[0], curr_pb[1] + 1)

        # Check if powerband is at the edge of the frequency range
        if pb_inds_lower[0] < min_index:
            pb_inds_lower = (min_index, pb_inds_lower[1])

        if pb_inds_higher[1] > max_index:
            pb_inds_higher = (pb_inds_higher[0], max_index)

    return curr_pb


def get_candidate_pbs_around_peak(max_pb_width, peak_index, parameters):
    """
    Get all possible powerband combinations around the peak of the SFS.
    params:
        pb_width (int): width of powerband, in number of bins
    """
    pb_candidates = [[peak_index, peak_index]]
    for i in range(1, max_pb_width + 1):
        for j in range(i):
            pb_candidates.append([peak_index - j, peak_index - j + i])

    fft_length = parameters["fft_subset_inds"][1] - parameters["fft_subset_inds"][0]
    for pb in pb_candidates:
        max_index = fft_length * ((peak_index // fft_length) + 1) - 1
        min_index = fft_length * (peak_index // fft_length)
        if pb[0] < min_index:
            pb[0] = min_index
        if pb[1] > max_index:
            pb[1] = max_index

    return pl.DataFrame({"PBs": pb_candidates}).unique()


def get_best_pb(putative_pb_df, X_df, y, pbs, parameters):
    scores = {"scores": []}
    # X_df = pl.from_numpy(X, schema=[f'fft_bin_{i}' for i in range(X.shape[1])], orient='col')
    for j in range(putative_pb_df.height):
        tmp_pbs = pbs.copy()
        # Skip overlapping powerbands
        if np.any(
            np.isin(
                np.array([tmp_pbs]).flatten(),
                np.arange(putative_pb_df["PBs"][j][0], putative_pb_df["PBs"][j][1] + 1),
            )
        ):
            scores["scores"].append(-np.inf)
        else:
            tmp_pbs.append(putative_pb_df["PBs"][j].to_list())
            scores["scores"].append(
                np.mean(cv_on_powerbands(X_df, y, tmp_pbs, parameters))
            )

    scores_df = pl.concat([putative_pb_df, pl.DataFrame(scores)], how="horizontal")

    return (
        scores_df.sort("scores", descending=True)["PBs"][0].to_list(),
        scores_df.sort("scores", descending=True)["scores"][0],
    )


def sfs_band_pipeline(df, parameters, impose_channel_constraint):
    """
    Run SFS band selection on the input dataframe.
    params:
        df (pl.DataFrame): input training data dataframe
        pb_width (int): width of powerband, in number of bins
        impose_channel_constraint: whether to impose the constraint that the powerbands selected must be implementable on the RC+S device
            e.g. You cannot have 3 powerbands come from the same Time Domain channel (TD_BG, TD_key2, TD_key3)
    """
    # TODO: Use sklearn.model_selection.Kfold to do cross-validation across SFS, in order to keep the same folds across SFS

    X_df, y = get_data_for_sfs(df, label_col="SleepStageBinary")
    pbs = []

    pb_selection = parameters["powerband_selection"]
    sfs_scores = []

    for i in range(1, 5):

        # Note that X is still a pl.DataFrame
        X = create_feature_matrix(X_df, pbs)

        if i == 1:
            top_features = sfs_pass(X.to_numpy(), y, i, parameters)
        else:
            if impose_channel_constraint:
                X = mask_data_based_on_channel_constraint(X, pbs, parameters)
                top_features = sfs_pass(X.to_numpy(), y, i, parameters)
            else:
                top_features = sfs_pass(X.to_numpy(), y, i, parameters)

            # Get the top feature original indices, prior to collapsing powerbands
            top_features = np.array(
                [
                    int(i.split("_")[-1])
                    for i in np.array(X.columns)[top_features]
                    if "PB" not in i
                ]
            )

        if pb_selection.lower() == "grow":
            # ! Assumes top_features[0] is the new top feature...
            # ! grow_powerband is not complete yet.. need to add previous pbs as features
            pb = grow_powerband(X_df, y, top_features[0], parameters)
        elif pb_selection.lower() == "search":
            putative_pb_df = get_candidate_pbs_around_peak(
                parameters["max_pb_width"], top_features[0], parameters
            )
            pb, score = get_best_pb(putative_pb_df, X_df, y, pbs, parameters)
        else:
            raise ValueError("pb_selection must be either 'grow' or 'search'")

        pbs.append(pb)
        sfs_scores.append(score)

    # Get the final cross-validated results of the powerband combination
    score_dict = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
    }
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

    pbs_results = cv_on_powerbands(X_df, y, pbs, parameters, score_dict)

    [
        scores[k].extend([np.mean(v)])
        for (k, v) in pbs_results.items()
        if k in scores.keys()
    ]
    [
        scores_stds[k].extend([np.std(v)])
        for (k, v) in pbs_results.items()
        if k in scores_stds.keys()
    ]

    tnr = (
        np.array(pbs_results["test_balanced_accuracy"]) * 2 - pbs_results["test_recall"]
    )
    scores["test_tnr"].extend([np.mean(tnr)])
    scores_stds["test_tnr"].extend([np.std(tnr)])

    df_pbs = pl.concat(
        [
            pl.DataFrame(
                {f"PB{i}_index": [pb] for i, pb in enumerate(pbs)}
                | {"sfs_scores": [sfs_scores]}
            ),
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

    return df_pbs
