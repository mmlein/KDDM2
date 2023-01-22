import matplotlib.colors as mcolors
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import pyod
from pathlib import Path
from sklearn.model_selection import train_test_split
from pyod.models.iforest import IForest
import string
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")

path_data = Path("Data/Input_Data/dataframe_with_outliers_3std_4.16%.csv").absolute()

columns = ["", "letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
        "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx", "outlier"]

letter_df_complete = pd.read_csv(path_data)

columns_train = columns[2:-1]
add_columns = ["Anomaly_Score", "projection"]
columns_df_res = columns + add_columns
columns_df_res.remove("letter")

print(letter_df_complete)

def prepare_dataset(letter: str, dataset: pd.DataFrame, test_prob: float = 0.2,
                    rand_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filters dataset for specific letter and splits into training and testing dataset.

    Args:
        letter (str): letter of interest.
        dataset (pd.DataFrame): dataframe containing the data.
        test_prob (float): percentage of data reserved for testing.
        rand_stat (int): random state for splitting the dataset.

    Returns:
        pd.DataFrame: dataframe filtered on letter.
        pd.DataFrame: dataframe for training.
        pd.DataFrame: dataframe for testing.
    """

    # Filter for the letter
    dataset_letter = dataset[dataset["letter"] == letter]

    # Create a copy of the dataset without the letter
    dataset_wo_let = dataset_letter.drop('letter', axis=1)

    # Split the dataset into training and test data
    letter_train, letter_test = train_test_split(
        dataset_wo_let, test_size=test_prob, train_size=(1-test_prob), random_state=rand_state)

    return dataset_letter, letter_train, letter_test


def isolation_forest(train_data: pd.DataFrame, test_data: pd.DataFrame, columns: list[str] = columns_train,
                     cont: float = 0.05, max_feat: int = 1.0, max_samp: int = 40, n_est: int = 100,
                     random_state: int = 0) -> tuple[pyod.models.iforest, np.ndarray, np.ndarray]:
    """
    Creates a new isolation forest.

    Args:
        train_data (pd.DataFrame): dataframe for training.
        test_data (pd.DataFrame): dataframe for testing.
        columns (list[str]): columns containing the training features.
        cont (float): contamination (estimated % of outliers).
        max_feat (int): features to train the isolation forest.
        max_samp (int): samples to train the isolations forest (impacts the tree size).
        n_est (int): numbers of trees in the ensemble.
        random_stat (int): random state.

    Returns:
        pyod.models.iforest: isolation forest.
        numpy.ndarray: numpy array containing training scores.
        numpy.ndarray: numpy array containing testing scores.
    """
    # Create a new iForest
    isft = IForest(behaviour='new', contamination=cont,
                   max_features=max_feat, max_samples=max_samp, n_estimators=n_est)

    # Fit iForest
    isft.fit(train_data[columns])

    # Training data
    y_train_scores = isft.decision_function(train_data[columns])
    y_train_pred = isft.predict(train_data[columns])

    # Test data
    y_test_scores = isft.decision_function(test_data[columns])
    y_test_pred = isft.predict(test_data[columns])

    return isft, y_train_scores, y_test_scores, y_train_pred, y_test_pred


def categorization(dataset: pd.DataFrame, pred_score: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Categorizes data into outliers and normal data.

    Args:
        dataset (pd.DataFrame): (training/testing) dataset.
        pred_score (np.ndarray): predictive scores from the isolation forest.
        threshold (float): threshold for classifying a datapoint as an outlier.

    Returns:
        pd.DataFrame: dataframe containing the statistics.
    """

    dataset['Anomaly_Score'] = pred_score
    dataset['projection'] = np.where(
        dataset['Anomaly_Score'] < threshold, 0, 1)

    return (dataset)


def descriptive_stat_threshold(dataset: pd.DataFrame) -> tuple[int, int, float]:
    """
    Determines the number of outliers and the percentage of outliers.
    
    Args:
        dataset (pd.DataFrame): (training/testing) dataset.

    Returns:
        int: length of the dataset.
        int: number of outliers.
        float: percentage of outliers.
    """

    total_datapoints = len(dataset)
    total_outliers = dataset["projection"].sum()
    percentage = total_outliers/total_datapoints

    return total_datapoints, total_outliers, percentage


def algorithm(dataset_complete: pd.DataFrame, cont: float = 0.05, max_feat: int = 1.0,
              max_samp: int = 40, n_est: int = 100, random_state: int = 0, write_to_csv: bool = False) -> tuple[pd.DataFrame,
                                                                                                                pd.DataFrame]:
    """
    Performs all necessary steps for determining outliers.

    Args:
        dataset_complete (pd.DataFrame): complete dataset.
        cont (float): contamination (estimated % of outliers).
        max_feat (int): features to train the isolation forest.
        max_samp (int): samples to train the isolations forest (impacts the tree size).
        n_est (int): numbers of trees in the ensemble.
        random_state (int): random state.
        write_to_csv (bool): boolean whether the results should be stored in a csv file.

    Returns:
        pd.DataFrame: DataFrame containing the absolute results.
        pd.DataFrame: DataFrame containing the relative results.
    """
    # Create dataframe for results
    alphabet = list(string.ascii_uppercase)
    params = ['Cont', 'Max_feat', 'Max_samp', 'n_est', 'random_state']
    cols = params + alphabet
    results_rel_dict = {"Cont": cont, "Max_feat": max_feat,
                        "Max_samp": max_samp, "n_est": n_est, "random_state": random_state}
    results_abs_dict = {"Cont": cont, "Max_feat": max_feat,
                        "Max_samp": max_samp, "n_est": n_est, "random_state": random_state}

    datapoints_total = 0
    outliers_total = 0

    results_fi = pd.DataFrame(columns=columns_df_res)
    # Loop though all letters
    for letter in alphabet:

        # Prepare dataset
        data_letter, letter_train, letter_test = prepare_dataset(
            letter, dataset_complete, test_prob=0.2)

        # Perform the isolation forest
        iso_for, y_train, y_test, y_train_pred, y_test_pred = isolation_forest(letter_train, letter_test, columns_train,
                                                                               cont, max_feat, max_samp, n_est, random_state)
        threshold_outlier = iso_for.threshold_

        # Categorize outliers
        res_letter = categorization(letter_test, y_test, threshold_outlier)
        results_fi = pd.concat([results_fi, res_letter], ignore_index=True)

        # Calculate statistics
        num_dp, num_out, perc = descriptive_stat_threshold(res_letter)
        datapoints_total += num_dp
        outliers_total += num_out

        # Save results to dictionary
        results_abs_dict[letter] = num_out
        results_rel_dict[letter] = perc

    # Save overall results
    results_fi = results_fi.astype(float)
    results_abs_dict["Total"] = datapoints_total
    results_rel_dict["Total"] = outliers_total/datapoints_total
    results_rel_dict["f1"] = f1_score(results_fi["outlier"], results_fi["projection"])
    results_rel_dict["accuracy"] = accuracy_score(results_fi["outlier"], results_fi["projection"])
    results_rel_dict["precision"] = precision_score(results_fi["outlier"], results_fi["projection"])
    results_rel_dict["recall"] = recall_score(results_fi["outlier"], results_fi["projection"])
    results_abs = pd.DataFrame([results_abs_dict])
    results_rel = pd.DataFrame([results_rel_dict])

    # Write results to csv
    results_fi = results_fi.drop("Anomaly_Score", axis=1)
    if write_to_csv == True:
        filename = Path("Data/Output_Data/02_Results_Ifor/TEST_Ifor_3std_4,16%_{}_{}_{}_{}_{}.csv".format(cont, 
                        max_feat, max_samp, n_est, random_state)).absolute()
        results_fi.to_csv(filename)
    return results_abs, results_rel


def cross_validation(params: dict, dataset_complete: pd.DataFrame,write_to_csv: bool = False) -> tuple[pd.DataFrame,
                                                                                                        pd.DataFrame]:
    """
    Performs cross validation.

    Args:
        params (dict): dicitionary containing all hyperparamters.
        dataset_complete (pd.DataFrame): complete dataset.
        write_to_csv (bool): boolean whether the results should be stored in a csv file.

    Returns:
        pd.DataFrame: absolute number of outliers.
        pd.DataFrame: relative number of outliers.
    """

    a = params.values()
    combinations = list(itertools.product(*a))
    alphabet = list(string.ascii_uppercase)
    params = ['Cont', 'Max_feat', 'Max_samp', 'n_est', 'random_state']
    cols = params + alphabet
    results_abs = pd.DataFrame(columns=cols)
    results_rel = pd.DataFrame(columns=cols)
    for c in combinations:
        abs, rel = algorithm(
            dataset_complete, cont=c[0], max_feat=c[1], max_samp=c[2], n_est=c[3], random_state=c[4], write_to_csv=write_to_csv)

        results_abs = pd.concat([results_abs, abs], ignore_index=True)
        results_rel = pd.concat([results_rel, rel], ignore_index=True)

    if write_to_csv == True:
        path_abs = Path("Data/Output_Data/02_Results_Ifor/TEST_Ifor_abs.csv").absolute()
        path_rel = Path("Data/Output_Data/02_Results_Ifor/TEST_Ifor_rel.csv").absolute()
        results_abs.to_csv(path_abs)
        results_rel.to_csv(path_rel)

    return results_abs, results_rel


def main():
    hyper_params = {
        'cont': [0.01, 0.05, 0.1],
        'max_feat': [4, 16],
        'max_samp': [10, 100],
        'max_est': [10, 100],
        'random_state': [0]}

    res_abs, res_rel = cross_validation(hyper_params, letter_df_complete, True)


main()
