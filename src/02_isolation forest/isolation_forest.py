
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import pyod
from pathlib import Path
from pyod.models.iforest import IForest
import string
from sklearn.model_selection import train_test_split

# ### Data Preparation


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
        dataset_wo_let, test_size=test_prob, random_state=rand_state)

    return dataset_letter, letter_train, letter_test


# ### Statistics
def descriptive_stat_threshold(dataset: pd.DataFrame, pred_score: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Calculates the statistics (https://towardsdatascience.com/use-the-isolated-forest-with-pyod-3818eea68f08).

    Args:
        dataset (pd.DataFrame): (training/testing) dataset.
        pred_score (np.ndarray): predictive scores from the isolation forest.
        threshold (float): threshold for classifying a datapoint as an outlier.

    Returns:
        pd.DataFrame: dataframe containinf the statistics.
    """

    dataset['Anomaly_Score'] = pred_score
    dataset['Group'] = np.where(
        dataset['Anomaly_Score'] < threshold, 'Normal', 'Outlier')

    # Calculate statistics:
    cnt = dataset.groupby('Group')['Anomaly_Score'].count(
    ).reset_index().rename(columns={'Anomaly_Score': 'Count'})
    cnt['Count %'] = (cnt['Count'] / cnt['Count'].sum()) * 100
    stat = dataset.groupby('Group').mean().round(2).reset_index()
    stat = cnt.merge(stat, left_on='Group', right_on='Group')

    return (stat)


# ### Models

def isolation_forest(train_data: pd.DataFrame, test_data: pd.DataFrame, cont: float = 0.05, max_feat: int = 1.0,
                     max_samp: int = 40, n_est: int = 100, random_state: int = 0) -> tuple[pyod.models.iforest, np.ndarray, np.ndarray]:
    """
    Creates a new isolation forest.

    Args:
        train_data (pd.DataFrame): dataframe for training.
        test_data (pd.DataFrame): dataframe for testing.
        cont (float): contamination (estimated % of outliers).
        max_feat (int): features to train the isolation forest.
        max_samp (int): samples to train the isolations forest (impacts the tree size).
        n_est (int): numbers of trees in the ensemble.
        rand_stat (int): random state.

    Returns:
        pyod.models.iforest: isolation forest.
        numpy.ndarray: numpy array containing training scores.
        numpy.ndarray: numpy array containing testing scores.
    """
    # Create a new iForest
    isft = IForest(behaviour='new', contamination=cont,
                   max_features=max_feat, max_samples=max_samp, n_estimators=n_est)

    # Fit iForest
    isft.fit(train_data)

    # Training data
    y_train_scores = isft.decision_function(train_data)
    y_train_pred = isft.predict(train_data)

    # Test data
    y_test_scores = isft.decision_function(test_data)
    y_test_pred = isft.predict(test_data)

    return isft, y_train_scores, y_test_scores


# ### Algorithm
def algorithm(dataset_complete: pd.DataFrame, cont: float = 0.05, max_feat: int = 1.0,
              max_samp: int = 40, n_est: int = 100, random_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs all necessary steps for determining outliers.

    Args:
        dataset_complete (pd.DataFrame): complete dataset.
        cont (float): contamination (estimated % of outliers).
        max_feat (int): features to train the isolation forest.
        max_samp (int): samples to train the isolations forest (impacts the tree size).
        n_est (int): numbers of trees in the ensemble.
        rand_state (int): random state.

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

    # Loop though all letters
    for letter in alphabet:

        # Perform the isolation forest
        data_letter, letter_train, letter_test = prepare_dataset(
            letter, dataset_complete, test_prob=0.2)
        iso_for, y_train, y_test = isolation_forest(letter_train, letter_test, cont, max_feat,
                                                    max_samp, n_est, random_state)
        threshold_outlier = iso_for.threshold_
        stats_df = descriptive_stat_threshold(
            letter_test, y_test, threshold_outlier)
        absolute = stats_df[stats_df['Group'] == 'Outlier']['Count']
        percentage = stats_df[stats_df['Group'] == 'Outlier']['Count %']
        total_data = stats_df['Count'].sum()

        results_abs_dict[letter] = absolute
        results_rel_dict[letter] = percentage

    results_abs = pd.DataFrame.from_dict(results_abs_dict)
    results_rel = pd.DataFrame.from_dict(results_rel_dict)
    return results_abs, results_rel


# ### Incorporating Cross Validation

def cross_validation(params: dict, dataset_complete: pd.DataFrame,
                     #write_to_excel: bool = True,
                     #path: str = "C:/Users/Florentina/Documents/Uni CSS/3. Semester/kddm2/Data/Results_Ifor.xlsx"
                     ) -> tuple[pd.DataFrame,
                                pd.DataFrame]:
    """
    Performs cross validation.

    Args:
        params (dict): dicitionary containing all hyperparamters.
        dataset_complete (pd.DataFrame): complete dataset.
        write_to_excel (bool): boolean variable if results should be written to excel.
        path (str): path of excel file.

    Returns:
        Any.
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
            dataset_complete, cont=c[0], max_feat=c[1], max_samp=c[2], n_est=c[3], random_state=c[4])

        results_abs = pd.concat([results_abs, abs], ignore_index=True)
        results_rel = pd.concat([results_rel, rel], ignore_index=True)
    '''
    if write_to_excel == True:
        with pd.ExcelWriter(path, mode = 'a') as writer:
            results_abs.to_excel(writer, sheet_name="absolute_results")
            results_rel.to_excel(writer, sheet_name="relative_results")
    '''
    return results_abs, results_rel
