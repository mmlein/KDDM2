import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math
import string
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

'''
rows = ,letter,x-box,y-box,width,high,onpix,x-bar,y-bar,x2bar,y2bar,xybar,x2ybr,xy2br,x-ege,xegvy,y-ege,yegvx,outlier
'''


def Load_Datafile() -> pd.DataFrame:
    """
    Reads data to dataframe.

    Args:
        any

    Returns:
        pd.DataFrame: Named dataframe with data from .data.
    """
    header = list(range(1, 17))
    header = ["idx", "letter"] + header + ["outlier"]

    letter_df_complete = pd.read_csv("Data/Input_Data/dataframe_with_outliers_3std_4.16%.csv",
                                     sep=",")

    letter_df_complete.columns = header
    letter_df_complete = letter_df_complete.drop('idx', axis=1)
    return letter_df_complete


def Filter_Dataset(letter: str, dataset: pd.DataFrame, test_prob: float = 0.2,
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


def Generate_Initial_Solution(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Creates dataframe with initial weights for features.

    Args:
        dataset (pd.DataFrame): dataframe containing the features' names in the header.

    Returns:
        pd.DataFrame: dataframe containing the initial (equal) weights for features.
    """
    # Get a list of the features' names
    feature_names = list(dataset.columns)

    # At the beginning, all weights are set equally to 10
    initial_weights = [10] * 16
    feature_weights_list = list(zip(feature_names, initial_weights))

    # Create a dataframe containing features and their resepective weights
    feature_weights = pd.DataFrame(
        feature_weights_list, columns=['Feature', 'Weight'])
    return feature_weights


def Termination_Criterion_Met(temperature: int, Threshold_temp: int = 10) -> bool:
    """
    Returns a boolean to stop the algorithm.

    Args:
        temperature (int): current temperature.
        Threshold_temp (int): threshold temperature.

    Returns:
        bool: boolean if the stopping criterion is met.
    """
    stop_algorithm = False

    # If the temperature falls below the predefined threshold, the algorithm should be stopped
    if temperature < Threshold_temp:
        stop_algorithm = True

    return stop_algorithm


def Find_Random_Neighbor(dataset_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Finds a neighboring solution.

    Args:
        dataset_weights (pd.DataFrame): Current weights for features.

    Returns:
        pd.DataFrame: New weights for features.
    """
    new_dataset_weights = dataset_weights.copy()

    # Pick two random features
    random_feature_first = random.randint(1, 16)
    random_feature_second = random.randint(1, 16)

    # Get the randomly chosen features' weights
    weight_feature_first = dataset_weights.loc[(
        dataset_weights.Feature == random_feature_first), "Weight"]
    weight_feature_second = dataset_weights.loc[(
        dataset_weights.Feature == random_feature_second), "Weight"]

    # If the first feature's weight exceeds 0 (thus is larger or equal to 1), deduct 1 from the first feature's weight
    # and add it to the second feature's weight
    if int(weight_feature_first) > 0:
        weight_feature_first -= 1
        weight_feature_second += 1

    new_dataset_weights.loc[(dataset_weights.Feature ==
                             random_feature_first), "Weight"] = weight_feature_first
    new_dataset_weights.loc[(dataset_weights.Feature ==
                             random_feature_second), "Weight"] = weight_feature_second

    return new_dataset_weights


def Get_Limits(dataset: pd.DataFrame, IQR_Factor: float = 0.0) -> pd.DataFrame:
    """
    Calculates limits for inlier or outlier.

    Args:
        dataset (pd.DataFrame): dataset containing all datapoints to calculate statistics.
        IQR_Factor (float): factor to set limits.

    Returns:
        pd.DataFrame: dataFrame containing limits.
    """

    limits = pd.DataFrame(columns=['Feature', 'low_limit', 'up_limit'])
    columns = list(dataset.columns)[0:16]
    limits.loc[:, 'Feature'] = columns
    dataset = dataset.astype(float)

    for feat in columns:
        q1 = dataset[feat].quantile(.25)
        q3 = dataset[feat].quantile(.75)
        IQR = q3 - q1
        low_limit_val = q1 - IQR * IQR_Factor
        up_limit_val = q3 + IQR * IQR_Factor
        limits.loc[(limits.Feature == feat, "low_limit")] = low_limit_val
        limits.loc[(limits.Feature == feat, "up_limit")] = up_limit_val

    return limits


def Get_Feature_Factor(dataset: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    """
    Determines if values for a given feature falls into specified range.

    Args:
        dataset (pd.DataFrame): dataset containing all datapoints to determine binary variable.
        limits (pd.DataFrame): dataframe containing the limits for all features.

    Returns:
        pd.DataFrame: dataframe with binary variables for each datapoint and each feature.
    """
    bin_df = dataset.copy()
    columns = list(bin_df.columns)[0:16]

    for feat in columns:
        # Retrieve the limits
        lower_limit = float(limits.loc[(limits.Feature == feat), "low_limit"])
        upper_limit = float(limits.loc[(limits.Feature == feat), "up_limit"])

        # For each row in the entire column, enter a 0 if the value is within the limits, and a 1 otherwise.
        bin_df.loc[:, feat] = [0 if float(x) >= lower_limit and float(
            x) <= upper_limit else 1 for x in dataset.loc[:, feat]]

    return bin_df


def Get_Scores(bin_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the score for datapoints.

    Args:
        bin_dataframe (pd.DataFrame): dataframe containing binary variables whether a datapoint's value for
            a specific feature falls in the range.
        weights (pd.DataFrame): weights for each feature.

    Returns:
        pd.DataFrame: total score for datapoints.
    """
    columns = list(bin_df.columns)[0:16]
    score_df = bin_df.copy()
    for feat in columns:
        score_df[feat] = score_df[feat].multiply(
            float(weights.loc[(weights.Feature == feat), "Weight"]))

    score_df['Total_Score'] = score_df.sum(axis=1)

    return score_df


def Get_Maximum_Gap(score_df) -> tuple[float, pd.DataFrame]:
    """
    Calculates the maximum gap to find border between inliers and outliers.

    Args:
        scores (pd.DataFrame): dataset containing the scores for all datapoints.

    Returns:
        float: maximum gap.
        pd.DataFrame: dataframe with differnces (gaps)
    """
    # Sort scores ascending
    score_df = score_df.sort_values(by=["Total_Score"])

    # Get the difference to the previous row and find the maximum differnce (maximum gap)
    score_df["Gap"] = score_df["Total_Score"].diff()
    max_gap = score_df["Gap"].max()

    occ = score_df['Gap'].value_counts()[max_gap]

    return max_gap, score_df


def Update_Temperature(temperature: float, Alpha: float = 0.9) -> float:
    """
    Updates temperature.

    Args:
        Temperature (float): current temperature.
        Alpha (float): hyperparameter to decrease temperature.

    Returns:
        float: updated temperature.
    """
    temperature *= Alpha
    return temperature


def Simulated_Annealing(dataset: pd.DataFrame, letter_bin_df: pd.DataFrame,
                        Initial_Temperature: float = 100, Alpha: float = 0.9, Counter: int = 1000) -> tuple[pd.DataFrame, float]:
    """
    Performes Simulated Annealing

    Args:
        dataset (pd.DataFrame): dataset containing all datapoints.
        letter_bin_df (pd.DataFrame): dataframe containing binary variables whether the value for a feature is
            considered in- or outlier.
        Initial_Temperature (float): starting temperature.
        Alpha (float): hyperparameter to decrease temperature.
        Counter (int): hyperparameter to decrease temperature.

    Returns:
        pd.DataFrame: final set of weights
        float: maximum gap.
    """

    print("Start Simulated Annealing")

    # Generate an initial solution
    weights = Generate_Initial_Solution(dataset)
    score_weights = Get_Scores(letter_bin_df, weights)
    gap_weights, score_weights = Get_Maximum_Gap(score_weights)

    # Initialize temperature and boolean
    temperature = Initial_Temperature
    stop_sim_ann = False

    # While the stopping criterion is not met
    while stop_sim_ann == False:

        for c in range(0, Counter):

            # Find a neighboring solution and evaluate its performance
            weights_n = Find_Random_Neighbor(weights)
            score_n_weights = Get_Scores(letter_bin_df, weights_n)
            gap_n_weights, score_n_weights = Get_Maximum_Gap(score_n_weights)

            # If the neighboring solution outperforms the previous one, accept it
            if gap_n_weights > gap_weights:
                weights = weights_n
                gap_weights = gap_n_weights

            # Else accept it with a certain probability
            else:
                p = random.uniform(0, 1)
                if p <= math.exp((-abs(gap_weights - gap_n_weights)/temperature)):
                    weights = weights_n
                    gap_weights = gap_n_weights

        # After Counter times, update the temperature and check stopping criterion
        temperature = Update_Temperature(temperature, Alpha)
        stop_sim_ann = Termination_Criterion_Met(temperature)

    return weights, gap_weights


def Get_In_Outliers_Train(score_df: pd.DataFrame) -> tuple[pd.DataFrame, float, int, bool]:
    """
    Determines in- and outliers of training dataset.

    Args:
        scores (pd.DataFrame): dataset containing the scores for all datapoints.

    Returns:
        pd.DataFrame: DataFrame containing information regarding in- and outliers.
        float: percentage of outliers.
        int: threshold score for in- and outliers.
        bool: boolean whether inliers are below a certain score or above.
    """

    # Sort scores ascending and determine threshold score
    score_df = score_df.sort_values(by=["Total_Score"])
    score_df["Gap"] = score_df["Total_Score"].diff()
    idx_max_gap = score_df["Gap"].idxmax()
    score_at_max_gap = score_df.loc[idx_max_gap, "Total_Score"]

    # Get the number of datapoints below and above the threshold score
    datapoints_lower_score = len(
        score_df[score_df["Total_Score"] < score_at_max_gap])
    datapoints_upper_score = len(
        score_df[score_df["Total_Score"] >= score_at_max_gap])

    # By default set the class to 0 (inlier)
    score_df["Class"] = 0

    # If the majority of datapoints scores below the threshold score, set the class of the datapoints scoring above the
    # threshold to 1 and the boolean to true (necessary for testing dataset)
    if datapoints_lower_score > datapoints_upper_score:
        score_df.loc[(score_df.Total_Score >= score_at_max_gap), "Class"] = 1
        inlier_up = True

    # Else set the class of the datapoints scoring below the threshold to 1 and the boolean to false
    else:
        score_df.loc[(score_df.Total_Score < score_at_max_gap), "Class"] = 1
        inlier_up = False

    # Determine the percentage of outliers
    perc_outliers = score_df["Class"].sum()/len(score_df)

    return score_df, perc_outliers, score_at_max_gap, inlier_up


def Get_In_Outliers_Test(score_df: pd.DataFrame, threshold_score: float, inlier_up: bool) -> tuple[pd.DataFrame, int,
                                                                                                   float]:
    """
    Determines in- and outliers of testing dataset.

    Args:
        scores (pd.DataFrame): dataset containing the scores for all datapoints.
        threshold_score (float): threshold score to split outliers and normal data.
        inlier_up (bool): boolean whether inlier are above or below the threshold.

    Returns:
        pd.DataFrame: DataFrame containing information regarding in- and outliers.
        int: number of outliers.
        float: percentage of outliers.
    """

    score_df["Class"] = 0

    if inlier_up == True:
        score_df.loc[(score_df.Total_Score >= threshold_score), "Class"] = 1
    else:
        score_df.loc[(score_df.Total_Score < threshold_score), "Class"] = 1

    outliers_amount = score_df["Class"].sum()
    perc_outliers = outliers_amount/len(score_df)

    return score_df, outliers_amount, perc_outliers


def Algorithm(combinations: list) -> list:
    """
    Performes Outlier detection.

    Args:
        combinations (list): list containing the hyperparameters.

    Returns:
        list: list containing the results.
    """

    # Start the timer
    startTime = time.time()

    # Load the datafile
    df_complete = Load_Datafile()
    df_without_outliers = df_complete.copy().iloc[:, 0:17]
    rows_complete = len(df_without_outliers.index)

    # Create dataframe for results
    alphabet = list(string.ascii_uppercase)
    columns = ["IQR", "t", "a", "Count"]
    columns = columns + alphabet
    columns.append("sum")
    results = pd.DataFrame(columns=columns)

    # Loop through all letters
    for comb in combinations:
        IQR_F = comb[0]
        Init_T = comb[1]
        A = comb[2]
        Count = comb[3]
        row = {"IQR": IQR_F, "t": Init_T, "a": A, "Count": Count}
        print(row)
        outliers = 0
        result_list = []

        for letter in alphabet:
            print("Letter: ", letter)

            # Train weights with training dataset
            dataset_letter, letter_train, letter_test = Filter_Dataset(
                letter, df_complete)
            limits_letter = Get_Limits(letter_train, IQR_F)
            letter_bin_df = Get_Feature_Factor(
                letter_train.iloc[:, 0:16], limits_letter)
            best_weights, best_gap = Simulated_Annealing(letter_train, letter_bin_df,
                                                         Initial_Temperature=Init_T, Alpha=A, Counter=Count)
            print(best_weights, best_gap)

            # Get the training scores and threshold
            train_scores = Get_Scores(letter_bin_df, best_weights)
            train_scores, perc_train, score_gap, inlier = Get_In_Outliers_Train(
                train_scores)

            # Get number of outliers with testing dataset
            # limits_letter_test = Get_Limits(letter_test)
            letter_bin_df_test = Get_Feature_Factor(letter_test, limits_letter)
            test_scores = Get_Scores(letter_bin_df_test, best_weights)
            alloc_test, amount_outliers, perc_test = Get_In_Outliers_Test(
                test_scores, score_gap, inlier)
            indexes_alloc_test = list(alloc_test.index.values)
            result_val = df_complete.loc[indexes_alloc_test, :]
            result_val["projection"] = alloc_test["Class"]
            result_list.append(result_val)

            # Save results to dataframe
            row[letter] = perc_test
            outliers += amount_outliers
            print(row)
            # results.loc[(results.letter == letter), new_column] = perc_test
            print("----------------------------------------------------------------\n")
        result_df_val = pd.concat(result_list)
        row["sum"] = outliers/rows_complete

        filename = f"Data/Output_Data/05_Own_algorithm/TEST_results_val_own_algo_{IQR_F}_{Init_T}_{A}_{Count}.csv"
        result_df_val.to_csv(filename)

        row["f1"] = f1_score(result_df_val["outlier"],
                             result_df_val["projection"])
        row["accuracy"] = accuracy_score(
            result_df_val["outlier"], result_df_val["projection"])
        row["precision"] = precision_score(
            result_df_val["outlier"], result_df_val["projection"])
        row["recall"] = recall_score(
            result_df_val["outlier"], result_df_val["projection"])
        results = results.append(row, ignore_index=True)
        print(results)

    # Stop the timer
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    exsheet_name = str(IQR_F) + "_" + str(Init_T) + \
        "_" + str(A) + "_" + str(Count)
    print(results)
    return results
    # with pd.ExcelWriter('Data\Results.xlsx', mode='a') as writer:
    #    results.to_excel(writer, sheet_name=exsheet_name)


def main():

    IQR_Vec = [0, 0.5, 1, 1.5]
    Init_T_Vec_1 = [1000]
    A_Vec_1 = [0.8, 0.9]
    Count_Vec_1 = [10, 100]

    Init_T_Vec_2 = [100, 500]
    A_Vec_2 = [0.95]
    Count_Vec_2 = [10]

    combinations = []

    for iqr in IQR_Vec:
        for t in Init_T_Vec_1:
            for a in A_Vec_1:
                for count in Count_Vec_1:
                    combinations.append([iqr, t, a, count])
    for iqr in IQR_Vec:
        for t in Init_T_Vec_2:
            for a in A_Vec_2:
                for count in Count_Vec_2:
                    combinations.append([iqr, t, a, count])

    results = Algorithm(combinations)

    results.to_csv(
        "Data/Output_Data/05_Own_algorithm/results_per_own_algo_hyper_para_3std_4_16%.csv")


main()
