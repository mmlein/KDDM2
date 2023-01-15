import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math
import string
import time

def Load_Datafile() -> pd.DataFrame:
    #Status: DONE
    """
    Reads data to dataframe and sets the name of the dataframe.
    
    (Args:
        filepath(str): Path of file.)
        
    Returns:
        pd.DataFrame: Named dataframe with data from .data.
    """
    header = list(range(1, 17))
    header = ["letter"] + header

    letter_df_complete = pd.read_csv("Data\letter-recognition.data", 
    sep = ",", 
    names = header)
    return letter_df_complete

def Filter_Dataset(letter: str, dataset: pd.DataFrame, test_prob: float = 0.2, 
                   rand_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Status: DONE
    """
    Filters dataset for specific letter and splits into training and testing dataset
    
    Args:
        letter (str): letter of interest
        dataset (pd.DataFrame): dataframe containing the data
        test_prob (float): percentage of data reserved for testing
        rand_stat (int): random state for splitting the dataset
        
    Returns:
        pd.DataFrame: dataframe filtered on letter
        pd.DataFrame: dataframe for training
        pd.DataFrame: dataframe for testing
    """
    #Filter for the letter
    dataset_letter = dataset[dataset["letter"] == letter]

    #Create a copy of the dataset without the letter
    dataset_wo_let = dataset_letter.drop('letter', axis = 1)

    #Split the dataset into training and test data
    letter_train, letter_test = train_test_split(dataset_wo_let, test_size = test_prob, random_state = rand_state)

    return dataset_letter, letter_train, letter_test

def Generate_Initial_Solution(dataset: pd.DataFrame) -> pd.DataFrame:
    #Status: DONE
    """
    Creates dataframe with initial weights for features
    
    Args:
        dataset (pd.DataFrame): dataframe containing the features' names in the header
        
    Returns:
        pd.DataFrame: dataframe containing the initial weights for features
    """
    feature_names = list(dataset.columns)
    #print(feature_names)
    #initial_weights = list(range(1, 17))
    initial_weights = [10] * 16 
    feature_weights_list = list(zip(feature_names, initial_weights))
    feature_weights = pd.DataFrame(feature_weights_list, columns=['Feature','Weight'])
    return feature_weights

def Termination_Criterion_Met(temperature, Threshold_temp = 10) -> bool:
    #Status: DONE
    """
    Returns a boolean to stop the algorithm
    
    Args:
        xxx
        
    Returns:
        bool: boolean if the stopping criterion is met
    """
    stop_algorithm = False

    #Code stopping criterion (Temperature below a certain level, number of maximum steps reached)
    if temperature < Threshold_temp:
        stop_algorithm = True

    return stop_algorithm

def Find_Random_Neighbor(dataset_weights: pd.DataFrame) -> pd.DataFrame:
    #Status: DONE
    """
    Finds a neighboring solution
    
    Args:
        dataset_weights (pd.DataFrame): Current weights for features

    Returns:
        pd.DataFrame: New weights for features
    """
    new_dataset_weights = dataset_weights.copy()
    
    random_feature_first = random.randint(1,16)
    random_feature_second = random.randint(1,16)

    weight_feature_first = dataset_weights.loc[(dataset_weights.Feature == random_feature_first),"Weight"]
    weight_feature_second = dataset_weights.loc[(dataset_weights.Feature == random_feature_second),"Weight"]

    if int(weight_feature_first) > 0:
        weight_feature_first -= 1
        weight_feature_second += 1

    new_dataset_weights.loc[(dataset_weights.Feature == random_feature_first), "Weight"] = weight_feature_first
    new_dataset_weights.loc[(dataset_weights.Feature == random_feature_second), "Weight"] = weight_feature_second

    return new_dataset_weights

def Get_Limits(dataset: pd.DataFrame, IQR_Factor: float = 0.0) -> tuple[float, float]:
    #Status: DONE
    """
    Calculates limits for inlier or outlier
    
    Args:
        dataset (pd.DataFrame): dataset containing all datapoints to calculate statistics
        IQR_Factor (float): factor to set limits

    Returns:
        limits (pd.DataFrame): dataFrame containing limits
    """
    
    limits = pd.DataFrame(columns = ['Feature', 'low_limit', 'up_limit'])
    limits.loc[:,'Feature'] = dataset.columns

    for feat in dataset.columns:
        q1 = dataset[feat].quantile(.25)
        q3 = dataset[feat].quantile(.75)
        IQR = q3 - q1
        low_limit_val = q1 - IQR * IQR_Factor
        up_limit_val = q3 + IQR * IQR_Factor
        limits.loc[(limits.Feature == feat, "low_limit")] = low_limit_val
        limits.loc[(limits.Feature == feat, "up_limit")] = up_limit_val

    return limits

def Get_Feature_Factor(dataset: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    #Status: DONE
    """
    Determines if values for a given feature falls into specified range
    
    Args:
        feature (str): feature of interest
        dataset (pd.DataFrame): dataset containing all datapoints to calculate statistics
        limits (pd.DataFrame)

    Returns:
        bin_dataframe: dataframe with binary variables for each datapoint and each feature
    """
    bin_df = dataset.copy()

    for feat in bin_df.columns:
        lower_limit = float(limits.loc[(limits.Feature == feat), "low_limit"])
        upper_limit = float(limits.loc[(limits.Feature == feat), "up_limit"])

        bin_df.loc[:, feat] = [0 if float(x)>= lower_limit and float(x)<= upper_limit else 1 for x in dataset.loc[:, feat]]

    return bin_df

def Get_Scores(bin_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    #Status: DONE
    """
    Calculates the score for datapoints
    
    Args:
        bin_dataframe (pd.DataFrame): dataframe containing binary variables whether a datapoint's value for 
            a specific feature falls in the range
        weights (pd.DataFrame): weights for each feature

    Returns:
        pd.DataFrame: total score for datapoints
    """
    score_df = bin_df.copy()
    for feat in bin_df.columns:
        score_df[feat] = score_df[feat].multiply(float(weights.loc[(weights.Feature == feat), "Weight"]))
    
    score_df['Total_Score'] = score_df.sum(axis=1)

    return score_df

def Get_Maximum_Gap(score_df) -> float:
    """
    Calculates the maximum gap to find border between inliers and outliers
    
    Args:
        scores (pd.DataFrame): dataset containing the scores for all datapoints

    Returns:
        float: maximum gap
    """
    score_df = score_df.sort_values(by = ["Total_Score"])
    #print("Score df:", score_df)

    score_df["Gap"] = score_df["Total_Score"].diff()
    max_gap = score_df["Gap"].max()

    occ = score_df['Gap'].value_counts()[max_gap]
    #print("Occurence: ", occ)

    return max_gap, score_df

def Update_Temperature(temperature: float, Alpha: float = 0.9) -> float:
    #Status: DONE
    """
    Updates temperature
    
    Args:
        Temperature (float): current temperature
        Alpha (float): hyperparameter to decrease temperature
        
    Returns:
        float: updated temperature
    """
    temperature *= Alpha
    return temperature

def Simulated_Annealing(dataset: pd.DataFrame, letter_bin_df: pd.DataFrame,
    Initial_Temperature: float = 100, Alpha: float = 0.9, Counter: int = 1000):
    print("Start Simulated Annealing")
    weights = Generate_Initial_Solution(dataset)
    score_weights = Get_Scores(letter_bin_df, weights)
    gap_weights, score_weights = Get_Maximum_Gap(score_weights)
    temperature = Initial_Temperature
    stop_sim_ann = False

    while stop_sim_ann == False:
        for c in range(0, Counter):
            weights_n = Find_Random_Neighbor(weights)
            score_n_weights = Get_Scores(letter_bin_df, weights_n)
            gap_n_weights, score_n_weights = Get_Maximum_Gap(score_n_weights)
            if gap_n_weights > gap_weights:
                weights = weights_n
                gap_weights = gap_n_weights
            else:
                p = random.uniform(0,1)
                if p <= math.exp((-abs(gap_weights - gap_n_weights)/temperature)):
                    weights = weights_n
                    gap_weights = gap_n_weights 

        temperature = Update_Temperature(temperature, Alpha)
        stop_sim_ann = Termination_Criterion_Met(temperature)

    
    return weights, gap_weights

def Get_In_Outliers_Train(score_df: pd.DataFrame):
    """
    Determines in- and outliers
    
    Args:
        scores (pd.DataFrame): dataset containing the scores for all datapoints
        max gap (float): maximum gap

    Returns:
        pd.DataFrame: DataFrame containing information regarding in- and outliers
        float: percentage of outliers
    """
    score_df = score_df.sort_values(by = ["Total_Score"])
    score_df["Gap"] = score_df["Total_Score"].diff()
    idx_max_gap = score_df["Gap"].idxmax()
    score_at_max_gap = score_df.loc[idx_max_gap, "Total_Score"]
    datapoints_lower_score = len(score_df[score_df["Total_Score"] < score_at_max_gap])
    datapoints_upper_score = len(score_df[score_df["Total_Score"] >= score_at_max_gap])
    #datapoints_lower_score = score_df.loc[(score_df.Total_Score < score_at_max_gap), :].count()
    #datapoints_upper_score = score_df.loc[(score_df.Total_Score >= score_at_max_gap), :].count()

    score_df["Class"] = 0

    if datapoints_lower_score > datapoints_upper_score:
        score_df.loc[(score_df.Total_Score >= score_at_max_gap), "Class"] = 1
        inlier_up = True
    else:
        score_df.loc[(score_df.Total_Score < score_at_max_gap), "Class"] = 1
        inlier_up = False

    perc_outliers = score_df["Class"].sum()/len(score_df)

    return score_df, perc_outliers, score_at_max_gap, inlier_up

def Get_In_Outliers_Test(score_df: pd.DataFrame, threshold_score: float, inlier_up: bool):
    
    score_df["Class"] = 0

    if inlier_up == True:
        score_df.loc[(score_df.Total_Score >= threshold_score), "Class"] = 1
    else:
        score_df.loc[(score_df.Total_Score < threshold_score), "Class"] = 1

    perc_outliers = score_df["Class"].sum()/len(score_df)

    return score_df, perc_outliers

def main():
    startTime = time.time()
    df_complete = Load_Datafile()
    alphabet = list(string.ascii_uppercase)
    results = pd.DataFrame(columns = ['letter', 'outlier'])
    results['letter'] = alphabet

    for letter in alphabet:
        print("Letter: ", letter)
        #Train weights with training dataset
        dataset_letter, letter_train, letter_test = Filter_Dataset(letter, df_complete)
        limits_letter = Get_Limits(letter_train, 0)
        letter_bin_df = Get_Feature_Factor(letter_train, limits_letter)
        best_weights, best_gap = Simulated_Annealing(letter_train, letter_bin_df, 100, 0.8, 10)
        print(best_weights, best_gap)

        train_scores = Get_Scores(letter_bin_df, best_weights)
        train_scores, perc_train, score_gap, inlier = Get_In_Outliers_Train(train_scores)

        #Get number of outliers with testing dataset
        limits_letter_test = Get_Limits(letter_test, 0)
        letter_bin_df_test = Get_Feature_Factor(letter_test, limits_letter_test)
        test_scores = Get_Scores(letter_bin_df_test, best_weights)
        alloc_test, perc_test = Get_In_Outliers_Test(test_scores, score_gap, inlier)

        #print(perc_test)
        results.loc[(results.letter == letter), 'outlier'] = perc_test
        print("----------------------------------------------------------------\n")

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print(results)

main()

#IQR
#Start temperature
#Alpha
#Counter