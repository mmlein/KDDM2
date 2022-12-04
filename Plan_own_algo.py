import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math

# Pseudo Code Simulated Annealing
def Simulated_Annealing():
    x = Generate_Initial_Solution()
    score_x = Get_Score(x)
    while Termination_Criterion_Met == False:
        x_n = Find_Random_Neighbor(x)
        score_n_x = Get_Score(x_n)
        if score_n_x > score_x:
            x = x_n
            score_x = score_n_x
        else:
            p = random.uniform(0,1)
            
            #Temperature is a hyperparameter of Simulated Annealing
            if p <= math.exp((-abs(score_x - score_n_x)/Temperature)):
                x = x_n
                score_x = score_n_x    
                #the worse the neighboring solution or the higher the temperature, 
                #the lower the probability for accepting that solution
        Temperature = Update_Temperature(Temperature)
        
    #In the first phase, this algorithm allows for exploration, 
    #in the second for exploitation

def Load_Datafile() -> pd.DataFrame:
    """
    Reads data to dataframe and sets the name of the dataframe.
    
    (Args:
        filepath(str): Path of file.)
        
    Returns:
        pd.DataFrame: Named dataframe with data from .data.
    """
    header = list(range(1, 17))
    header = ["letter"] + header

    letter_df_complete = pd.read_csv("letter-recognition.data", 
    sep = ",", 
    names = header)
    return letter_df_complete

def Filter_Dataset(letter: str, dataset: pd.DataFrame, test_prob: float = 0.2, 
                   rand_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    """
    Creates dataframe with initial weights for features
    
    Args:
        dataset (pd.DataFrame): dataframe containing the features' names in the header
        
    Returns:
        pd.DataFrame: dataframe containing the initial weights for features
    """
    feature_names = list(dataset.columns)
    intial_weights = list(range(1, 16))
    feature_weights_list = list(zip(feature_names, intial_weights))
    feature_weights = pd.DataFrame(feature_weights_list, columns=['Feature','Weight'])
    return feature_weights

def Termination_Criterion_Met() -> bool:
    """
    Returns a boolean to stop the algorithm
    
    Args:
        xxx
        
    Returns:
        bool: boolean if the stopping criterion is met
    """
    stop_algorithm = False

    #Code stopping criterion

    return stop_algorithm

def Find_Random_Neighbor(dataset_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Finds a neighboring solution
    
    Args:
        dataset_weights (pd.DataFrame): Current weights for features

    Returns:
        pd.DataFrame: New weights for features
    """
    new_dataset_weights = dataset_weights.copy()
    
    #Code changes in new dataset

    return new_dataset_weights

def Get_Limits(feature: str, dataset: pd.DataFrame, IQR_Factor: float) -> tuple[float, float]:
    """
    Calculates limits for inlier or outlier
    
    Args:
        feature (str): feature of interest
        dataset (pd.DataFrame): dataset containing all datapoints to calculate statistics
        IQR_Factor (float): factor to set limits

    Returns:
        float: lower limit for inliers
        float: upper limit for inliers
    """
    
    #Calculate q1, q3, IQR for feature
    lower_limit = 0
    upper_limit = 0
    #calculate lower and upper limit

    return lower_limit, upper_limit

def Get_Feature_Factor(feature: str, dataset: pd.DataFrame, lower_limit: float, upper_limit: float) -> bin:
    """
    Determines if value for a given feature falls into specified range
    
    Args:
        feature (str): feature of interest
        dataset (pd.DataFrame): dataset containing all datapoints to calculate statistics
        lower_limit (float): lower limit of range
        upper_limit (float): upper limit of range

    Returns:
        bin: binary variable, whether data for feature falls into specified range
    """

def Get_Score(features_list: list, dataset: pd.DataFrame, lower_limit: float, upper_limit: float, 
              weights: pd.DataFrame) -> int:
    """
    Calculates the score for one datapoint
    
    Args:
        features_list (list): list of all features
        dataset (pd.DataFrame): dataset containing the values for all features for a given datapoint
        lower_limit (float): lower limit of range
        upper_limit (float): upper limit of range
        weights (pd.DataFrame): weights for each feature

    Returns:
        int: total score for a datapoint
    """
    score = 0
    for feature in features_list:
        factor = Get_Feature_Factor(feature, dataset, lower_limit, upper_limit)
        score += weights[feature] * factor

    return score

def Update_Temperature(Temperature: float, Alpha: float = 0.9) -> float:
    """
    Updates temperature
    
    Args:
        Temperature (float): current temperature
        Alpha (float): hyperparameter to decrease temperature
        
    Returns:
        float: updated temperature
    """
    Temperature *= Alpha
    return Temperature