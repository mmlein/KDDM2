import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random


# feature information
columns = {"letter": "capital letter (26 values from A to Z)", "x-box": "horizontal position of box(integer)",
           "y-box": "vertical position of box(integer)", "width": "width	width of box(integer)",
           "high": "height of box(integer)", "onpix": "onpix	total", "x-bar": "mean x of on pixels in box(integer)",
           "y-bar": "mean y of on pixels in box(integer)", "x2bar": "mean x variance(integer)", "y2bar": "mean y variance (integer)",
           "xybar": "mean x y correlation	(integer)", "x2ybr": "mean of x * x * y	(integer)", "xy2br": "mean of x * y * y (integer)",
           "x-ege": "mean edge count left to right (integer)", "xegvy": "correlation of x-ege with y (integer)",
           "y-ege": "mean edge count bottom to top	(integer)", "yegvx": "correlation of y-ege with x (integer)"}


path = Path("Data/Input_Data/letter-recognition.data").absolute()

# get the whole alphabet
alphabet = list(string.ascii_uppercase)

# read the data
df_letters = pd.read_csv(path, names=list(columns.keys()))

dataframes = dict()

sum = 0

# fill dict with letter as key and filtered dataframe per letter and get the number of all entries
for letter in alphabet:
    df = df_letters.loc[(df_letters.letter == letter), :]
    #df = df.drop('letter', axis=1, inplace=False)
    sum += len(df.index)
    print(letter, len(df.index))
    dataframes[letter] = df

columns = list(df_letters.columns)

decision = [True, False]
outliers = 0
dataframes_with_outliers = []
outlier_std = 3  # number of the standard diviations the data should be away
outliers_dict = {}
outlier_percentage = 0.05  # wanted percentage of outliers

# iterate over the part-dataframes
for letter, dataframe in dataframes.items():
    part_outliers = 0  # number of outliers of this part/letter
    original_dataframe = dataframe.copy()
    indexes = list(dataframe.index)  # get the indexes of this part
    len_dataframe = len(dataframe.index)  # get lenght of this dataframe part
    # amount, how many outliers per letter per column should be generated
    amount = int(len_dataframe/16 * outlier_percentage)
    st_div = pd.DataFrame(dataframe.std())  # generate std per column
    st_div.columns = ["Std"]
    st_div["factor"] = st_div["Std"] * outlier_std  # generate the factors
    # column to mark whether it is an outlier(1) or not (0)
    dataframe["outlier"] = 0

    for column in columns[1:]:
        part_outliers += amount  # add the amount of outliers for the part/letter
        outliers += amount  # add add the amount of outliers to overall amount
        # decide randomnly whether to add of substract
        addition = random.sample(decision, 1)
        # take random which samples to make to outliers
        indexes_to_use = random.sample(indexes, amount)
        if addition:  # if True, add the outlier factore, else substract
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] + st_div.loc[column, "factor"]
        else:
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] - st_div.loc[column, "factor"]
        # mark outliers as outliers
        dataframe.loc[indexes_to_use, "outlier"] = 1
        # delete used outliers
        indexes = [ele for ele in indexes if ele not in indexes_to_use]
    # add percentage of outliers of letter
    outliers_dict[letter] = part_outliers/len_dataframe
    dataframes_with_outliers.append(dataframe)

outlier_dataframe = pd.concat(
    dataframes_with_outliers)  # combine all dataframes

# add the total percentage of outliers
outliers_dict["sum"] = outliers/len(outlier_dataframe.index)
percentage_of_outlier = outliers/len(outlier_dataframe.index)*100

print(
    f"percentage of outliers general {percentage_of_outlier}%")

solution = pd.DataFrame.from_dict(outliers_dict, orient='index')

solution.to_csv(
    f"Data/Output_Data/solution_{outlier_std}std_{percentage_of_outlier}%.csv")  # save solution of percentages

outlier_dataframe.to_csv(
    f"Data/Output_Data/dataframe_with_outliers_{outlier_std}std_{percentage_of_outlier}%.csv")  # save dataframe with outliers
