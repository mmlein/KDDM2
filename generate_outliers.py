import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random

columns = {"letter": "capital letter (26 values from A to Z)", "x-box": "horizontal position of box(integer)",
           "y-box": "vertical position of box(integer)", "width": "width	width of box(integer)",
           "high": "height of box(integer)", "onpix": "onpix	total", "x-bar": "mean x of on pixels in box(integer)",
           "y-bar": "mean y of on pixels in box(integer)", "x2bar": "mean x variance(integer)", "y2bar": "mean y variance (integer)",
           "xybar": "mean x y correlation	(integer)", "x2ybr": "mean of x * x * y	(integer)", "xy2br": "mean of x * y * y (integer)",
           "x-ege": "mean edge count left to right (integer)", "xegvy": "correlation of x-ege with y (integer)",
           "y-ege": "mean edge count bottom to top	(integer)", "yegvx": "correlation of y-ege with x (integer)"}


def inspection(x_data):

    plt.figure()
    sns.displot(x_data, kde=True)
    plt.show()
    plt.figure()
    sns.boxplot(x=x_data)
    plt.show()
    print(np.mean(x_data))


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

outlier_std = 3

outliers_dict = {}

for letter, dataframe in dataframes.items():
    part_outliers = 0
    original_dataframe = dataframe.copy()
    indexes = list(dataframe.index)
    len_dataframe = len(dataframe.index)
    amount = int(len_dataframe/16*0.05)
    st_div = pd.DataFrame(dataframe.std())
    st_div.columns = ["Std"]
    st_div["factor"] = st_div["Std"] * outlier_std
    dataframe["outlier"] = 0

    for column in columns[1:]:
        part_outliers += amount
        outliers += amount
        addition = random.sample(decision, 1)
        indexes_to_use = random.sample(indexes, amount)
        print(dataframe.loc[indexes_to_use, column])
        print(addition)
        if addition:
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] + st_div.loc[column, "factor"]
        else:
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] - st_div.loc[column, "factor"]
        print(dataframe.loc[indexes_to_use, column])
        dataframe.loc[indexes_to_use, "outlier"] = 1
        indexes = [ele for ele in indexes if ele not in indexes_to_use]
    outliers_dict[letter] = part_outliers/len_dataframe
    dataframes_with_outliers.append(dataframe)

outlier_dataframe = pd.concat(dataframes_with_outliers)

outliers_dict["sum"] = outliers/len(outlier_dataframe.index)
percentage_of_outlier = outliers/len(outlier_dataframe.index)*100

print(
    f"percentage of outliers general {percentage_of_outlier}%")

print(outliers_dict)
print(outlier_dataframe)

solution = pd.DataFrame.from_dict(outliers_dict, orient='index')

solution.to_csv(f"Data/Output_Data/TEST_solution_{outlier_std}std_{percentage_of_outlier}%.csv")

outlier_dataframe.to_csv(
    f"Data/Output_Data/TEST_dataframe_with_outliers_{outlier_std}std_{percentage_of_outlier}%.csv")
