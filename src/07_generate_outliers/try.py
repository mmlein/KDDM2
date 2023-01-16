import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import string
import random

columns = {"letter": "capital letter (26 values from A to Z)", "x-box": "horizontal position of box(integer)",
           "y-box": "vertical position of box(integer)", "width": "width	width of box(integer)",
           "high": "height of box(integer)", "onpix": "onpix	total", "x-bar": "mean x of on pixels in box(integer)",
           "y-bar": "mean y of on pixels in box(integer)", "x2bar": "mean x variance(integer)", "y2bar": "mean y variance (integer)",
           "xybar": "mean x y correlation	(integer)", "x2ybr": "mean of x * x * y	(integer)", "xy2br": "mean of x * y * y (integer)",
           "x-ege": "mean edge count left to right (integer)", "xegvy": "correlation of x-ege with y (integer)",
           "y-ege": "mean edge count bottom to top	(integer)", "yegvx": "correlation of y-ege with x (integer)"}


path = Path(
    "C:/Users/dt/Documents/CodingProjects/kddm2/Data/letter-recognition.data")

# get the whole alphabet
alphabet = list(string.ascii_uppercase)

# read the data
df_letters = pd.read_csv(path, names=list(columns.keys()))

dataframes = dict()

sum = 0

# fill dict with letter as key and filtered dataframe per letter and get the number of all entries
for letter in alphabet:
    df = df_letters.loc[(df_letters.letter == letter), :]
    df = df.drop('letter', axis=1, inplace=False)
    sum += len(df.index)
    print(letter, len(df.index))
    dataframes[letter] = df

columns = list(df_letters.columns)

decision = [True, False]

outliers = 0

dataframes_with_outliers = []

for letter, dataframe in dataframes.items():
    original_dataframe = dataframe.copy()
    indexes = list(dataframe.index)
    amount = int(len(dataframe.index)/16)
    st_div = pd.DataFrame(dataframe.std())
    st_div.columns = ["Std"]
    st_div["std*0.5"] = st_div["Std"] * 0.5

    for column in columns[1:]:
        outliers += amount
        addition = random.sample(decision, 1)
        indexes_to_use = random.sample(indexes, amount)
        print(dataframe.loc[indexes_to_use, column])
        print(addition)
        if addition:
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] + st_div.loc[column, "std*0.5"]
        else:
            dataframe.loc[indexes_to_use, column] = dataframe.loc[indexes_to_use,
                                                                  column] - st_div.loc[column, "std*0.5"]
        print(dataframe.loc[indexes_to_use, column])
        indexes = [ele for ele in indexes if ele not in indexes_to_use]
    dataframes_with_outliers.append(pd.concat([dataframe, original_dataframe]))

outlier_dataframe = pd.concat(dataframes_with_outliers)

print(
    f"percentage of outliers general {outliers/len(outlier_dataframe.index)*100}%")

print(outlier_dataframe)
