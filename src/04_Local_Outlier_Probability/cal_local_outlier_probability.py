import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import string
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# column names to get a data overview
columns = {"letter": "capital letter	(26 values from A to Z)", "x-box": "horizontal position of box(integer)",
           "y-box": "vertical position of box(integer)", "width": "width	width of box(integer)",
           "high": "height of box(integer)", "onpix": "onpix	total", "x-bar": "mean x of on pixels in box(integer)",
           "y-bar": "mean y of on pixels in box(integer)", "x2bar": "mean x variance(integer)", "y2bar": "mean y variance	(integer)",
           "xybar": "mean x y correlation	(integer)", "x2ybr": "mean of x * x * y	(integer)", "xy2br": "mean of x * y * y (integer)",
           "x-ege": "mean edge count left to right (integer)", "xegvy": "correlation of x-ege with y	(integer)",
           "y-ege": "ean edge count bottom to top	(integer)", "yegvx": "correlation of y-ege with x	(integer)", "outlier": "class	(integer)"}

path = Path("Data/Input_Data/letter-recognition.data")
outlier_path = Path("Data/Input_Data/dataframe_with_outliers_3std_4.16%.csv").absolute()

# get the whole alphabet
alphabet = list(string.ascii_uppercase)

# read the data
df_letters = pd.read_csv(outlier_path, names=list(columns.keys()))

dataframes = dict()
sum = 0

# fill dict with letter as key and filtered dataframe per letter and get the number of all entries
for letter in alphabet:
    df = df_letters.loc[df_letters.letter == letter, :]
    df = df.drop('letter', axis=1, inplace=False)
    sum += len(df.index)
    print(letter, len(df.index))
    dataframes[letter] = df

print(sum)

# get a quality check for the amount of entries
amount = len(df_letters.index)
print(amount)

# get the means for each letter per column
means = df_letters.groupby("letter").mean()
print(df_letters)


def hyper_parameter_testing(dataframes, extents, n_neighbors, critical_values):
    columns = ["e", "n", "%"]
    columns = columns + alphabet
    columns.append("sum")
    print(columns)
    optimization_results = pd.DataFrame(columns=columns)

    results = {}

    for extent in extents:  # hyper parameter 1
        for n in n_neighbors:  # hyper parameter 2
            for value in critical_values:  # hyper parameter 3
                name = f"e={extent},n={n},%={value}"
                part_result_df = []
                # total number of outliers
                outliers = 0
                # the row which should be added
                row = {"e": extent, "n": n, "%": value}

                print(
                    f"Results for extent = {extent}, n_neighbors= {n} and critical value = {value}")
                for letter, dataframe in dataframes.items():
                    if letter == "outlier":
                        continue
                    # print(dataframe.isnull().sum())
                    dataframe = dataframe.astype(float)
                    dataframe_for_model = dataframe.iloc[:, 0:15]
                    m = loop.LocalOutlierProbability(
                        dataframe_for_model, n_neighbors=n, extent=extent).fit()
                    # get the probabilities per feature
                    scores = list(m.local_outlier_probabilities)
                    part_outlier_number = 0
                    part_outlier_list = []
                    for score in scores:
                        if score >= value:  # consider it as an outlier, when score > critical value
                            part_outlier_number += 1
                            part_outlier_list.append(1)
                        else:
                            part_outlier_list.append(0)
                    dataframe["projection"] = part_outlier_list
                    # add the outlier of a part-dataframe to the total outlier number
                    outliers += part_outlier_number
                    # calculate the percentage of outliers for the letter and add it to the row
                    percentage = part_outlier_number/len(dataframe.index)
                    row[letter] = percentage
                    print(f"{letter}: {percentage}")
                    part_result_df.append(dataframe)

                # add the percentage of total outliers to the row
                results[name] = pd.concat(part_result_df)
                row["sum"] = outliers/amount
                row["f1"] = f1_score(
                    results[name]["outlier"], results[name]["projection"])
                row["accuracy"] = accuracy_score(
                    results[name]["outlier"], results[name]["projection"])
                row["precision"] = precision_score(
                    results[name]["outlier"], results[name]["projection"])
                row["recall"] = recall_score(
                    results[name]["outlier"], results[name]["projection"])
                print(f"Overall Result: {outliers/amount}")

                optimization_results = optimization_results.append(
                    row, ignore_index=True)  # insert the row to the dataframe
    return results, optimization_results


# parameters to diviate
extents = [2]  # number of standarddivations, the value has to diviate
n_neighbors = [10]  # number of neighbors to consider
# citical Local outlier probability - min value to consider as a outlier
critical_values = [0.6]  # np.arange(0.60, 0.71, 0.05)
''''
extent = 2
n = 10
value = 0.8
'''

# Results
results_val_dict, results_percentage = hyper_parameter_testing(
    dataframes=dataframes, extents=extents, n_neighbors=n_neighbors, critical_values=critical_values)

for name, data in results_val_dict.items():
    data.to_csv(
        f"Data/Output_Data/04_Local_outlier_prob/results_lop_val_3std_4,16%_{name}.csv")


# save the result in a csv file to continue working later
results_percentage.to_csv(
    "Data/Output_Data/04_Local_outlier_prob/results_lop_per_3std_4,16%_06_07.csv")
