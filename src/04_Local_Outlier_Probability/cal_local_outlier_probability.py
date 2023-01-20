import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import string


# Reading the data and get an overview


# column names to get a data overview
columns = {"letter": "capital letter	(26 values from A to Z)", "x-box": "horizontal position of box(integer)",
           "y-box": "vertical position of box(integer)", "width": "width	width of box(integer)",
           "high": "height of box(integer)", "onpix": "onpix	total", "x-bar": "mean x of on pixels in box(integer)",
           "y-bar": "mean y of on pixels in box(integer)", "x2bar": "mean x variance(integer)", "y2bar": "mean y variance	(integer)",
           "xybar": "mean x y correlation	(integer)", "x2ybr": "mean of x * x * y	(integer)", "xy2br": "mean of x * y * y (integer)",
           "x-ege": "mean edge count left to right (integer)", "xegvy": "correlation of x-ege with y	(integer)",
           "y-ege": "ean edge count bottom to top	(integer)", "yegvx": "correlation of y-ege with x	(integer)"}

path = Path("../../Data/letter-recognition.data")
outlier_path = Path(
    "Data/dataframe_with_outliers_3std_4.16%.csv").absolute()

# get the whole alphabet
alphabet = list(string.ascii_uppercase)

# read the data
df_letters = pd.read_csv(outlier_path, names=list(columns.keys()))

dataframes = dict()


# In[8]:


df_letters.head(2)


# In[13]:


sum = 0

# fill dict with letter as key and filtered dataframe per letter and get the number of all entries
for letter in alphabet:
    df = df_letters.loc[(df_letters.letter == letter), :]
    df = df.drop('letter', axis=1, inplace=False)
    sum += len(df.index)
    print(letter, len(df.index))
    dataframes[letter] = df

print(sum)


# In[5]:


# get a quality check for the amount of entries
amount = len(df_letters.index)
amount


# In[6]:


# get the means for each letter per column
means = df_letters.groupby("letter").mean()
print(df_letters)


# Prepare for Hyperparameter testing


# Algorithm

def hyper_parameter_testing(dataframes, extents, n_neighbors, critical_values):
    columns = ["e", "n", "%"]
    columns = columns + alphabet
    columns.append("sum")
    print(columns)
    optimization_results = pd.DataFrame(columns=columns)

    for extent in extents:  # hyper parameter 1
        for n in n_neighbors:  # hyper parameter 2
            for value in critical_values:  # hyper parameter 3
                outliers = 0
                # the row which should be added
                row = {"e": extent, "n": n, "%": value}

                print(
                    f"Results for extent = {extent}, n_neighbors= {n} and critical value = {value}")
                for letter, dataframe in dataframes.items():
                    # print(dataframe.isnull().sum())
                    dataframe = dataframe.astype(float)
                    m = loop.LocalOutlierProbability(
                        dataframe, n_neighbors=n, extent=extent).fit()
                    # get the probabilities per feature
                    scores = list(m.local_outlier_probabilities)
                    part_outlier_number = 0
                    for score in scores:
                        if score >= value:  # consider it as an outlier, when score > critical value
                            part_outlier_number += 1
                    # add the outlier of a part-dataframe to the total outlier number
                    outliers += part_outlier_number
                    # calculate the percentage of outliers for the letter and add it to the row
                    percentage = part_outlier_number/len(dataframe.index)
                    row[letter] = percentage
                    print(f"{letter}: {percentage}")

                # add the percentage of total outliers to the row
                row["sum"] = outliers/amount
                print(f"Overall Result: {outliers/amount}")

                optimization_results = optimization_results.append(
                    row, ignore_index=True)  # insert the row to the dataframe
    # print(optimization_results)
    return optimization_results


# parameters to diviate
extents = [2]  # number of standarddivations, the value has to diviate
#n_neighbors = range(10, 21) 
n_neighbors = [10] # number of neighbors to consider
# citical Local outlier probability - min value to consider as a outlier
critical_values = np.arange(0.60, 0.71, 0.05)
''''
extent = 2
n = 10
value = 0.8
'''

# Results
results = hyper_parameter_testing(
    dataframes=dataframes, extents=extents, n_neighbors=n_neighbors, critical_values=critical_values)
# save the result in a csv file to continue working later
results.to_csv(
    "results_lop_3std_4,16%_06_07.csv")
