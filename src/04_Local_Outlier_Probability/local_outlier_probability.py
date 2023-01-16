#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import matplotlib.pyplot as plt
import string


# Reading the data and get an overview

# In[2]:


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
    "Data/dataframe_with_outliers_3std_12_5%.csv").absolute()

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

# In[7]:


# parameters to diviate
extents = [1, 2, 3]  # number of standarddivations, the value has to diviate
n_neighbors = range(1, 21)  # number of neighbors to consider
# citical Local outlier probability - min value to consider as a outlier
critical_values = np.arange(0.80, 1, 0.01)


# In[11]:


# make the columns for the results dataframe
columns = ["e", "n", "%"]
columns = columns + alphabet
columns.append("sum")
print(columns)
optimization_results = pd.DataFrame(columns=columns)
optimization_results

extent = 2
n = 10
value = 0.8

# Algorithm


def algorithm(dataframes, extent, n, value):

    outliers = 0
    row = {"e": extent, "n": n, "%": value}  # the row which should be added

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
    print(optimization_results)
    return optimization_results


# Hypter-parameter testing

# In[12]:

def hyper_parameter_testing(dataframes, extents, n_neighbors, critical_values):

    for extent in extents:  # hyper parameter 1
        for n in n_neighbors:  # hyper parameter 2
            for value in critical_values:  # hyper parameter 3
                outliers = 0
                # the row which should be added
                row = {"e": extent, "n": n, "%": value}

                print(
                    f"Results for extent = {extent}, n_neighbors= {n} and critical value = {value}")
                for letter, dataframe in dataframes.items():
                    print(f"{letter} inserted")
                    m = loop.LocalOutlierProbability(
                        dataframe, extent=extent, n_neighbors=n).fit()
                    # get the probabilities per feature
                    scores = list(m.local_outlier_probabilities)
                    part_outlier_number = 0
                    for score in scores:
                        if score >= value:  # consider it as an outlier, when score > critical value
                            part_outlier_number += 1
                    # add the outlier of a part-dataframe to the total outlier number
                    outliers += part_outlier_number
                    # calculate the percentage of outliers for the letter and add it to the row
                    row[letter] = part_outlier_number/len(dataframe.index)

                # add the percentage of total outliers to the row
                row["sum"] = outliers/amount

                optimization_results = optimization_results.append(
                    row, ignore_index=True)  # insert the row to the dataframe
    # print(optimization_results)
    return optimization_results


# Results

# save the result in a csv file to continue working later
optimization_results.to_csv(
    "local_outlier_probability_hyper_parameter_tuning_with_outlier.csv")


# Plotting

# In[4]:


# read the file
optimization_data = pd.read_csv("Optimization_data.csv")
optimization_data


# In[5]:


# filter the data to have a look, which impact which parameter has
result_e = optimization_data.groupby(["e"]).mean().reset_index()
result_n = optimization_data.groupby(["n"]).mean().reset_index()
result_per = optimization_data.groupby(["%"]).mean().reset_index()


# In[8]:


# make the column to string for plotting
result_e["e"] = ["e1", "e2", "e3"]


# In[9]:


# make a alphabet list for plotting
alphabet = list(string.ascii_uppercase)
alphabet.append("sum")
print(alphabet)


# In[47]:


# plot the extent variable
fig = plt.figure(figsize=(20, 10))

for letter in alphabet:
    if letter == "sum":
        plt.plot(result_e["e"], result_e[letter],
                 "b-", label=letter, linewidth=3)
    else:
        plt.plot(result_e["e"], result_e[letter],
                 ":", label=letter, linewidth=1)

plt.xlabel("Extent")
plt.ylabel("Percentage of outliers")
plt.title("Effect of the extent")
plt.legend()
plt.grid(color='#DDDDDD', linestyle=':', linewidth=0.5)
plt.savefig("local_out_prob_extent.pdf")
plt.show()


# In[46]:


# plot the neighbors variable
fig = plt.figure(figsize=(20, 10))

for letter in alphabet:
    if letter == "sum":
        plt.plot(result_n["n"], result_n[letter],
                 "b-", label=letter, linewidth=3)
    else:
        plt.plot(result_n["n"], result_n[letter],
                 ":", label=letter, linewidth=1)

plt.xlabel("Neighbors")
plt.ylabel("Percentage of outliers")
plt.title("Effect of the Neighbors")
plt.legend()
plt.grid(color='#DDDDDD', linestyle=':', linewidth=0.5)
plt.xticks(range(0, 21, 2))
plt.savefig("local_out_prob_neighbors.pdf")
plt.show()


# In[10]:


# plot the critical value variable
fig = plt.figure(figsize=(20, 10))

for letter in alphabet:
    if letter == "sum":
        plt.plot(result_per["%"], result_per[letter],
                 "b-", label=letter, linewidth=3)
    else:
        plt.plot(result_per["%"], result_per[letter],
                 ":", label=letter, linewidth=1)

plt.xlabel("Critical value (percentage)")
plt.ylabel("Percentage of outliers")
plt.title("Effect of the critical value")
plt.legend()
plt.grid(color='#DDDDDD', linestyle=':', linewidth=0.5)
plt.xticks(range(80, 101, 2))
plt.savefig("local_out_prob_crit_val.pdf")
plt.show()


# In[ ]:
