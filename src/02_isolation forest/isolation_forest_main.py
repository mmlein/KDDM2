import matplotlib.colors as mcolors
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
from isolation_forest import *


path_data = Path(
    'C:/Users/Florentina\Documents/Uni CSS/3. Semester/kddm2/Data/dataframe_with_outliers_3std_4.16%.csv')

header = list(range(1, 17))
header = ["letter"] + header

letter_df_complete = pd.read_csv(path_data,
                                 sep=",")

hyper_params = {
    'cont': [0.01, 0.05, 0.1],
    'max_feat': [1, 8, 16],
    'max_samp': [10, 50, 100],
    'max_est': [10, 100],
    'random_state': [0]
}

res_abs, res_rel = algorithm(letter_df_complete)


#res_abs, res_rel = isolation_forest.cross_validation(
#    hyper_params, dataset_complete=letter_df_complete)

#print(res_abs, res_rel)

# ### Plots
"""


results = res_rel.copy()

results.loc[:, "A":"Z"] = res_rel.loc[:, "A":"Z"]/100
results.to_csv(
    "C:/Users/Florentina/Documents/Uni CSS/3. Semester/Results_ifor/Results_Ifor_3std_4,16%.csv")



alphabet = list(string.ascii_uppercase)
res_abs = res_abs.fillna(0)
res_rel = res_rel.fillna(0)
alpha_s = ["A", "B", "C"]

plt.figure(figsize=(10, 20))
ax = sns.violinplot(data=results[alphabet], orient="h", palette="Blues")
ax.set(xlabel="percentage of outliers",
       ylabel="letter", title="Density of outliers")
plt.show()
#sns.kdeplot(res_rel["A"], shade = True)


res_abs_plot = pd.DataFrame(columns=["Letter", "Average"])
res_abs_plot["Letter"] = alphabet

for i in alphabet:
    m = res_abs[i].mean()
    res_abs_plot.loc[(res_abs_plot.Letter == i), 'Average'] = m


# Reorder the dataframe
res_abs_plot = res_abs_plot.sort_values(by=['Average'])

# initialize the figure
plt.figure(figsize=(20, 10))
ax = plt.subplot(111, polar=True)
plt.axis('off')

# Constants = parameters controling the plot layout:
upperLimit = 100
lowerLimit = 30
labelPadding = 4

# Compute max and min in the dataset
max = res_abs_plot['Average'].max()

# Let's compute heights: they are a conversion of each item value in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (max - lowerLimit) / max
heights = slope * res_abs_plot.Average + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(res_abs_plot.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(res_abs_plot.index)+1))
angles = [element * width for element in indexes]
angles


my_cmap = plt.get_cmap("plasma")

# Draw bars
bars = ax.bar(
    x=angles,
    height=heights,
    width=width,
    bottom=lowerLimit,
    linewidth=2,
    edgecolor="white",
    color=my_cmap.colors,
)

# Add labels
for bar, angle, height, label in zip(bars, angles, heights, res_abs_plot["Letter"]):

    # Labels are rotated. Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"

    # Finally add the labels
    ax.text(
        x=angle,
        y=lowerLimit + bar.get_height() + labelPadding,
        s=label,
        ha=alignment,
        va='center',
        rotation=rotation,
        rotation_mode="anchor")
"""