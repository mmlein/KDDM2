import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import string
from scipy.io import loadmat

path_with_letters = Path(
    "C:/Users/dt/Documents/CodingProjects/kddm2/Data/letter-recognition.data")
outlier_file = Path(
    "C:/Users/dt/Documents/CodingProjects/kddm2/Data/letter.mat")

with_letters = pd.read_csv(path_with_letters)

outlier_dict = loadmat(outlier_file)
outlier_df_X = pd.DataFrame(outlier_dict['X'])
outlier_df_y = pd.DataFrame(outlier_dict['y'])


print(outlier_df_X)

'''
corrmat = letter_df_X_old.corr()
round(corrmat,2)
sns.heatmap(corrmat)
plt.show()
'''
