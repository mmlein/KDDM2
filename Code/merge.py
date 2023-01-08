import pandas as pd
import numpy as np
from pathlib import Path
from PyNomaly import loop
import string
from scipy.io import loadmat

path_with_letters = Path("C:/Users/Dani/Documents/temp/kddm2\Data/letter-recognition.data")
outlier_file = Path("C:/Users/Dani/Documents/temp/kddm2/Data/letter.mat")


annots = loadmat(outlier_file)
print(annots)