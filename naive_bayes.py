# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from library_project.metode import NaiveBayes
from library_project.dataproses import split_data
sns.set()

# %%
dataku = pd.read_csv("feature_data_arythimia_std.csv")
# %%
train, validation = split_data(dataku, 70, "right", True)
# %%
kelas = sorted([i for i in train["diagnosis"].unique()])
nb = NaiveBayes(train, "diagnosis", kelas)
# %%
valid, truth_data = validation.drop(
    columns=["diagnosis"]), validation["diagnosis"]
# %%
nb.cetak_hasil(valid, truth_data)
# %%
