# %%
import pandas as pd
import numpy as np
from library_project.dataproses import split_data
from library_project.metode import KnearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %%
# import data
dataku = pd.read_csv("feature_data_arythimia_std.csv")
# %%
# bagi menjadi 4 data dan split menjadi train dan test data
data1 = split_data(dataku[:113], 70, "right", True)
data2 = split_data(dataku[113:226], 70, "left", True)
data3 = split_data(dataku[226:339], 70, "middle", True)
data4 = split_data(dataku[339:], 70, "right", True)

# %%
train, test = data2
# %%
# prediksi akurasi k =15
knn = KnearestNeighbors()
train1, test1 = data3
akurasi_euclid = knn.evaluasi("euclidean", train1, test1, "diagnosis", k=15)
akurasi_manha = knn.evaluasi("manhattan", train1, test1, "diagnosis", k=15)

# %%
# rata-rata akurasi
list_data = [data1, data2, data4]
knn.get_average_accuracy(
    "euclidean", list_data, "diagnosis", 15, True)


# %%
# plot evaluasi akurasi 1-30
knn.plot_evaluasi_k("euclidean", list_data, "diagnosis", 1, 31)

# %%
