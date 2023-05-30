# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import scipy.stats as stats
from library_project.metode import Regresi
sns.set()

# %%
pd.set_option('display.max_row', 1000)
dataku = pd.read_csv("arrhythmia.csv")
# %%
# melihat class klasifikasi
list_unik = sorted([i for i in dataku['diagnosis'].unique()])

# %%
# menganti tanda tanya dengan value kosong
dataku = dataku.replace(to_replace="?", value=None)
# %%
# menghimpun kolom yang tidak full
kolom_tidak_full = [i for i in dataku.columns if dataku[i].isna().sum() > 0]
# %%
# Mendapatkan kolom numerik
kolom_numerik = [i for i in dataku.columns if dataku[i].dtypes != "O"]
# mendapatkan kolom kategorical
kolom_ordinal = [i for i in dataku.columns if dataku[i].dtypes == 'O']
# %%
dataku[kolom_ordinal] = dataku[kolom_ordinal].astype(float)

# %%
# mendapatkan kolom numerik
kolom_numerik = [i for i in dataku.columns if dataku[i].dtypes != "O"]
kolom_ordinal = [i for i in dataku.columns if dataku[i].dtypes == 'O']
# %%
kolom_numerik_kosong = [
    kolom for kolom in kolom_tidak_full if kolom in kolom_numerik]

# %%
# uji normalitas
data_pvalue_kolom_numerik_kosong = {}
for kolom in kolom_numerik_kosong:
    statistic, p_value = stats.normaltest(dataku[kolom].dropna())
    data_pvalue_kolom_numerik_kosong[kolom] = [p_value]

data_frame_p_value = pd.DataFrame(
    data_pvalue_kolom_numerik_kosong)

# %%
fig1, axs1 = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))
for i in range(5):
    axs1[i].hist(dataku[kolom_numerik_kosong[i]], bins=25)
    axs1[i].set_title(kolom_numerik_kosong[i])
plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))
for i in range(5):
    qqplot(dataku[kolom_numerik_kosong[i]], line="s", ax=axs[i])
    axs[i].set_title(kolom_numerik_kosong[i])

plt.tight_layout()
plt.show()

# %%
korelasi = dataku.dropna().drop("sex", axis=1).corr()
korelasi_tinggi = korelasi[korelasi[kolom_numerik_kosong].abs(
) >= 0.5][kolom_numerik_kosong]

# %%
# isi kolom kosong
kol_x = {
    'T': 'DIII_187',
    'P': 'AVL_206',
    'QRST': 'DIII_189',
    'Heart Rate': 'Q-T',
    'J': 'AVF_210'
}


for kolom in kolom_tidak_full:
    reg = Regresi(dataku)
    dataku = reg.isi_missing_value(kol_x[kolom], kolom, "sederhana")

# %%
# cek lagi distribusi data kosong
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))
for i in range(5):
    qqplot(dataku[kolom_numerik_kosong[i]], line="s", ax=axs[i])
    axs[i].set_title(kolom_numerik_kosong[i])

plt.tight_layout()
plt.show()

# %%
fig1, axs1 = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))
for i in range(5):
    axs1[i].hist(dataku[kolom_numerik_kosong[i]], bins=25)
    axs1[i].set_title(kolom_numerik_kosong[i])
plt.tight_layout()
plt.show()

# %%
dataku.to_csv('eda_data_arythimia.csv', index=False)
# %%
# cek outliers
semua_kolom = [i for i in dataku.columns if i not in [
    "index", "sex", "diagnosis"]]

# %%

# %%
