# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from library_project.metode import Regresi
from library_project.preprocessing import Scalling
sns.set()

# %%
pd.set_option('display.max_row', 1000)
dataku = pd.read_csv("arrhythmia.csv")
# %%
# melihat class klasifikasi
list_unik = sorted([i for i in dataku['diagnosis'].unique()])

# %%
# Mendapatkan kolom numerik
kolom_numerik = [i for i in dataku.columns if dataku[i].dtypes != "O"]
# mendapatkan kolom kategorical
kolom_ordinal = [i for i in dataku.columns if dataku[i].dtypes == 'O']

# %%
# menganti tanda tanya dengan value kosong
dataku = dataku.replace(to_replace="?", value=None)
# %%
dataku[kolom_ordinal] = dataku[kolom_ordinal].astype(float)
# %%
# menghimpun kolom yang tidak full
kolom_tidak_full = [i for i in dataku.columns if dataku[i].isna().sum() > 0]
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
# cek outliers
semua_kolom = [i for i in dataku.columns if i not in [
    "index", "sex", "diagnosis"]]

# %%
# Cek outliers
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
row = 0
col = 0
for kolom in semua_kolom[270:277]:
    if col == 3 and row < 4:
        col = 0
        row += 1

    ax[row][col].boxplot(dataku[kolom])
    ax[row][col].set_title(kolom)

    col += 1

plt.show()


# %%
semua_kolom = [i for i in dataku.columns if i != "diagnosis"]

# %%
fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(30, 30))
row = 0
col = 0
for kolom in semua_kolom[:30]:
    if col == 5 and row < 6:
        col = 0
        row += 1

    ax[row][col].scatter(dataku[kolom], dataku["diagnosis"])
    ax[row][col].set_title(kolom+" - Diagnosis")

    col += 1

# %%

kolom_stat = [i for i in dataku.columns if i not in [
    'sex', 'diagnosis', 'EOR_Rwave', 'EODD_Rwave', 'EOR_Pwave', 'EODD_Pwave', 'EOR_Twave', 'EODD_Twave']]
# cek kolom yang homogen
kolom_homogen = []
for kolom in kolom_stat:
    varians = dataku[kolom].var()
    if varians < 16:  # karena jumlah klasifikasi total 16
        kolom_homogen.append(kolom)

# %%
# drop kolom homogen + tidak relevan
kolom_homogen.append("sex")
dataku = dataku.drop(columns=kolom_homogen)

# %%
kolom_baru = [kolom for kolom in dataku.columns if kolom not in ['diagnosis',
                                                                 'EOR_Rwave', 'EODD_Rwave', 'EOR_Pwave', 'EODD_Pwave', 'EOR_Twave', 'EODD_Twave']]
skal = Scalling(dataku)
dataku = skal.standarisasi(kolom_baru)
# %%
# korelasi pearson
nilai_korelasi_pearson = {}
for kolom in kolom_baru:
    if kolom not in ['diagnosis',
                     'EOR_Rwave', 'EODD_Rwave', 'EOR_Pwave', 'EODD_Pwave', 'EOR_Twave', 'EODD_Twave']:
        r, p = pearsonr(dataku[kolom], dataku["diagnosis"])
        nilai_korelasi_pearson[kolom] = [r]

nilai_korelasi_data_frame_pearson = pd.DataFrame(nilai_korelasi_pearson)
# %%
# korelasi spearman
nilai_korelasi_spearman = {}
for kolom in kolom_baru:
    if kolom not in ['diagnosis',
                     'EOR_Rwave', 'EODD_Rwave', 'EOR_Pwave', 'EODD_Pwave', 'EOR_Twave', 'EODD_Twave']:
        rho, p = spearmanr(dataku[kolom], dataku["diagnosis"])
        nilai_korelasi_spearman[kolom] = [rho]


nilai_korelasi_data_frame_spearman = pd.DataFrame(nilai_korelasi_spearman)

# %%
dataku[kolom_baru] = dataku[kolom_baru].astype(float)
# %%
dataku.to_csv('feature_data_arythimia.csv', index=False)

# %%
