#%%
import pandas as pd
import numpy as np
# %%
pd.set_option('display.max_row', 1000)
dataku = pd.read_csv("arrhythmia.csv")
# %%
# melihat class klasifikasi
list_unik =sorted([i for i in dataku['diagnosis'].unique()])

# %%
#menganti tanda tanya dengan value kosong
dataku = dataku.replace(to_replace="?", value=None)
# %%
#menghimpun kolom yang tidak full
kolom_tidak_full = [i for i in dataku.columns if dataku[i].isna().sum()>0]
# %%
#Mendapatkan kolom numerik
kolom_numerik = [i for i in dataku.columns if dataku[i].dtypes != "O"]
#mendapatkan kolom kategorical
kolom_ordinal = [i for i in dataku.columns if dataku[i].dtypes == 'O']
# %%
dataku[kolom_ordinal] = dataku[kolom_ordinal].astype(float)