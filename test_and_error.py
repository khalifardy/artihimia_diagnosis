# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataku = pd.read_csv("eda_data_arythimia.csv")
# %%
x = dataku.drop(columns=["diagnosis"])
y = dataku["diagnosis"]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=23)
# %%
model = GaussianNB()

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
# %%
