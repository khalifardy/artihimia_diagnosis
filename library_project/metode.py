import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Regresi:
    def __init__(self, data):
        self.data = data.copy()

    def sederhana(self, kolom_x, kolom_y):
        mean_x = self.data[kolom_x].mean()
        mean_y = self.data[kolom_y].mean()

        data_ = self.data.copy()
        data_["sigma_x"] = data_[kolom_x].apply(lambda x: x-mean_x)
        data_["sigma_x^2"] = data_[kolom_x].apply(lambda x: (x-mean_x)**2)
        data_["sigma_y"] = data_[kolom_y].apply(lambda y: y-mean_y)
        data_["xy"] = data_["sigma_x"]*data_["sigma_y"]

        pembilang = data_["xy"].sum()
        penyebut = data_["sigma_x^2"].sum()

        b = pembilang/penyebut
        a = mean_y - b * mean_x

        return a, b

    def hasil_prediksi(self, kolom_x, kolom_y, x, tipe):
        if tipe == "sederhana":
            a, b = self.sederhana(kolom_x, kolom_y)
            y = a + b*x

            return y

    def isi_missing_value(self, kolom_x, kolom_y, tipe):
        id_missing = self.data[self.data[kolom_y].isna()].index

        for index in id_missing:
            x = self.data.loc[index, kolom_x]
            y = self.hasil_prediksi(kolom_x, kolom_y, x, tipe)

            self.data.loc[index, kolom_y] = y

        return self.data


class KnearestNeighbors:
    def __init__(self, k=1):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def manhattan_distance(self, x1, x2):
        return np.sum(np.absolute(x1-x2))

    def list_distance(self, tipe, xtrain, xtest, posisi):
        distance = []
        for i in range(len(xtrain)):
            if tipe == "euclidean":
                distance.append(self.euclidean_distance(
                    xtrain.loc[i], xtest.loc[posisi]))
            elif tipe == "manhattan":
                distance.append(self.manhattan_distance(
                    xtrain.loc[i], xtest.loc[posisi]))

        return distance

    def predict(self, tipe, xtrain, xtest, posisi, ytrain, k=None):

        if k is None:
            k = self.k
        distance = self.list_distance(
            tipe, xtrain, xtest, posisi)
        data_ = xtrain.copy()
        data_["distance"] = distance
        data_["label"] = ytrain

        data_ = data_.sort_values(by="distance")
        y_pred = data_[:k].label.mode()

        return y_pred[0]

    def accuracy(self, y_pred, y_test):
        benar = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                benar += 1

        return benar/len(y_test)

    def evaluasi(self, tipe, train, test, label, k=None, ):
        if k is None:
            k = self.k

        xtrain, ytrain = train.drop(label, axis=1), train[label]
        xtest, ytest = test.drop(label, axis=1), test[label]

        ypred = []
        for i in range(len(xtest)):
            ypred.append(self.predict(tipe, xtrain, xtest, i, ytrain, k))

        return self.accuracy(ypred, ytest)

    def get_average_accuracy(self, tipe, list_data, label, k=None, cetak=False):
        if k is None:
            k = self.k

        akurasi = []

        for i in range(len(list_data)):
            train, test = list_data[i]
            akurasi.append(self.evaluasi(tipe, train, test, label, k))

        if cetak:
            print("Untuk k : {} , Rata-rata akurasi:{}".format(k,
                  sum(akurasi)/len(akurasi)))
        else:
            return sum(akurasi)/len(akurasi)

    def plot_evaluasi_k(self, tipe, list_data, label, start, end, step=1):

        average_accuracy = []
        k_list = []
        for i in range(start, end, step):
            k_list.append(i)
            average_accuracy.append(
                self.get_average_accuracy(tipe, list_data, label, i))

        plt.plot(k_list, average_accuracy)
        plt.title("evaluasi K {}-{}".format(start, end-1))
        plt.xlabel("k value")
        plt.ylabel("accuracy")
        plt.show()
