import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
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


class NaiveBayes:
    def __init__(self, data_train, kolom_y, kelas):
        self.data_train = data_train
        self.kolom_y = kolom_y
        self.kelas = kelas

    def collect_mean(self, kolom_target):
        record = dict(self.data_train.groupby(
            self.kolom_y)[kolom_target].count())
        count = {key: record[key]+1 for key in record.keys()}
        total = dict(self.data_train.groupby(self.kolom_y)[kolom_target].sum())

        mean = {key: total[key]/count[key] for key in count.keys()}

        return mean

    def collect_std(self, kolom_target, mean):
        pembilang = 0
        std = {}
        for key in mean.keys():

            miu = mean[key]
            rec = self.data_train[kolom_target].loc[self.data_train[self.kolom_y] == key]
            for i in range(len(rec)):
                value = (rec.iloc[i]-miu)**2
                pembilang += value
            std[key] = math.sqrt(pembilang/(len(rec)+1))

        return std

    def collect_probablity_y(self):
        record = dict(self.data_train[self.kolom_y].value_counts())
        dictio = {key: record[key]+1 for key in record.keys()}
        for i in dictio.keys():
            dictio[i] = dictio[i]/(len(self.data_train)+len(self.kolom_y))

        return dictio

    def calculate_probability(self, mean, std, x):
        pembilang = math.exp(-((x-mean)**2/(2 * std**2)))
        penyebut = std*math.sqrt(2*math.pi)

        return 1/penyebut*pembilang

    def max_prob(self, dictio):
        key = [i for i in dictio.keys()]
        maxi = dictio[key[0]]
        maxi_id = key[0]

        for keys in key[1:]:
            if maxi < dictio[keys]:
                maxi_id = keys
                maxi = dictio[keys]

        return maxi_id

    def prediksi(self, data_test, truth_colom=None):
        result = []

        for i in range(len(data_test)):
            y_prob = self.collect_probablity_y()
            dict_prob = {}
            for kolom in data_test.columns:
                nilai_x = data_test[kolom].iloc[i]
                mean = self.collect_mean(kolom)
                std = self.collect_std(kolom, mean)

                for key in y_prob.keys():

                    y_prob[key] *= self.calculate_probability(
                        mean[key], std[key], nilai_x)
            dict_prob["id"] = i
            dict_prob["probabilitas"] = y_prob
            dict_prob["result"] = self.max_prob(y_prob)
            if type(truth_colom) != type(None):
                dict_prob["truth_data"] = truth_colom.iloc[i]
                print("record_id : {}, prediksi: {}, truth: {}".format(
                    dict_prob["id"], dict_prob["result"], dict_prob["truth_data"]))
            result.append(dict_prob)

        return result

    def akurasi(self, TP, TN, FP, FN):
        return (TP+TN)/(TP+TN+FP+FN)

    def presisi(self, TP, FP):
        try:
            return TP/(TP+FP)
        except:
            return 0

    def recall(self, TP, FN):
        try:
            return TP/(TP+FN)
        except:
            return 0

    def confusionMatrix(self, result):

        result_matrix = []

        for kelas in self.kelas:

            TP = 0
            TN = 0
            FP = 0
            FN = 0

            dictio = {}
            print(kelas)

            for res in result:
                if res["truth_data"] == kelas and res["truth_data"] == res["result"]:

                    TP += 1
                elif res["truth_data"] != kelas and res["truth_data"] == res["result"]:
                    TN += 1
                elif res["truth_data"] == kelas and res["truth_data"] != res["result"]:
                    FP += 1
                elif res["truth_data"] != kelas and res["truth_data"] != res["result"]:
                    FN += 1

            dictio["kelas"] = kelas
            dictio["kumpulan"] = [TP, TN, FP, FN]
            print(dictio["kumpulan"])

            dictio["akurasi"] = self.akurasi(TP, TN, FP, FN)
            dictio["presisi"] = self.presisi(TP, FP)
            dictio["recall"] = self.recall(TP, FN)
            result_matrix.append(dictio)

        return result_matrix

    def cetak_hasil(self, data_test, truth_colom=None):
        hasil = self.prediksi(data_test, truth_colom)
        matrix = self.confusionMatrix(hasil)

        akurasi = 0
        presisi = 0
        recall = 0

        for data in matrix:
            akurasi += data["akurasi"]
            presisi += data["presisi"]
            recall += data["recall"]

        akurasi = akurasi / len(self.kelas)
        presisi = presisi / len(self.kelas)
        recall = recall / len(self.kelas)

        print("akurasi: {}, presisi: {}, recall: {}".format(
            akurasi, presisi, recall))
