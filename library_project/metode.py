import pandas as pd
import numpy as np


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
