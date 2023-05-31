class Scalling:
    def __init__(self, data):
        self.data = data

    def normalisasi(self, kolom):
        for kol in kolom:
            self.data[kol] = (self.data[kol]-self.data[kol].min()) / \
                (self.data[kol].max()-self.data[kol].min())

        return self.data

    def standarisasi(self, kolom):
        for kol in kolom:
            self.data[kol] = (self.data[kol]-self.data[kol].mean()) / \
                (self.data[kol].std())

        return self.data
