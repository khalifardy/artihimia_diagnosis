import pandas as pd


def split_data(data, traininggPercentage, location, shuffle=False):
    lengthTraining = int(len(data)*traininggPercentage/100)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    train = []
    validation = []

    if (location == "left"):
        train, validation = data.iloc[:lengthTraining].reset_index(
            drop=True), data.iloc[lengthTraining:].reset_index(drop=True)
    elif (location == "right"):
        validation, train = data.iloc[:abs(lengthTraining-len(data))].reset_index(
            drop=True), data.iloc[abs(lengthTraining-len(data)):].reset_index(drop=True)
    elif (location == 'middle'):
        train = data.iloc[int(abs(lengthTraining-len(data))/2):len(data)-int(abs(lengthTraining-len(data))/2)].reset_index(
            drop=True)
        validation = pd.concat([data.iloc[:int(abs(lengthTraining-len(data))/2)],
                               data.iloc[len(data)-int(abs(lengthTraining-len(data))/2):]]).reset_index(
            drop=True)

    return train, validation
