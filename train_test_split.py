'''
Split the final data into training, testing and PCA analysis sets.
'''
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def my_train_test_split(symbol="MFC", flag=0):
    os.chdir(symbol)
    df = pd.read_csv(symbol+"_final.csv")
    data = np.array(df.values.tolist())

    # +1 to actual index because pandas adds index column while loading data
    x = data[:-1, 2:]
    if flag == 0:
        y = data[1:, 2:6]
    elif flag == 1:  # only High
        y = data[1:, 2]
    elif flag == 2:  # only Low
        y = data[1:, 3]
    elif flag == 3:  # only Close
        y = data[1:, 4]
    elif flag == 4:  # only Adj Close
        y = data[1:, 5]

    assert len(x) == len(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=6)

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    for_pca = data[:, 7:]
    os.chdir("..")

    return x_train.astype(float), x_test.astype(float), y_train.astype(float), y_test.astype(float), for_pca.astype(float)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, for_pca = my_train_test_split()
    print(len(x_train[0]))
