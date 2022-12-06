# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing


def z_score_normalize(dataframe):
    data = np.array(dataframe.values[:, 1:])
    label = np.array(dataframe['label'].tolist())
    feature_name = dataframe.columns.tolist()

    # z-score标准化
    z_score_scaler = preprocessing.StandardScaler().fit(data)
    processed_data = z_score_scaler.transform(data)
    new_data = np.concatenate((label[..., np.newaxis], processed_data), axis=1)

    new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=feature_name)
    return new_dataframe


if __name__ == '__main__':
    data_path = r'.\data\train_numeric_feature.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=1, how='any')
    z_score_dataframe = z_score_normalize(df)
    print(np.array(z_score_dataframe.values[:, 1]).mean())
    print(np.array(z_score_dataframe.values[:, 1]).std())
