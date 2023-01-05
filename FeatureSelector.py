# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.utils import safe_mask
from scipy.stats import kruskal


class FeatureSelectByRFE(object):
    def __init__(self, n_features_to_select=20, classifier=SVC(kernel='linear')):
        self.n_features_to_select = n_features_to_select
        self.__classifier = classifier
        self._rank = None
        pass

    def get_selected_feature_index(self, dataframe):
        data = np.array(dataframe.values[:, 1:])
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = np.array(dataframe['label'].tolist())

        if data.shape[1] < self.n_features_to_select:
            print('RFE: The number of features {:d} in dataframe is smaller than the required number {:d}'.format(
                data.shape[1], self.n_features_to_select))
            self.n_features_to_select = data.shape[1]

        fs = RFE(self.__classifier, n_features_to_select=self.n_features_to_select, step=0.05)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        self._rank = fs.ranking_

        return feature_index.tolist()

    def run(self, dataframe, store_folder=''):
        data = np.array(dataframe.values[:, 1:])
        label = np.array(dataframe['label'].tolist())
        feature_name = dataframe.columns.tolist()[1:]
        selected_index = self.get_selected_feature_index(dataframe)

        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'RFE_selected_features.csv'))

        return new_dataframe

    def get_name(self):
        return "RFE"


class FeatureSelectByANOVA(object):
    def __init__(self, n_features_to_select=20):
        self.n_features_to_select = n_features_to_select
        self._f_value = np.array([])
        self._p_value = np.array([])

    def GetSelectedFeatureIndex(self, data, label):
        if data.shape[1] < self.n_features_to_select:
            print(
                'ANOVA: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                    data.shape[1], self.n_features_to_select))
            self.n_features_to_select = data.shape[1]
        fs = SelectKBest(f_classif, k=self.n_features_to_select)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        f_value, p_value = f_classif(data, label)
        return feature_index.tolist(), f_value, p_value

    def run(self, dataframe, store_folder=''):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()[1:]

        selected_index, self._f_value, self._p_value = self.GetSelectedFeatureIndex(data, label)
        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'ANOVA_selected_features.csv'))


        return new_dataframe

    def get_name(self):
        return "AMPVA"


class FeatureSelectByRelief(object):
    def __init__(self, n_features_to_select=10, iter_ratio=1):
        self.n_features_to_select = n_features_to_select
        self.__iter_radio = iter_ratio
        self._weight = None

    def __SortByValue(self, feature_score):

        feature_list = []
        sorted_feature_number_list = []
        for feature_index in range(len(feature_score)):
            feature_list_unit = []
            feature_list_unit.append(feature_score[feature_index])
            feature_list_unit.append(feature_index)
            feature_list.append(feature_list_unit)
        sorted_feature_list = sorted(feature_list, key=lambda x: abs(x[0]), reverse=True)
        for feature_index in range(len(sorted_feature_list)):
            sorted_feature_number_list.append(sorted_feature_list[feature_index][1])

        return sorted_feature_number_list

    def __DistanceNorm(self, Norm, D_value):
        # initialization

        # Norm for distance
        if Norm == '1':
            counter = np.absolute(D_value)
            counter = np.sum(counter)
        elif Norm == '2':
            counter = np.power(D_value, 2)
            counter = np.sum(counter)
            counter = np.sqrt(counter)
        elif Norm == 'Infinity':
            counter = np.absolute(D_value)
            counter = np.max(counter)
        else:
            raise Exception('We will program this later......')

        return counter

    def __SortByRelief(self, dataframe):
        data = np.array(dataframe.values[:, 1:])
        label = dataframe['label'].values

        # initialization
        (n_samples, n_features) = np.shape(data)
        distance = np.zeros((n_samples, n_samples))
        weight = np.zeros(n_features)

        if self.__iter_radio >= 0.5:
            # compute distance
            for index_i in range(n_samples):
                for index_j in range(index_i + 1, n_samples):
                    D_value = data[index_i] - data[index_j]
                    distance[index_i, index_j] = self.__DistanceNorm('2', D_value)
            distance += distance.T
        else:
            pass

            # start iteration

        for iter_num in range(int(self.__iter_radio * n_samples)):
            # print iter_num;
            # initialization
            nearHit = list()
            nearMiss = list()
            distance_sort = list()

            # random extract a sample
            index_i = iter_num
            self_features = data[index_i]

            # search for nearHit and nearMiss
            if self.__iter_radio >= 0.5:
                distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index_i, index], index, label[index]])
            else:
                # compute distance respectively
                distance = np.zeros(n_samples)
                for index_j in range(n_samples):
                    D_value = data[index_i] - data[index_j]
                    distance[index_j] = self.__DistanceNorm('2', D_value)
                distance[index_i] = np.max(distance)  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index], index, label[index]])
            distance_sort.sort(key=lambda x: x[0])
            for index in range(n_samples):
                if len(nearHit) == 0 and distance_sort[index][2] == label[index_i]:
                    # nearHit = distance_sort[index][1];
                    nearHit = data[distance_sort[index][1]]
                elif len(nearMiss) == 0 and distance_sort[index][2] != label[index_i]:
                    # nearMiss = distance_sort[index][1]
                    nearMiss = data[distance_sort[index][1]]
                elif len(nearHit) != 0 and len(nearMiss) != 0:
                    break
                else:
                    continue

                    # update weight
            weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
        result = self.__SortByValue(weight / (self.__iter_radio * n_samples))
        self._weight = weight
        return result



    def GetSelectedFeatureIndex(self, dataframe):
        feature_sort_list = self.__SortByRelief(dataframe)
        if len(feature_sort_list) < self.n_features_to_select:
            print(
                'Relief: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                    len(feature_sort_list), self.n_features_to_select))
            self.n_features_to_select = len(feature_sort_list)
        selected_feature_index = feature_sort_list[:self.n_features_to_select]
        return selected_feature_index

    def GetDescription(self):
        text = "Before build the model, we used Relief to select features. Relief selects sub data set and find the " \
               "relative features according to the label recursively. "
        return text

    def run(self, dataframe, store_folder=''):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()[1:]

        selected_index = self.GetSelectedFeatureIndex(dataframe)
        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'Relief_selected_features.csv'))

        return new_dataframe

    def get_name(self):
        return "Relief"



class FeatureSelectByKruskalWallis(object):
    def __init__(self, n_features_to_select=10):
        self.n_features_to_select = n_features_to_select
        self._f_value = np.array([])
        self._p_value = np.array([])

    def KruskalWallisAnalysis(self, array, label):
        args = [array[safe_mask(array, label == k)] for k in np.unique(label)]
        neg, pos = args[0], args[1]
        f_list, p_list = [], []
        for index in range(array.shape[1]):
            f, p = kruskal(neg[:, index], pos[:, index])
            f_list.append(f), p_list.append(p)
        return np.array(f_list), np.array(p_list)

    def GetSelectedFeatureIndex(self, dataframe):
        data = np.array(dataframe.values[:, 1:])
        label = dataframe['label'].values

        if data.shape[1] <  self.n_features_to_select:
            print('KW: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                data.shape[1],  self.n_features_to_select))
            self.n_features_to_select = data.shape[1]

        fs = SelectKBest(self.KruskalWallisAnalysis, k=self.n_features_to_select)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        self._f_value, self._p_value = self.KruskalWallisAnalysis(data, label)
        return feature_index.tolist()

    def GetDescription(self):
        text = "Before build the model, we used Kruskal Wallis to select features. KruskalWallis was a common method " \
               "to explore the significant features corresponding to the labels. F-value was calculated to evaluate " \
               "the relationship between features and the label. We sorted features according to the corresponding " \
               "F-value and selected top N features according to validation performance."
        return text

    def run(self, dataframe, store_folder=''):
        data = dataframe.values[:, 1:]
        label = dataframe['label'].values
        feature_name = dataframe.columns.tolist()[1:]

        selected_index = self.GetSelectedFeatureIndex(dataframe)
        new_data = data[:, selected_index]
        new_feature_name = [feature_name[t] for t in selected_index]
        new_feature_name.insert(0, 'label')
        new_data = np.concatenate((label[..., np.newaxis], new_data), axis=1)
        new_dataframe = pd.DataFrame(data=new_data, index=dataframe.index, columns=new_feature_name)
        if store_folder != '':
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            new_dataframe.to_csv(os.path.join(store_folder, 'KW_selected_features.csv'))

        return new_dataframe

    def get_name(self):
        return "KW"


if __name__ == '__main__':
    data_path = r'/homes/syli/dataset/EC_all/model/dilation_model/T1CE/dilation_1/train_numeric_feature.csv'
    df = pd.read_csv(data_path, index_col=0)
    # df = df.replace(np.inf, np.nan)
    # df = df.dropna(axis=1, how='any')
    rfe = FeatureSelectByKruskalWallis(n_features_to_select=10)
    save_path = r'/homes/syli/dataset/EC_all/model/dilation_model/T1CE/output'
    output_df = rfe.run(df, save_path)
    print(output_df)
