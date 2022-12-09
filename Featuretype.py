# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os


def split_feature_type(total_dataframe, store_folder=''):
    """
    :param total_dataframe: 包含所有特征的df
    :return: 形状/一阶/纹理各自的df
    """
    data = total_dataframe.values[:, 1:]
    label = total_dataframe['label'].values
    feature_name = total_dataframe.columns.tolist()[1:]

    shape_index = []
    firstorder_index = []
    texture_index = []
    for i, feature in enumerate(feature_name):
        if 'shape' in feature:
            shape_index.append(i)
        elif 'firstorder' in feature:
            firstorder_index.append(i)
        else:
            texture_index.append(i)

    shape_data = data[:, shape_index]
    shape_feature = [feature_name[t] for t in shape_index]
    shape_feature.insert(0, 'label')
    shape_data = np.concatenate((label[..., np.newaxis], shape_data), axis=1)
    shape_dataframe = pd.DataFrame(data=shape_data, index=total_dataframe.index, columns=shape_feature)

    firstorder_data = data[:, firstorder_index]
    firstorder_feature = [feature_name[t] for t in firstorder_index]
    firstorder_feature.insert(0, 'label')
    firstorder_data = np.concatenate((label[..., np.newaxis], firstorder_data), axis=1)
    firstorder_dataframe = pd.DataFrame(data=firstorder_data, index=total_dataframe.index, columns=firstorder_feature)

    texture_data = data[:, texture_index]
    texture_feature = [feature_name[t] for t in texture_index]
    texture_feature.insert(0, 'label')
    texture_data = np.concatenate((label[..., np.newaxis], texture_data), axis=1)
    texture_dataframe = pd.DataFrame(data=texture_data, index=total_dataframe.index, columns=texture_feature)

    if store_folder != '':
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        shape_dataframe.to_csv(os.path.join(store_folder, 'shape_features.csv'))
        firstorder_dataframe.to_csv(os.path.join(store_folder, 'firstorder_features.csv'))
        texture_dataframe.to_csv(os.path.join(store_folder, 'texture_features.csv'))

    return shape_dataframe, firstorder_dataframe, texture_dataframe


def split_image_type(total_df, image_types=['original', 'log-sigma', 'wave'], store_folder='', split="train"):
    label_df = total_df.iloc[:, :1]
    all_image_df = []
    for image in image_types:
        image_df = total_df.filter(regex=f'{image}')
        image_df = label_df.join(image_df)
        if store_folder != '':
            os.makedirs(store_folder+f'/{image}', exist_ok=True)
            image_df.to_csv(os.path.join(store_folder, image, f'/{split}_features.csv'))
        all_image_df.append(image_df)
    return all_image_df