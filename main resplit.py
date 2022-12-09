# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from Classifier import LR, SVM
from DataBalance import UpSampling
from DimensionReductionByPCC import DimensionReductionByPCC
from DataSplit import DataSplit, set_new_dataframe
from DrawROC import draw_roc_list
from FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA
from Normalizer import z_score_normalize
from Featuretype import split_feature_type
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# from Evaluate import Evaluate

# def main(csv_path):



if __name__ == '__main__':
    # csv1 = r'C:\Users\HJ Wang\Desktop\ML\My_work\songyang\ibsi\IBSI_features功能 全.csv'
    # csv1 = r'C:\Users\HJ Wang\Desktop\ML\My_work\songyang\ibsi\IBSI_features结构 全.csv'
    # output_path = r'C:\Users\HJ Wang\Desktop\ML\My_work\songyang\ibsi\resplit'
    csv1 = r'C:\Users\HJ Wang\Desktop\ML\My_work\221017EC_MTL\radiomics\P53 NEW\dwi origin and log.csv'
    output_path = r'C:\Users\HJ Wang\Desktop\ML\My_work\221017EC_MTL\radiomics\P53 NEW'

    df1 = pd.read_csv(csv1, index_col=0)

    modeling = LR
    # modeling = SVM

    selection = FeatureSelectByRFE
    # selection = FeatureSelectByANOVA

    repeat_split = 30
    max_feature_num = 15
    output_dict = {'repeat_split': [],
                   'feature num': [],
                   'train AUC': [],
                   'cv val AUC': [],
                   'test AUC': [],
                   'features': [],
                   }

    pcc = DimensionReductionByPCC(threshold=0.99)
    data_split = DataSplit()
    up = UpSampling()
    df1 = pcc.run(df1)
    df1 = z_score_normalize(df1)

    # for i in range(repeat_split):
    random_seed = 1
    print(f'split random seed {random_seed*10}')
    store_path = os.path.join(output_path, str(random_seed))
    cv = StratifiedKFold(shuffle=True, random_state=random_seed*10)

    train_df, test_df, _ = data_split.run(df1, random_state=random_seed*10)

    cv_result = {'shape': [],
                 'firstorder': [],
                 'texture': []}

    train_shape_df, train_first_df, train_texture_df = split_feature_type(train_df)
    test_shape_df, test_first_df, test_texture_df = split_feature_type(test_df)

    total_train = [train_shape_df, train_first_df, train_texture_df]
    total_test = [test_shape_df, test_first_df, test_texture_df]
    feature_types = ['shape', 'firstorder', 'texture']
    # output_AUC = {'shape': [[] for i in range(max_feature_num)],
    #               'firstorder': [[] for i in range(max_feature_num)],
    #               'texture': [[] for i in range(max_feature_num)]}

    candidate_feature = ['label']
    for j in range(3):
        feature_type = feature_types[j]
        print(f'    training {feature_type} model')
        temp_train = total_train[j]
        temp_test = total_test[j]

        max_val_AUC = 0
        selected_features = []
        for k in range(max_feature_num):
            if k > (len(temp_train.columns.tolist()) - 1):
                break
            rfe = selection(n_features_to_select=k+1)
            selected_train_df = rfe.run(temp_train)

            fold5_auc = []
            for l, (train_index, val_index) in enumerate(cv.split(train_df.values[:, 1:], train_df['label'].values)):
                real_index = selected_train_df.index
                cv_train_df = set_new_dataframe(selected_train_df, real_index[train_index])
                cv_val_df = set_new_dataframe(selected_train_df, real_index[val_index])
                upsampling_cv_train_df = up.run(cv_train_df)

                model = modeling(upsampling_cv_train_df)
                cv_train_predict = model.predict(cv_train_df.values[:, 1:])
                cv_val_predict = model.predict(cv_val_df.values[:, 1:])

                cv_train_label = cv_train_df['label'].tolist()
                cv_val_label = cv_val_df['label'].tolist()
                label = [cv_train_label, cv_val_label]
                name = ['cv_train', 'cv_val']
                pred = [cv_train_predict, cv_val_predict]
                auc, ci_lower_list, ci_upper_list = draw_roc_list(pred, label, name, is_show=False)
                fold5_auc.append(auc[1])  #这里为啥要加auc[1]
            mean_cv_val_auc = np.array(fold5_auc).mean()
            if mean_cv_val_auc > max_val_AUC and mean_cv_val_auc > 0.6:
                max_val_AUC = mean_cv_val_auc
                selected_features = selected_train_df.columns.tolist()[1:]
        if len(selected_features) > 0:
            print(f'        best {feature_type} model val AUC {max_val_AUC} feature num {len(selected_features)}')
            candidate_feature.extend(selected_features)
        else:
            print(f'        no suitable {feature_type} model, val AUC {max_val_AUC} < 0.6')

    # 得到了分成三组的三个特征，再回到正常的组学流程
    print(f'        there are {len(candidate_feature)-1} feature num for final radiomics')
    train_df = train_df[candidate_feature]
    test_df = test_df[candidate_feature]
    max_val_AUC = 0
    selected_features = []
    for k in range(max_feature_num):
        if k > (len(train_df.columns.tolist()) - 1):
            break
        rfe = selection(n_features_to_select=k+1)
        selected_train_df = rfe.run(train_df)

        fold5_auc = []
        for l, (train_index, val_index) in enumerate(cv.split(train_df.values[:, 1:], train_df['label'].values)):
            real_index = selected_train_df.index
            cv_train_df = set_new_dataframe(selected_train_df, real_index[train_index])
            cv_val_df = set_new_dataframe(selected_train_df, real_index[val_index])
            upsampling_cv_train_df = up.run(cv_train_df)

            model = modeling(upsampling_cv_train_df)
            cv_train_predict = model.predict(cv_train_df.values[:, 1:])
            cv_val_predict = model.predict(cv_val_df.values[:, 1:])

            cv_train_label = cv_train_df['label'].tolist()
            cv_val_label = cv_val_df['label'].tolist()
            label = [cv_train_label, cv_val_label]
            name = ['cv_train', 'cv_val']
            pred = [cv_train_predict, cv_val_predict]
            auc, ci_lower_list, ci_upper_list = draw_roc_list(pred, label, name, is_show=False)
            fold5_auc.append(auc[1])
        mean_cv_val_auc = np.array(fold5_auc).mean()

        upsampling_train_df = up.run(selected_train_df)
        temp_test = test_df.loc[:, selected_train_df.columns.tolist()]
        model = modeling(upsampling_train_df)
        train_predict = model.predict(upsampling_train_df.values[:, 1:])
        test_predict = model.predict(temp_test.values[:, 1:])

        train_label = upsampling_train_df['label'].tolist()
        test_label = temp_test['label'].tolist()
        label = [train_label, test_label]
        name = ['train', 'val']
        pred = [train_predict, test_predict]
        auc, ci_lower_list, ci_upper_list = draw_roc_list(pred, label, name, is_show=False)

        output_dict['repeat_split'].append(random_seed*10)
        output_dict['feature num'].append(k)
        output_dict['train AUC'].append(auc[0])
        output_dict['cv val AUC'].append(mean_cv_val_auc)
        output_dict['test AUC'].append(auc[1])
        output_dict['features'].append(selected_train_df.columns.tolist())
        print(f'        feature num {k} train AUC {auc[0]} test AUC {auc[1]}')
    dataframe = pd.DataFrame(output_dict)
    dataframe.to_excel(r'C:\Users\HJ Wang\Desktop\output jiegou.xlsx')
