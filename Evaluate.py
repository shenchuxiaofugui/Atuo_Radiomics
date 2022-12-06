# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, recall_score, f1_score, roc_auc_score


class Evaluate(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, predict_list, label_list, name='', store_path=''):
        if not isinstance(predict_list, list):
            predict_list = [predict_list]
        if not isinstance(label_list, list):
            label_list = [label_list]

        evaluate = dict()

        if name == '':
            name = [x for x in range(len(predict_list))]
            if len(label_list) == 2:
                name = ['train', 'test']

        for j in range(len(predict_list)):
            evaluate[name[j] + 'AUC'] = [roc_auc_score(label_list[j], predict_list[j])]
            for i in range(len(predict_list[j])):
                if predict_list[j][i] >= 0.5:
                    predict_list[j][i] = 1
                else:
                    predict_list[j][i] = 0
            evaluate[name[j] + 'accuracy'] = [accuracy_score(label_list[j], predict_list[j])]
            evaluate[name[j] + 'recall'] = [recall_score(label_list[j], predict_list[j])]
            evaluate[name[j] + 'f1'] = [f1_score(label_list[j], predict_list[j])]

        if store_path != '':
            evaluate_df = pd.DataFrame(evaluate)
            evaluate_df.to_csv(os.path.join(store_path, 'evaluate.csv'), index=False)
        return evaluate[name[0] + 'AUC'][0], evaluate[name[1] + 'AUC'][0]
