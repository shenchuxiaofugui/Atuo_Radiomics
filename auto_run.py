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
from Featuretype import split_feature_type, split_image_type
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Evaluate import Evaluate
from Metric import EstimatePrediction
from DataContainer import DataContainer
from copy import deepcopy

# def main(csv_path):


class Radiomics():
    def __init__(self, selectors, classifiers, savepath, max_feature_num=10, random_seed=1):
        self.max_feature_num = max_feature_num
        self.selectors = selectors
        self.classifiers = classifiers
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.random_seed = random_seed
        self._up = UpSampling()
        self.savepath = savepath
        self.combine_features = None



    def load_csv(self, train_path, test_path):
        pcc = DimensionReductionByPCC(threshold=0.99)
        train_data = pd.read_csv(train_path, index_col=0)
        train_data = pcc.run(train_data)
        self.train_data = z_score_normalize(train_data)
        test_data = pd.read_csv(test_path, index_col=0)
        self.test_data = z_score_normalize(test_data)
        self.train_label = self.train_data["label"].values
        self.test_label = self.test_data["label"].values
        print("数据加载完毕")

    def save_prediction(self, label, data_df, model, predict_store_path, key):
        #这个用来预测和保存预测结果，返回指标的字典
        predict_columns = ['label', 'Pred']
        prediction = model.predict(data_df.values[:, 1:])
        new_data = np.concatenate((label[:, np.newaxis], prediction[:, np.newaxis]), axis=1)
        predict_df = pd.DataFrame(data=new_data, index=data_df.index, columns=predict_columns)
        predict_df.to_csv(predict_store_path + f"/{key}_prediction.csv")
        metrics = EstimatePrediction(prediction, label, key)
        return metrics

    def run(self):
        #这个是全部的流程
        feature_types = ['firstorder', 'texture', 'shape']
        image_types = ['original', 'log-sigma', 'wave']
        train_image_dfs = split_image_type(self.train_data)
        test_image_dfs = split_image_type(self.test_data)
        self.combine_features = ["label"]
        combine_store_path = os.path.join(self.savepath, "original")
        for image_type, train_df, test_df in zip(image_types, train_image_dfs, test_image_dfs):
            print("*"*10+image_type+"*"*10)
            store_path = os.path.join(self.savepath, image_type)
            os.makedirs(store_path, exist_ok=True)
            train_shape_df, train_first_df, train_texture_df = split_feature_type(train_df)
            # test_shape_df, test_first_df, test_texture_df = split_feature_type(test_df)
            total_train = [train_first_df, train_texture_df, train_shape_df]
            # total_test = [test_shape_df, test_first_df, test_texture_df]
            candidate_feature = ['label']
            if image_type == "original":
                j = 3
            else:
                j = 2
            for i in range(j):
                feature_type = feature_types[i]
                print(f'    training {feature_type} model')
                temp_train = total_train[i]
                sub_select_features, max_val_AUC, _ = self.cross_validation(temp_train)
                if len(sub_select_features) > 0:
                    print(
                        f'        best {feature_type} model val AUC {max_val_AUC} feature num {len(sub_select_features)}')
                    candidate_feature.extend(sub_select_features)
            print(f'        there are {len(candidate_feature) - 1} feature num for final radiomics')
            selected_features = self.predict_save(train_df, test_df, candidate_feature, store_path)
            # 这里一个图像类型已经跑完了，保存下，开始跑联合模型
            self.combine_features += selected_features
            if image_type != "original":
                combine_store_path += f"+{image_type}"
                os.makedirs(combine_store_path, exist_ok=True)
                self.predict_save(self.train_data, self.test_data, self.combine_features, combine_store_path)


    def cross_validation(self, train_df, onese=True):
        # 这个用来跑子模型
        cv = StratifiedKFold(shuffle=True, random_state=self.random_seed * 10)
        max_val_AUC = 0
        selected_features = []
        best_classifier = None
        for selection in self.selectors:
            for modeling in self.classifiers:
                for k in range(self.max_feature_num):
                    if k > (len(list(train_df)) - 1):
                        break
                    rfe = selection(n_features_to_select=k + 1)
                    selected_train_df = rfe.run(train_df)

                    fold5_auc = []
                    for l, (train_index, val_index) in enumerate(cv.split(train_df.values[:, 1:], train_df['label'].values)):
                        real_index = selected_train_df.index
                        cv_train_df = set_new_dataframe(selected_train_df, real_index[train_index])
                        cv_val_df = set_new_dataframe(selected_train_df, real_index[val_index])
                        upsampling_cv_train_df = self._up.run(cv_train_df)

                        model = modeling(upsampling_cv_train_df)
                        cv_train_predict = model.predict(cv_train_df.values[:, 1:])
                        cv_val_predict = model.predict(cv_val_df.values[:, 1:])

                        cv_train_label = cv_train_df['label'].tolist()
                        cv_val_label = cv_val_df['label'].tolist()
                        label = [cv_train_label, cv_val_label]
                        name = ['cv_train', 'cv_val']
                        pred = [cv_train_predict, cv_val_predict]
                        auc, ci_lower_list, ci_upper_list = draw_roc_list(pred, label, name, is_show=False)
                        fold5_auc.append(auc[1])  # 这里为啥要加auc[1]
                    mean_cv_val_auc = np.array(fold5_auc).mean()
                    # cv_aucs.append(mean_cv_val_auc)
                    if mean_cv_val_auc > max_val_AUC and mean_cv_val_auc > 0.6:
                        max_val_AUC = mean_cv_val_auc
                        selected_features = list(selected_train_df)[1:]
                        best_classifier = modeling
                print("跑完一个模型啦")
                # if onese:
                #     feat  明天写明天写
        return selected_features, max_val_AUC, best_classifier

    def predict_save(self, train_df, test_df, candidate_feature, store_path):
        #这个用来跑总模型和保存结果
        metrics = {}
        train_df = train_df[candidate_feature]
        test_df = test_df[candidate_feature]
        train_df.to_csv(os.path.join(store_path, "train_data.csv"))
        test_df.to_csv(os.path.join(store_path, "test_data.csv"))
        select_features, max_val_AUC, best_classifier = self.cross_validation(train_df)
        print("即将胜利啦,挑了这么多特征", len(select_features))
        select_results = deepcopy(select_features)
        select_features.insert(0, "label")
        selected_train_df = train_df[select_features]
        selected_test_df = test_df[select_features]
        selected_train_df.to_csv(os.path.join(store_path, "selected_train_data.csv"))
        selected_test_df.to_csv(os.path.join(store_path, "selected_test_data.csv"))
        upsampling_train_df = self._up.run(selected_train_df)
        model = best_classifier(upsampling_train_df)
        model.save(store_path)
        train_label = selected_train_df["label"].values
        test_label = test_df["label"].values
        train_metric = self.save_prediction(train_label, selected_train_df, model, store_path, "train")
        test_metric = self.save_prediction(test_label, selected_test_df, model, store_path, "test")
        metrics.update(train_metric)
        metrics.update(test_metric)
        metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['values'])
        metric_df.to_csv(store_path+"/metric_info.csv")
        print("好耶跑完一个图像了")
        return select_results

    def get_selected_dataframe(self):
        return self.train_data[self.combine_features], self.test_data[self.combine_features]


class Multi_modal():
    def __init__(self, single_model: Radiomics, savepath, modals, max_feature_num=10, random_seed=1):
        self.features = None
        self.max_feature_num = max_feature_num
        self.singel_model = single_model
        self.train_label = None
        self.test_label = None
        self.random_seed = random_seed
        self._up = UpSampling()
        self.root = savepath
        self.train_df = None
        self.test_df = None
        self.modals = modals

    def run_single(self):
        #这个用来跑所有的模型
        for l, modality in enumerate(self.modals):
            single_path = os.path.join(self.root, modality)
            a.savepath = single_path
            self.single_model.load_csv(single_path+"/train_numeric_feature.csv", single_path+"/test_numeric_feature.csv")
            self.single_model.run()
            if l == 0:
                self.train_df, self.test_df = self.single_model.get_selected_dataframe()
            else:
                train_df, test_df = self.single_model.get_selected_dataframe()
                self.train_df = pd.merge(self.train_df, train_df, left_index=True, right_index=True, on=["label"])
                self.test_df = pd.merge(self.test_df, test_df, left_index=True, right_index=True, on=["label"])
                assert len(train_df) != len(self.train_df), "merge wrong"
        store_path = os.path.join(self.root, "combine_radiomics")
        os.makedirs(store_path, exist_ok=True)
        self.single_model.predict_save(self.train_df, self.test_df, list(train_df), store_path)

    def combine_with_df(self):
        #这里加入临床表也可以，不过是组学特征直接拼接临床表的形式
        for l, modality in enumerate(self.modals):
            train_df_path = os.path.join(self.root, modality, "original+log-sigma+wave", "selected_train_data.csv")
            test_df_path = os.path.join(self.root, modality, "original+log-sigma+wave", "selected_test_data.csv")
            if l == 0:
                self.train_df = pd.read_csv(train_df_path, index_col=0)
                self.test_df = pd.read_csv(test_df_path, index_col=0)
            else:
                train_df = pd.read_csv(train_df_path, index_col=0)
                test_df = pd.read_csv(test_df_path, index_col=0)
                self.train_df = pd.merge(self.train_df, train_df, left_index=True, right_index=True, on=["label"])
                self.test_df = pd.merge(self.test_df, test_df, left_index=True, right_index=True, on=["label"])
                assert len(train_df) != len(self.train_df), "merge wrong"
        store_path = os.path.join(self.root, "combine_radiomics")
        os.makedirs(store_path, exist_ok=True)
        self.singel_model.predict_save(self.train_df, self.test_df, list(train_df), store_path)

def merge_radiomics_clinic(pred_df, clinical_df, a, store_path):
    #组学预测结果跟临床特征拼接的形式
    radiomics_clinical = pd.merge(pred_df, clinical_df, left_index=True, right_index=True, on=["label"])
    #算了没必要写成一个函数




if __name__ == "__main__":
    #单模态
    a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA], [LR, SVM], r"D:\python\dataset\test", max_feature_num=10)
    a.load_csv(r"D:\python\dataset\test\train_numeric_feature.csv", r"D:\python\dataset\test\test_numeric_feature.csv")
    a.run()
    #多模态
    a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA], [LR, SVM], r"D:\python\dataset\test", max_feature_num=10)
    root = r""
    b = Multi_modal(a, root, ["DWI", "T1CE","T2"])