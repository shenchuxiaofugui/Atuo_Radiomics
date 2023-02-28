import pandas as pd
import numpy as np
import os
from Classifier import LR, SVM
from DataBalance import UpSampling
from DimensionReductionByPCC import DimensionReductionByPCC
from DataSplit import set_new_dataframe
from DrawROC import draw_roc_list
from FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA
from Featuretype import split_feature_type, split_image_type
from sklearn.model_selection import StratifiedKFold
import warnings
from Metric import EstimatePrediction
from copy import deepcopy
from time import time
from multiprocessing import Process
import json
import split_by_p

warnings.simplefilter(action='ignore', category=FutureWarning)
# def main(csv_path):


def split_and_merge(feature_root, modals, clinical_path, store_path):
    train_df, test_df = split_by_p.main_run(clinical_path, output_path=store_path, repeat_times=200)
    for modal in modals:
        feature_df = pd.read_csv(os.path.join(feature_root, f"{modal}_features.csv"))
        new_train_df = pd.merge(train_df, feature_df, on=["CaseName"], validate="one_to_one")
        new_test_df = pd.merge(test_df, feature_df, on=["CaseName"], validate="one_to_one")
        assert len(train_df) + len(test_df) == len(new_train_df) + len(new_test_df), "拼接不对"
        store = os.path.join(store_path, modal)
        os.makedirs(store, exist_ok=True)
        new_train_df.to_csv(os.path.join(store, "train_numeric_feature.csv"), index=False)
        new_test_df.to_csv(os.path.join(store, "test_numeric_feature.csv"), index=False)


class Radiomics():
    def __init__(self, selectors, classifiers, savepath="", max_feature_num=10, random_seed=1):
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
        self.combine_features = ["label"]

    def load_csv(self, train_path, test_path, pcc_zscore=True):
        train_data = pd.read_csv(train_path, index_col=0)
        test_data = pd.read_csv(test_path, index_col=0)
        if pcc_zscore:
            pcc = DimensionReductionByPCC(threshold=0.99)
            train_data = pcc.run(train_data)
            train_data.to_csv(os.path.join(self.savepath, "train_pcc_result.csv"))
            test_data = test_data[list(train_data)]
            X = train_data.iloc[:,1:]
            mean = X.mean()
            std = X.std()
            train_data.iloc[:,1:] = (X-mean)/ std
            test_data.iloc[:, 1:] = (test_data.iloc[:, 1:]-mean)/std
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = self.train_data["label"].values
        self.test_label = self.test_data["label"].values
        print("数据加载完毕")

    def save_prediction(self, data_df, model, predict_store_path, key):
        #这个用来预测和保存预测结果，返回指标的字典
        label = data_df["label"].values.astype(int)
        predict_columns = ['label', 'Pred']
        prediction = model.predict(data_df.values[:, 1:])
        new_data = np.concatenate((label[:, np.newaxis], prediction[:, np.newaxis]), axis=1)
        predict_df = pd.DataFrame(data=new_data, index=data_df.index, columns=predict_columns)
        predict_df.to_csv(predict_store_path + f"/{key}_prediction.csv")
        metrics = EstimatePrediction(prediction, label, key)
        return metrics, prediction

    def single_image_type(self, image_type, train_df, test_df):
        feature_types = ['firstorder', 'texture', 'shape']
        print("*" * 10 + image_type + "*" * 10)
        store_path = os.path.join(self.savepath, image_type)
        os.makedirs(store_path, exist_ok=True)
        train_shape_df, train_first_df, train_texture_df = split_feature_type(train_df)
        test_shape_df, test_first_df, test_texture_df = split_feature_type(test_df)
        total_train = [train_first_df, train_texture_df, train_shape_df]
        total_test = [test_first_df, test_texture_df, test_shape_df]
        candidate_feature = ['label']
        if image_type == "original":
            j = 3
        else:
            j = 2
        for i in range(j):
            feature_type = feature_types[i]
            print(f'    training {feature_type} model')
            temp_train = total_train[i]
            temp_test = total_test[i]
            temp_store_path = os.path.join(store_path, feature_type)
            os.makedirs(temp_store_path, exist_ok=True)
            sub_select_features, max_val_AUC = self.predict_save(temp_train, temp_test, temp_store_path, False)
            if len(sub_select_features) > 0:
                print(
                    f'        best {feature_type} model val AUC {max_val_AUC} feature num {len(sub_select_features)}')
                candidate_feature.extend(sub_select_features)
        print(f'        there are {len(candidate_feature) - 1} feature num for final radiomics')
        train_df = train_df[candidate_feature]
        test_df = test_df[candidate_feature]
        self.predict_save(train_df, test_df, store_path, True)


    def _combine(self):
        print("开始跑图像组合模型")
        jobs = []
        for image_type in ['original', 'log-sigma', 'wave']:
            train_df_path = os.path.join(self.savepath, image_type, "best_model", "selected_train_data.csv")
            test_df_path = os.path.join(self.savepath, image_type, "best_model", "selected_test_data.csv")
            if image_type == "original":
                combine_path = os.path.join(self.savepath, image_type)
                combine_train_df = pd.read_csv(train_df_path, index_col=0)
                combine_test_df = pd.read_csv(test_df_path, index_col=0)
            else:
                combine_path += f"+{image_type}"
                os.makedirs(combine_path, exist_ok=True)
                train_df = pd.read_csv(train_df_path, index_col=0).iloc[:, 1:]
                test_df = pd.read_csv(test_df_path, index_col=0).iloc[:, 1:]
                combine_train_df = pd.merge(combine_train_df, train_df, left_index=True, right_index=True,
                                    validate="one_to_one")  # , validate="one_to_one"
                combine_test_df = pd.merge(combine_test_df, test_df, left_index=True, right_index=True, validate="one_to_one")
                assert len(combine_train_df) == len(train_df), "wrong"
                p = Process(target=self.predict_save, args=(combine_train_df, combine_test_df, combine_path, True))
                p.start()
                jobs.append(p)
        for job in jobs:
            job.join()



    def run(self):
        #这个是全部的流程
        start = time()
        image_types = ['original', 'log-sigma', 'wave']
        train_image_dfs = split_image_type(self.train_data)
        test_image_dfs = split_image_type(self.test_data)
        process_list = []
        #单图像
        for image_type, train_df, test_df in zip(image_types, train_image_dfs, test_image_dfs):
            p = Process(target=self.single_image_type, args=(image_type, train_df, test_df))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

        self._combine()

        end = time()
        print(f"spend {(end - start) / 3600} hours")

    def cross_validation(self, train_df, test_df=None, onese=True, save_path=""): #, save_path=False
        # 这个用来跑子模型
        cv = StratifiedKFold(shuffle=True, random_state=self.random_seed * 10)
        max_val_AUC = 0
        selected_features = []
        best_classifier, best_cv_selector, best_cv_train_info, best_cv_val_info = None, None, None, None
        for selection in self.selectors:
            for modeling in self.classifiers:
                for k in range(self.max_feature_num):
                    if k + 1 > (len(list(train_df)) - 1):
                        break
                    selector = selection(n_features_to_select=k + 1)
                    selected_train_df = selector.run(train_df)
                    fold5_auc = []
                    train_info = pd.DataFrame(columns=["label", "Pred", "group"])
                    val_info = pd.DataFrame(columns=["label", "Pred", "group"])
                    for l, (train_index, val_index) in enumerate(cv.split(train_df.values[:, 1:], train_df['label'].values)):
                        real_index = selected_train_df.index
                        cv_train_df = set_new_dataframe(selected_train_df, real_index[train_index])
                        cv_val_df = set_new_dataframe(selected_train_df, real_index[val_index])
                        upsampling_cv_train_df = self._up.run(cv_train_df)

                        model = modeling(upsampling_cv_train_df)

                        cv_train_predict = model.predict(cv_train_df.values[:, 1:])
                        cv_val_predict = model.predict(cv_val_df.values[:, 1:])

                        cv_train_df["group"] = l + 1
                        cv_val_df["group"] = l + 1
                        cv_train_df["Pred"] = cv_train_predict
                        cv_val_df["Pred"] = cv_val_predict
                        train_info = pd.concat([train_info, cv_train_df[["label", "Pred", "group"]]])
                        val_info = pd.concat([val_info, cv_val_df[["label", "Pred", "group"]]])

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
                        best_cv_selector = selector.get_name()
                        best_cv_train_info = train_info
                        best_cv_val_info = val_info
                    if save_path != '':
                        selected_test_df = test_df[list(selected_train_df)]
                        store_path = os.path.join(save_path, f"{selector.get_name()}_{k+1}", model.get_name())
                        os.makedirs(store_path, exist_ok=True)
                        upsampling_train_df = self._up.run(selected_train_df)
                        model = modeling(upsampling_train_df)
                        self._save_info(selected_train_df, selected_test_df, train_info, val_info, model, store_path)
                # if onese:
                #     feat  明天写明天写

        return selected_features, max_val_AUC, best_classifier, best_cv_selector, best_cv_train_info, best_cv_val_info

    def predict_save(self, train_df, test_df, store_path, save_median_results):
        #这个用来跑总模型和保存结果
        pipline_info = {}
        train_df.to_csv(os.path.join(store_path, "train_data.csv"))
        test_df.to_csv(os.path.join(store_path, "test_data.csv"))
        if save_median_results:
            select_features, max_val_AUC, best_classifier, used_selector, cv_train, cv_val = \
                self.cross_validation(train_df, test_df, save_path=store_path)
        else:
            select_features, max_val_AUC, best_classifier, used_selector, cv_train, cv_val = \
                self.cross_validation(train_df)
        pipline_info["selector"] = used_selector
        pipline_info["feature number"] = len(select_features)
        print("即将胜利啦,挑了这么多特征", len(select_features))
        select_results = deepcopy(select_features)
        select_features.insert(0, "label")
        selected_train_df = train_df[select_features]
        selected_test_df = test_df[select_features]

        upsampling_train_df = self._up.run(selected_train_df)
        model = best_classifier(upsampling_train_df)
        pipline_info["classifier"] = model.get_name()
        with open(os.path.join(store_path, "pipline_info.json"), "w") as f:
            json.dump(pipline_info, f)
        os.makedirs(store_path+"/best_model", exist_ok=True)
        self._save_info(selected_train_df, selected_test_df, cv_train, cv_val, model, store_path+"/best_model")
        print("好耶跑完一个图像了")
        return select_results, max_val_AUC

    def _calculate_val_metric(self, cv_val_data):
        for i in range(1, 6):
            single_cv_val = cv_val_data[cv_val_data["group"] == i]
            if i == 1:
                cv_metric = EstimatePrediction(single_cv_val["Pred"].values, single_cv_val["label"].values.astype(int),
                                               "cv_val")
                for key in cv_metric:
                    if isinstance(cv_metric[key], str):
                        cv_metric[key] = eval(cv_metric[key])
            else:
                single_cv_metric = EstimatePrediction(single_cv_val["Pred"].values,
                                                      single_cv_val["label"].values.astype(int),
                                                      "cv_val")
                for key in cv_metric:
                    if isinstance(single_cv_metric[key], str):
                        if isinstance(cv_metric[key], list):
                            cv_metric[key] = [x + y for x, y in zip(cv_metric[key], eval(single_cv_metric[key]))]
                        else:
                            cv_metric[key] = cv_metric[key] + eval(single_cv_metric[key])
                    else:
                        cv_metric[key] = cv_metric[key] + single_cv_metric[key]
        for key in cv_metric:
            if isinstance(cv_metric[key], list):
                cv_metric[key] = [round(x / 5, 4) for x in cv_metric[key]]
            else:
                cv_metric[key] = round(cv_metric[key] / 5, 4)
        return cv_metric

    def _save_info(self, selected_train_df, selected_test_df, cv_train_info, cv_val_info, model, store_path):
        model.save(store_path)
        metrics = {}
        cv_train_info.to_csv(os.path.join(store_path, "cv_train_prediction.csv"))
        cv_val_info.to_csv(os.path.join(store_path, "cv_val_prediction.csv"))
        cv_metric = self._calculate_val_metric(cv_val_info)
        selected_train_df.to_csv(os.path.join(store_path, "selected_train_data.csv"))
        selected_test_df.to_csv(os.path.join(store_path, "selected_test_data.csv"))
        train_metric, train_pred = self.save_prediction(selected_train_df, model, store_path, "train")
        test_metric, test_pred = self.save_prediction(selected_test_df, model, store_path, "test")
        draw_roc_list([train_pred, test_pred], [selected_train_df["label"].tolist(), selected_test_df["label"].tolist()],
                      ["train", "test"], store_path, is_show=False)
        metrics.update(train_metric)
        metrics.update(cv_metric)
        metrics.update(test_metric)
        metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['values'])
        metric_df.to_csv(store_path + "/metric_info.csv")

    def get_selected_dataframe(self):
        return self.train_data[self.combine_features], self.test_data[self.combine_features]


def run_all_modals(modals, root, selectors, classifiers):
    jobs = []
    for modal in modals:
        path = os.path.join(root, modal)
        a = Radiomics(selectors, classifiers, path, max_feature_num=10)
        p = Process(target=run_single, args=(a, ))
        p.start()
        jobs.append(p)
    for job in jobs:
        job.join()
    combine_with_df(modals, root, selectors, classifiers)



def run_single(single_model:Radiomics):
    #这个用来跑所有的模型
    single_model.load_csv(single_model.savepath+"/train_numeric_feature.csv", single_model.savepath+"/test_numeric_feature.csv")
    single_model.run()


def combine_with_df(modals, root, selectors, classifiers):
    #这里加入临床表也可以，不过是组学特征直接拼接临床表的形式
    for l, modality in enumerate(modals):
        # train_df_path = os.path.join(root, modality, "original+log-sigma+wave", "best_model", "selected_train_data.csv")
        # test_df_path = os.path.join(root, modality, "original+log-sigma+wave", "best_model", "selected_test_data.csv")
        train_df_path = os.path.join(root, modality, "best_model", "selected_train_data.csv")
        test_df_path = os.path.join(root, modality, "best_model", "selected_test_data.csv")
        if l == 0:
            combine_train_df = pd.read_csv(train_df_path, index_col=0)
            combine_test_df = pd.read_csv(test_df_path, index_col=0)
        else:
            train_df = pd.read_csv(train_df_path, index_col=0).iloc[:, 1:]
            test_df = pd.read_csv(test_df_path, index_col=0).iloc[:, 1:]
            combine_train_df = pd.merge(combine_train_df, train_df, left_index=True, right_index=True, validate="one_to_one")
            combine_test_df = pd.merge(combine_test_df, test_df, left_index=True, right_index=True, validate="one_to_one")
            assert len(combine_train_df) == len(train_df), "wrong"
    store_path = os.path.join(root, "combine_radiomics")
    os.makedirs(store_path, exist_ok=True)
    single_model = Radiomics(selectors, classifiers)
    single_model.predict_save(combine_train_df, combine_test_df, store_path, True)




if __name__ == "__main__":
    split_and_merge(r"\\mega\syli\dataset\EC_all\all_frame\zheci", ["DWI", "T1CE", "T2"], \
                    r"\\mega\syli\dataset\EC_all\all_frame\zheci\clinical_data.csv", r"\\mega\syli\dataset\EC_all\model")
    #单模态
    path = "/homes/syli/dataset/EC_all/model/T1CE"
    a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA], [LR, SVM], path,
                  max_feature_num=10)
    # a.load_csv(r"D:\python\dataset\b1000\train_numeric_feature.csv", r"D:\python\dataset\b1000\test_numeric_feature.csv")
    train_df = pd.read_csv(path + "/original/train_data.csv", index_col=0)
    test_df = pd.read_csv(path + "/original/test_data.csv", index_col=0)
    a.predict_save(train_df, test_df, path, True)

