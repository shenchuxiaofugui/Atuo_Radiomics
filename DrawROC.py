from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

color_list = sns.color_palette('deep') + sns.color_palette('bright')


def draw_roc_list(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
    # To Draw the ROC curve.
    # :param pred_list: The list of the prediction.
    # :param label_list: The list of the label.
    # :param name_list: The list of the legend name.
    # :param store_path: The store path. Support jpg and eps.
    # :return: None
    if not isinstance(pred_list, list):
        pred_list = [pred_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if not isinstance(name_list, list):
        name_list = [name_list]

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    CI_index = 0.95

    auc_list = []
    confidence_lower_list = []
    confidence_upper_list = []
    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        bootstrapped_scores = []

        np.random.seed(3)
        seed_index = np.random.randint(0, 65535, 1000)
        for seed in seed_index.tolist():
            np.random.seed(seed)
            pred_one_sample = np.random.choice(pred_list[index], size=len(pred_list[index]), replace=True)
            np.random.seed(seed)
            label_one_sample = np.random.choice(label_list[index], size=len(label_list[index]), replace=True)

            if len(np.unique(label_one_sample)) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(label_one_sample, pred_one_sample)
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        # std_auc = np.std(sorted_scores)
        # mean_auc = np.mean(sorted_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]

        auc_list.append(auc)
        confidence_lower_list.append(confidence_lower)
        confidence_upper_list.append(confidence_upper)

        name_list[index] = name_list[index] + (' AUC = {:.3f} (95% CI: {:.3f}-{:.3f})'.format(auc, confidence_lower,
                                                                                              confidence_upper))
        axes.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc, linewidth=3)

    axes.plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)

    axes.set_xlabel('1 - specificity', fontsize=15)
    axes.set_ylabel('Sensitivity', fontsize=15)
    # axes.set_title('Receiver operating characteristic curve')
    # axes.set_title('T1CE+T2WI radiomics features + clinical characteristics', fontsize=15)
    # axes.set_title('Clinical characteristics', fontsize=15)
    axes.set_title('ROC', fontsize=15)
    axes.legend(name_list, loc="lower right", fontsize=12)
    if store_path != '':
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        fig.set_tight_layout(True)
        fig.savefig(os.path.join(store_path, 'ROC.jpg'), dpi=300, format='jpeg')

    if is_show:
        plt.show()
    return auc_list, confidence_lower_list, confidence_upper_list


if __name__ == '__main__':
    pass
