# Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics as matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def ExtraTreesClassification(data_x, data_y, folds, output_graph = "graph/Extra-trees/") : 
    result = []

    if not os.path.isdir(output_graph):
        os.makedirs(output_graph)

    for n_splits in folds:
        kfold = KFold(n_splits=n_splits, shuffle=True)

        confmat = []
        accuracy = []
        precision = []
        specificity = []
        recall = []
        err_rate = []

        for i, (train_index, test_index) in enumerate(kfold.split(data_x)):
            # classification
            classification = ExtraTreesClassifier()
            model = classification.fit(
                data_x.iloc[train_index, :], data_y.iloc[train_index])
            ypred = classification.predict(data_x.iloc[test_index, :])

            # evaluation
            cm = matrix.confusion_matrix(data_y.iloc[test_index], ypred)
            confmat.append(cm)
            accuracy.append(matrix.accuracy_score(data_y.iloc[test_index], ypred))
            precision.append(matrix.precision_score(data_y.iloc[test_index], ypred))
            recall.append(matrix.recall_score(data_y.iloc[test_index], ypred))
            specificity.append(cm[0][0] / (cm[0][0] + cm[0][1]))
            err_rate.append(1-matrix.accuracy_score(data_y.iloc[test_index], ypred))

            # plot the confmat
            indices = np.argsort(cm.sum(axis=1))[::-1]
            cm_sorted = cm[indices, :][:, indices]
            labels_sorted = ['Positif', 'Negatif']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_sorted, display_labels=labels_sorted)
            disp.plot(cmap='Blues')
            # plt.show()
            plt.savefig(output_graph + str(n_splits) + " folds_" + str(i) + ".png")
            plt.clf()

        # Average Evaluation
        avg_accuracy = round(sum(accuracy) / len(accuracy) * 100, 2)
        avg_precision = round(sum(precision) / len(precision) * 100, 2)
        avg_specificity = round(sum(specificity) / len(specificity) * 100, 2)
        avg_recall = round(sum(recall) / len(recall) * 100, 2)
        avg_err_rate = round(sum(err_rate) / len(err_rate) * 100, 2)
        
        print("Selesai running Extra-Trees dengan " + str(n_splits) + " folds.")
        # print("\nRata-rata -- Extra-Trees dengan " + str(n_splits) + " folds:")
        # print("Accuracy =", avg_accuracy, "%")
        # print("Precision =", avg_precision, "%")
        # print("Specificity =", avg_specificity, "%")
        # print("Recall =", avg_recall, "%")
        # print("Error rate =", avg_err_rate, "%")

        result.append(avg_accuracy)
        result.append(avg_precision)
        result.append(avg_specificity)
        result.append(avg_recall)
        result.append(avg_err_rate)

    result = np.transpose(result)
    df = pd.DataFrame(result, columns=["Extra Trees"])
    return df
