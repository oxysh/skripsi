# Import libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import sklearn.metrics as matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def SVMClassification(data_x, data_y, folds, kernels, output_graph = "graph/SVM/") :  
    result = []
    
    if not os.path.isdir(output_graph):
        os.makedirs(output_graph)

    for kernel in kernels:
        subresult = []
        colnames = []
        for n_splits in folds:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

            confmat = []
            accuracy = []
            precision = []
            specificity = []
            recall = []
            err_rate = []

            for i, (train_index, test_index) in enumerate(kfold.split(data_x)):
                # classification -> SVM
                # C=5 -> akurasi yg lbih tinggi; margin lbih kecil. C=1 -> akurasi lbih rendah; margin lbih besar
                svm = SVC(kernel=kernel, C=5)
                model = svm.fit(data_x.iloc[train_index, :], data_y.iloc[train_index])
                ypred = svm.predict(data_x.iloc[test_index, :])

                # evaluation
                cm = matrix.confusion_matrix(data_y.iloc[test_index], ypred)
                confmat.append(cm)
                accuracy.append(matrix.accuracy_score(data_y.iloc[test_index], ypred))
                precision.append(matrix.precision_score(data_y.iloc[test_index], ypred))
                recall.append(matrix.recall_score(data_y.iloc[test_index], ypred))
                specificity.append(cm[0][0] / (cm[0][0] + cm[0][1]))
                err_rate.append(1 - matrix.accuracy_score(data_y.iloc[test_index], ypred))

                # plot the confmat
                indices = np.argsort(cm.sum(axis=1))[::-1]
                cm_sorted = cm[indices, :][:, indices]
                labels_sorted = ['Positif', 'Negatif']
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_sorted, display_labels=labels_sorted)
                disp.plot(cmap='Blues')
                # plt.show()
                plt.savefig(output_graph + kernel + "_" + str(n_splits) + " folds_" + str(i) + ".png")

            # Average Evaluation
            avg_accuracy = round(sum(accuracy) / len(accuracy) * 100, 2)
            avg_precision = round(sum(precision) / len(precision) * 100, 2)
            avg_specificity = round(sum(specificity) / len(specificity) * 100, 2)
            avg_recall = round(sum(recall) / len(recall) * 100, 2)
            avg_err_rate = round(sum(err_rate) / len(err_rate) * 100, 2)

            print("Selesai running SVM kernel " + kernel + " dengan " + str(n_splits) + " folds")
            # print("\nRata-rata -- SVM kernel " + kernel + " dengan " + str(n_splits) + " folds:")
            # print("Accuracy =", avg_accuracy, "%")
            # print("Precision =", avg_precision, "%")
            # print("Specificity =", avg_specificity, "%")
            # print("Recall =", avg_recall, "%")
            # print("Error rate =", avg_err_rate, "%")

            subresult.append(avg_accuracy)
            subresult.append(avg_precision)
            subresult.append(avg_specificity)
            subresult.append(avg_recall)
            subresult.append(avg_err_rate)
        result.append(subresult)
        colnames.append("svm_" + kernel)

    result = np.transpose(result)
    df = pd.DataFrame(result, columns=kernels)
    return df
