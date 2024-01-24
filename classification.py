# Import libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics as matrix


def SVMClassification(
    data_x, data_y, test_size, kernel="linear", c=1, deg=3, iterasi=1
):
    if kernel == "poly":
        title = "SVM C=" + str(c) + " - " + kernel + " deg=" + str(deg)
    else:
        title = "SVM C=" + str(c) + " - " + kernel

    print("Running " + title + "...")

    confmat = []
    accuracy = []
    precision = []
    specificity = []
    recall = []
    err = []

    for i in range(iterasi):
        # print("iterasi " + str(i + 1) + "...")

        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=0.3
        )

        svm = SVC(kernel=kernel, C=c, degree=deg)
        model = svm.fit(x_train, y_train)
        ypred = svm.predict(x_test)

        # evaluation
        cm = np.array(matrix.confusion_matrix(y_test, ypred))
        confmat.append(cm.tolist())
        accuracy.append(round(matrix.accuracy_score(y_test, ypred), 2))
        precision.append(round(matrix.precision_score(y_test, ypred), 2))
        recall.append(round(matrix.recall_score(y_test, ypred), 2))
        specificity.append(round(cm[0][0] / (cm[0][0] + cm[0][1]), 2))
        err.append(
            round(
                (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[1][0] + cm[0][1] + cm[0][0]), 2
            )
        )

    colnames = [
        "Test Size",
        "Algoritma",
        "Iterasi",
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Error",
    ]
    result = []
    result.append([test_size] * iterasi)
    result.append([title] * iterasi)
    result.append(list(range(1, iterasi + 1)))
    result.append(accuracy)
    result.append(precision)
    result.append(specificity)
    result.append(recall)
    result.append(err)
    result = np.transpose(result)
    result = pd.DataFrame(result, columns=colnames)

    confusion_matrix = []
    cm_colnames = ["Algoritma", "Iterasi", "", "0", "1"]
    for i in range(len(confmat)):
        for j in range(len(confmat[i])):
            row = []
            row.append(title)
            row.append(i + 1)
            row.append(j)
            row.append(confmat[i][j][0])
            row.append(confmat[i][j][1])
            confusion_matrix.append(row)
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=cm_colnames)

    # Average Evaluation
    avg_accuracy = round(sum(accuracy) / len(accuracy) * 100, 2)
    avg_precision = round(sum(precision) / len(precision) * 100, 2)
    avg_specificity = round(sum(specificity) / len(specificity) * 100, 2)
    avg_recall = round(sum(recall) / len(recall) * 100, 2)
    avg_err = round(sum(err) / len(err) * 100, 2)

    # avg_confmat = np.empty((2, 2), int)
    # for i in range(2):
    #     for j in range(2):
    #         sum_confmat = 0
    #         for k in range(len(confmat)):
    #             sum_confmat = sum_confmat + confmat[k][j][i]
    #         avg_confmat[j][i] = round(sum_confmat / len(confmat), 0)

    avg_colnames = [
        "Test Size",
        "Algoritma",
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Error",
    ]
    avg_result = []
    avg_record = []
    avg_record.append(test_size)
    avg_record.append(title)
    avg_record.append(avg_accuracy)
    avg_record.append(avg_precision)
    avg_record.append(avg_specificity)
    avg_record.append(avg_recall)
    avg_record.append(avg_err)
    avg_result.append(avg_record)
    avg_result = pd.DataFrame(avg_result, columns=avg_colnames)

    return result, avg_result, confusion_matrix


def ExtraTreesClassification(data_x, data_y, test_size, iterasi=1):
    title = "Extra-trees"
    print("Running " + title + "...")

    confmat = []
    accuracy = []
    precision = []
    specificity = []
    recall = []
    err = []

    for i in range(iterasi):
        # print("iterasi " + str(i + 1) + "...")

        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=0.3
        )

        et = ExtraTreesClassifier()
        model = et.fit(x_train, y_train)
        ypred = et.predict(x_test)

        # evaluation
        cm = np.array(matrix.confusion_matrix(y_test, ypred))
        confmat.append(cm.tolist())
        accuracy.append(round(matrix.accuracy_score(y_test, ypred), 2))
        precision.append(round(matrix.precision_score(y_test, ypred), 2))
        recall.append(round(matrix.recall_score(y_test, ypred), 2))
        specificity.append(round(cm[0][0] / (cm[0][0] + cm[0][1]), 2))
        err.append(
            round(
                (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[1][0] + cm[0][1] + cm[0][0]), 2
            )
        )

    colnames = [
        "Test Size",
        "Algoritma",
        "Iterasi",
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Error",
    ]
    result = []
    result.append([test_size] * iterasi)
    result.append([title] * iterasi)
    result.append(list(range(1, iterasi + 1)))
    result.append(accuracy)
    result.append(precision)
    result.append(specificity)
    result.append(recall)
    result.append(err)
    result = np.transpose(result)
    result = pd.DataFrame(result, columns=colnames)

    confusion_matrix = []
    cm_colnames = ["Algoritma", "Iterasi", "", "0", "1"]
    for i in range(len(confmat)):
        for j in range(len(confmat[i])):
            row = []
            row.append(title)
            row.append(i + 1)
            row.append(j)
            row.append(confmat[i][j][0])
            row.append(confmat[i][j][1])
            confusion_matrix.append(row)
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=cm_colnames)

    # Average Evaluation
    avg_accuracy = round(sum(accuracy) / len(accuracy) * 100, 2)
    avg_precision = round(sum(precision) / len(precision) * 100, 2)
    avg_specificity = round(sum(specificity) / len(specificity) * 100, 2)
    avg_recall = round(sum(recall) / len(recall) * 100, 2)
    avg_err = round(sum(err) / len(err) * 100, 2)

    # avg_confmat = np.empty((2, 2), int)
    # for i in range(2):
    #     for j in range(2):
    #         sum_confmat = 0
    #         for k in range(len(confmat)):
    #             sum_confmat = sum_confmat + confmat[k][j][i]
    #         avg_confmat[j][i] = round(sum_confmat / len(confmat), 0)

    avg_colnames = [
        "Test Size",
        "Algoritma",
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Error",
    ]
    avg_result = []
    avg_record = []
    avg_record.append(test_size)
    avg_record.append(title)
    avg_record.append(avg_accuracy)
    avg_record.append(avg_precision)
    avg_record.append(avg_specificity)
    avg_record.append(avg_recall)
    avg_record.append(avg_err)
    avg_result.append(avg_record)
    avg_result = pd.DataFrame(avg_result, columns=avg_colnames)

    return result, avg_result, confusion_matrix
