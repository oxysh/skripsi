# Import libraries
import pandas as pd
from fs_extratrees import ExtraTreesFeatureSelection
from c_svm import SVMClassification
from c_extratrees import ExtraTreesClassification

# Read data
data = pd.read_excel("new-data.xlsx")

# Preprocessing
data.rename(columns={"T Stage ": "T Stage"}, inplace=True)
data["Race"].replace({"White": 1, "Black": 2, "Other": 3}, inplace=True)
data["Marital Status"].replace(
    {
        "Single ": 1, 
        "Married": 2, 
        "Separated": 3, 
        "Divorced": 4, 
        "Widowed": 5
    },
    inplace=True,
)
data["T Stage"].replace({"T1": 1, "T2": 2, "T3": 3, "T4": 4}, inplace=True)
data["N Stage"].replace({"N1": 1, "N2": 2, "N3": 3}, inplace=True)
data["6th Stage"].replace(
    {
        "IIA": 1, 
        "IIB": 2, 
        "IIIA": 3, 
        "IIIB": 4, 
        "IIIC": 5
    }, 
    inplace=True
)
data["differentiate"].replace(
    {
        "Well differentiated": 1,
        "Moderately differentiated": 2,
        "Poorly differentiated": 3,
        "Undifferentiated": 4,
    },
    inplace=True,
)
data["Grade"].replace({" anaplastic; Grade IV": 4}, inplace=True)
data["Grade"] = data["Grade"].astype(int)
data["A Stage"].replace({"Regional": 1, "Distant": 2}, inplace=True)
data["Estrogen Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
data["Progesterone Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
data["Status"].replace({"Alive": 1, "Dead": 0}, inplace=True)

data = pd.get_dummies(data,columns=['Race','Marital Status'], dtype=int, drop_first=True)

data_x = data.drop(["Status"], axis=1)
data_y = data["Status"]

# Feature selection
data_x_transformed = ExtraTreesFeatureSelection(data_x, data_y)

# KFold
folds = [4, 5, 10]

# SVM Kernel
kernels = ["linear", "poly", "rbf", "sigmoid"]

writer = pd.ExcelWriter("result.xlsx")
for i in range(2):
    print('\n--- Iterasi ' + str(i+1) + " ---")

    # SVM
    df_SVM = SVMClassification(data_x_transformed, data_y, folds, kernels)

    # Extra-Trees
    df_ExtraTrees = ExtraTreesClassification(data_x_transformed, data_y, folds)

    # Merge df
    df = pd.concat([df_SVM, df_ExtraTrees], axis=1)
    df = df.rename(index={
        0: "4 fold_accuracy",
        1: "4 fold_precision",
        2: "4 fold_specificity",
        3: "4 fold_recall",
        4: "4 fold_err_rate",
        5: "5 fold_accuracy",
        6: "5 fold_precision",
        7: "5 fold_specificity",
        8: "5 fold_recall",
        9: "5 fold_err_rate",
        10: "10 fold_accuracy",
        11: "10 fold_precision",
        12: "10 fold_specificity",
        13: "10 fold_recall",
        14: "10 fold_err_rate"
    })

    # Export to excel
    # with pd.ExcelWriter("result.xlsx") as writer:
    df.to_excel(writer, sheet_name="iterasi " + str(i+1))

writer.save()