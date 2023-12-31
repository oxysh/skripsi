# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from fs_extratrees import ExtraTreesFeatureSelection
from c_svm import SVMClassification
from c_extratrees import ExtraTreesClassification

# Read data
data = pd.read_excel("new-data.xlsx")

# Preprocessing
data.rename(columns={"T Stage ": "T Stage"}, inplace=True)
data["Race"].replace({"White": 1, "Black": 2, "Other": 3}, inplace=True)
data["Marital Status"].replace(
    {"Single ": 1, "Married": 2, "Separated": 3, "Divorced": 4, "Widowed": 5},
    inplace=True,
)
data["T Stage"].replace({"T1": 1, "T2": 2, "T3": 3, "T4": 4}, inplace=True)
data["N Stage"].replace({"N1": 1, "N2": 2, "N3": 3}, inplace=True)
data["6th Stage"].replace(
    {"IIA": 1, "IIB": 2, "IIIA": 3, "IIIB": 4, "IIIC": 5}, inplace=True
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

data = pd.get_dummies(
    data, columns=["Race", "Marital Status"], dtype=int
)

data_x = data.drop(["Status"], axis=1)
data_y = data["Status"]

# plot about status
def createStatusPlot(data_y, title, filename) :
    data_y0 = np.count_nonzero(data_y.to_numpy() == 0)
    data_y1 = np.count_nonzero(data_y.to_numpy() == 1)

    fig, ax = plt.subplots(figsize=(6, 7))
    rects = ax.bar(
        ['Dead', 'Alive'], 
        [data_y0, data_y1], 
        color="#18ACA4", 
        width=.4
    )
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{int(height)}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.xlabel('Status')
    plt.ylabel('Jumlah')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# plot before handling-imbalance
createStatusPlot(
    data_y, 
    'Jumlah subjek pada setiap kelas\nsebelum handling-imbalance',
    'graph/handling-imbalance_before.png'
)

# handling imbalance
smote = SMOTE(random_state=42)
data_x, data_y = smote.fit_resample(data_x, data_y)

# plot after handling-imbalance
createStatusPlot(
    data_y, 
    'Jumlah subjek pada setiap kelas\nsetelah handling-imbalance',
    'graph/handling-imbalance_after.png'
)

# Feature selection
data_x_transformed = ExtraTreesFeatureSelection(data_x, data_y)
# data_x_transformed = data_x

# KFold
folds = [4, 5, 10]

# SVM Kernel
kernels = ["linear", "poly", "rbf", "sigmoid"]

result = pd.DataFrame()
for i in range(1):
    print("\n--- Iterasi " + str(i + 1) + " ---")

    output_graph="graph/iterasi "+ str(i+1) +"/"

    # SVM
    df_SVM = SVMClassification(data_x_transformed, data_y, folds, kernels, output_graph+"SVM/")

    # Extra-Trees
    df_ExtraTrees = ExtraTreesClassification(data_x_transformed, data_y, folds, output_graph+"Extra-trees/")

    # Merge df
    # df = df_SVM.append(df_ExtraTrees, ignore_index=True)
    df = pd.concat([df_SVM, df_ExtraTrees])

    # Iteration
    df["Iterasi"] = i + 1

    result = pd.concat([result, df])

# Export to excel
writer = pd.ExcelWriter("result.xlsx")
result.to_excel(writer, sheet_name="main")

# REVISIT:
# need to count evaluation mean of ever fold+classification automatically in python

writer.close()
