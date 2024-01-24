# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import time
import os
import openpyxl
from fs_extratrees import ExtraTreesFeatureSelection
from classification import SVMClassification, ExtraTreesClassification

# save time start
start_time = time.time()

# Read data
data = pd.read_excel("new-data.xlsx")

# Preprocessing
data.rename(columns={"T Stage ": "T Stage"}, inplace=True)
data["T Stage"].replace({"T1": 0, "T2": 1, "T3": 2, "T4": 3}, inplace=True)
data["N Stage"].replace({"N1": 0, "N2": 1, "N3": 2}, inplace=True)
data["6th Stage"].replace(
    {"IIA": 0, "IIB": 1, "IIIA": 2, "IIIB": 3, "IIIC": 4}, inplace=True
)
data["differentiate"].replace(
    {
        "Well differentiated": 0,
        "Moderately differentiated": 1,
        "Poorly differentiated": 2,
        "Undifferentiated": 3,
    },
    inplace=True,
)
data["Grade"].replace({" anaplastic; Grade IV": 4}, inplace=True)
data["Grade"] = data["Grade"].astype(int)
data["A Stage"].replace({"Regional": 0, "Distant": 2}, inplace=True)
data["Estrogen Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
data["Progesterone Status"].replace({"Positive": 1, "Negative": 0}, inplace=True)
data["Status"].replace({"Alive": 1, "Dead": 0}, inplace=True)

data = pd.get_dummies(data, columns=["Race", "Marital Status"], dtype=int)

data_x = data.drop(["Status"], axis=1)
data_y = data["Status"]


# plot about status
def createStatusPlot(data_y, title, filename):
    data_y0 = np.count_nonzero(data_y.to_numpy() == 0)
    data_y1 = np.count_nonzero(data_y.to_numpy() == 1)

    fig, ax = plt.subplots(figsize=(6, 7))
    rects = ax.bar(["Dead", "Alive"], [data_y0, data_y1], color="#18ACA4", width=0.4)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.xlabel("Status")
    plt.ylabel("Jumlah")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# plot before handling-imbalance
createStatusPlot(
    data_y,
    "Jumlah subjek pada setiap kelas\nsebelum handling-imbalance",
    "graph/handling-imbalance_before.png",
)

# handling imbalance
smote = SMOTE(random_state=42)
data_x, data_y = smote.fit_resample(data_x, data_y)

# plot after handling-imbalance
createStatusPlot(
    data_y,
    "Jumlah subjek pada setiap kelas\nsetelah handling-imbalance",
    "graph/handling-imbalance_after.png",
)

# Feature selection
data_x = ExtraTreesFeatureSelection(data_x, data_y)

# classification
test_sizes = [0.1, 0.2, 0.3]
iterasi = 5

# SVM Kernel
kernels = ["linear", "poly", "rbf", "sigmoid"]
c = [1, 3, 5]
deg = [1, 3, 6]
result = pd.DataFrame()
avg_result = pd.DataFrame()
confusion_matrix = pd.DataFrame()

for test_size in test_sizes:
    # SVM
    for c_value in c:
        for kernel in kernels:
            if kernel == "poly":
                for deg_value in deg:
                    [res, avg, confmat] = SVMClassification(
                        data_x,
                        data_y,
                        test_size,
                        kernel=kernel,
                        c=c_value,
                        deg=deg_value,
                        iterasi=iterasi,
                    )
                    result = pd.concat([result, res], ignore_index=True)
                    avg_result = pd.concat([avg_result, avg], ignore_index=True)
                    confusion_matrix = pd.concat(
                        [confusion_matrix, confmat], ignore_index=True
                    )
            else:
                [res, avg, confmat] = SVMClassification(
                    data_x, data_y, test_size, kernel=kernel, c=c_value, iterasi=iterasi
                )
                result = pd.concat([result, res], ignore_index=True)
                avg_result = pd.concat([avg_result, avg], ignore_index=True)
                confusion_matrix = pd.concat(
                    [confusion_matrix, confmat], ignore_index=True
                )

    # Extra-Trees
    [res, avg, confmat] = ExtraTreesClassification(
        data_x, data_y, test_size, iterasi=iterasi
    )
    result = pd.concat([result, res], ignore_index=True)
    avg_result = pd.concat([avg_result, avg], ignore_index=True)
    confusion_matrix = pd.concat([confusion_matrix, confmat], ignore_index=True)


# Export to excel
# don't forget to change the sheet-name!!
filename = "result.xlsx"
sheetname = "none"
if not os.path.isfile(filename):
    openpyxl.Workbook().save(filename)

last_file = pd.ExcelFile(filename)
last_results = {}
for sheet in last_file.sheet_names:
    last_results[sheet] = last_file.parse(sheet)
last_results[sheetname + "-all"] = result
last_results[sheetname + "-avg"] = avg_result
last_results[sheetname + "-cm"] = confusion_matrix
with pd.ExcelWriter(filename) as writer:
    for sheet, last_result in last_results.items():
        last_result.to_excel(writer, sheet_name=sheet, index=False)


# print running time
print(f"--- runtime: {((time.time() - start_time)/60):.2f} minutes ---")
