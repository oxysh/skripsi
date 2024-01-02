# Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import time


def ExtraTreesFeatureSelection(data_x, data_y, output_graph="graph/"):
    if not os.path.isdir(output_graph):
        os.makedirs(output_graph)

    max_features = int(round(np.sqrt(data_x.shape[1])))
    print("Jumlah fitur yang digunakan:", max_features)
    extraTrees = ExtraTreesClassifier(
        n_estimators=150, random_state=0, max_features=max_features
    )
    extraTrees.fit(data_x, data_y)
    # print("feature importance", extraTrees.feature_importances_)
    # print("feature importance", extraTrees.get_support())
    # show bar - feature importances
    plt.figure(figsize=(18, 8))
    plt.barh(data_x.columns, extraTrees.feature_importances_, color="#18ACA4")
    plt.xlabel('Feature importances')
    plt.ylabel('Atribut')
    plt.title('Feature importances dari setiap atribut')
    plt.savefig(output_graph + "feature_importances.png")

    feature = SelectFromModel(extraTrees, max_features=max_features)
    feature.fit(data_x, data_y)
    idx = feature.get_support()
    x = 'feature: '
    for i in range(len(idx)):
        if idx[i] == True:
            x = x + data_x.columns[i] + ', '
    x = x[0:-2] + '.'
    print(x)

    data_x_transformed = pd.DataFrame(feature.transform(data_x))

    print("Seleksi fitur dengan Extra-Trees telah selesai.")
    return data_x_transformed
