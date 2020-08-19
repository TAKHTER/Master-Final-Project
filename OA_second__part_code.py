# Required Python Packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import cross_val_score


import matplotlib
import matplotlib_venn as venn
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles


#
# SMALL_SIZE = 18
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# File Paths
INPUT_PATH = "OA_data_shuffled.csv"


def random_forest_classifier(features, target):
    clf = RandomForestClassifier(random_state=1234)
    clf.fit(features, target)
    return clf

def gradient_boosting_classifier(features, target):
    clf = GradientBoostingClassifier()
    clf.fit(features, target)
    return clf

def knn_mostImportantFeatures(features, target):
    clf = KNeighborsClassifier(n_neighbors=3)
    features_array = features.values

    n_feats = features.shape[1]
    importances = np.zeros(n_feats)

    for i in range(n_feats):
        X = features_array[:, i].reshape(-1, 1)
        scores = cross_val_score(clf, X, target)
        importances[i] =  scores.mean()

    return importances

def knn_classifier(features, target):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features, target)
    knn_mostImportantFeatures(features, target)
    return clf

def plot_most_important_feature(title, columns, importances):
    plt.figure()
    plt.title(title)

    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(columns)), importances[indices], color="r", align="center")
    plt.xticks(range(len(columns)), columns[indices])
    plt.xlim([-1, 10.5])
    plt.show()


def get_model_detail(trained_model, train_x, train_y, test_x, test_y):
    # for i in xrange(0, 5):
    #     print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

    predictions = trained_model.predict(test_x)
    return [accuracy_score(train_y, trained_model.predict(train_x)) , accuracy_score(test_y, predictions)]


def plot_venn_diagram(rnd_clf, gbm_clf, knn_clf, data, column):

    rnd_y = rnd_clf.predict(data)
    gbm_y = gbm_clf.predict(data)
    knn_y = knn_clf.predict(data)

    rnd_y0 = [i for i, e in enumerate(rnd_y) if e == 0]
    rnd_y1 = [i for i, e in enumerate(rnd_y) if e == 1]
    gbm_y0 = [i for i, e in enumerate(gbm_y) if e == 0]
    gbm_y1 = [i for i, e in enumerate(gbm_y) if e == 1]
    knn_y0 = [i for i, e in enumerate(knn_y) if e == 0]
    knn_y1 = [i for i, e in enumerate(knn_y) if e == 1]

    #oa == 0
    rnd_y0_set = set(data[column].values[rnd_y0])
    gbm_y0_set = set(data[column].values[gbm_y0])
    knn_y0_set = set(data[column].values[knn_y0])

    figure, axes = plt.subplots(1, 2)
    venn3([rnd_y0_set, gbm_y0_set,knn_y0_set], set_labels = ('Random Forest', 'GBM', "KNN"), ax=axes[0])
    axes[0].set_title(column + ' value distribution when oa=0')


    rnd_y1_set = set(data[column].values[rnd_y1])
    gbm_y1_set = set(data[column].values[gbm_y1])
    knn_y1_set = set(data[column].values[knn_y1])

    venn3([rnd_y1_set, gbm_y1_set, knn_y1_set], set_labels = ('Random Forest', 'GBM', "KNN"), ax=axes[1])
    axes[1].set_title(column + ' value distribution when oa=1')

    plt.show()


def run_all_algorithm( train_dataset, test_dataset ):
    test_y = test_dataset[test_dataset.columns[0]]
    test_x = test_dataset[test_dataset.columns[1:]]
    train_y = train_dataset[train_dataset.columns[0]]
    train_x = train_dataset[train_dataset.columns[1:]]


    trained_model_random_forest = random_forest_classifier(train_x, train_y)
    rnd_accuracy = get_model_detail(trained_model_random_forest, train_x, train_y, test_x, test_y)
    importances = trained_model_random_forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in trained_model_random_forest.estimators_], axis=0)
    plot_most_important_feature("Random Forest Feature importances", train_x.columns.values, importances)

    trained_model_gbm = gradient_boosting_classifier(train_x, train_y)
    gbm_accuracy = get_model_detail(trained_model_gbm, train_x, train_y, test_x, test_y)
    importances_gbm = trained_model_gbm.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in trained_model_random_forest.estimators_], axis=0)
    plot_most_important_feature("Gradient Boosting Feature importances", train_x.columns.values, importances_gbm)


    trained_model_knn = knn_classifier(train_x, train_y)
    knn_accuracy = get_model_detail(trained_model_knn, train_x, train_y, test_x, test_y)
    importances_knn = knn_mostImportantFeatures(train_x, train_y)
    plot_most_important_feature("KNN Acuracy by Features", train_x.columns.values, importances_knn)


    plot_venn_diagram(trained_model_random_forest, trained_model_gbm, trained_model_knn, test_x, "Arg")
    plot_venn_diagram(trained_model_random_forest, trained_model_gbm, trained_model_knn, test_x, "Orn")
    plot_venn_diagram(trained_model_random_forest, trained_model_gbm, trained_model_knn, test_x, "Ac.Orn")


    data = {
        'Algorithim Name' : ["Random Forest", "Gradient Boosting", "KNN"],
        'Training Accuracy': [round(x*100,2) for x in [rnd_accuracy[0], gbm_accuracy[0], knn_accuracy[0]]],
        'Test Accuracy': [round(x*100,2) for x in [rnd_accuracy[1], gbm_accuracy[1], knn_accuracy[1]]]
    }
    return pd.DataFrame(data=data)

def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(INPUT_PATH, header=0)
    dataset_split = np.array_split(dataset, 4)

    #print dataset_split[2].head(n=5)

    # Get basic statistics of the loaded dataset

    # Filter missing values
    # dataset = handel_missing_values(dataset, HEADERS[6], '?')

    test_dataset = dataset_split[0]
    train_dataset = pd.concat([
        dataset_split[1],
        dataset_split[2],
        dataset_split[3]
    ])
    data = run_all_algorithm(train_dataset, test_dataset)
    print data.to_string(index=False)

if __name__ == "__main__":
    main()
