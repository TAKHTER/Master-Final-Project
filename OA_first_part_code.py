# Required Python Packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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

def knn_classifier(features, target):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features, target)
    return clf


def get_model_detail(trained_model, train_x, train_y, test_x, test_y):
    # for i in xrange(0, 5):
    #     print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

    predictions = trained_model.predict(test_x)
    return [accuracy_score(train_y, trained_model.predict(train_x)) , accuracy_score(test_y, predictions)]


def run_all_algorithm(test_dataset, train_dataset ):
    test_y = test_dataset[test_dataset.columns[0]]
    test_x = test_dataset[test_dataset.columns[1:]]
    train_y = train_dataset[train_dataset.columns[0]]
    train_x = train_dataset[train_dataset.columns[1:]]


    trained_model_random_forest = random_forest_classifier(train_x, train_y)
    rnd_accuracy = get_model_detail(trained_model_random_forest, train_x, train_y, test_x, test_y)

    trained_model_gbm = gradient_boosting_classifier(train_x, train_y)
    gbm_accuracy = get_model_detail(trained_model_gbm, train_x, train_y, test_x, test_y)

    trained_model_knn = knn_classifier(train_x, train_y)
    knn_accuracy = get_model_detail(trained_model_knn, train_x, train_y, test_x, test_y)

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
    print dataset.shape

    # Filter missing values
    # dataset = handel_missing_values(dataset, HEADERS[6], '?')
    run_sequence = [[0,1,2,3,], [1,0,2,3,], [2,0,1,3], [3,0,1,2]]

    for i in range (0, len(run_sequence)):
        test_dataset = dataset_split[run_sequence[i][0]]
        train_dataset = pd.concat([
            dataset_split[run_sequence[i][1]],
            dataset_split[run_sequence[i][2]],
            dataset_split[run_sequence[i][3]]
        ])
        print "\nRun Sequence: Test set# "+ str(run_sequence[i][0]) +",  Training Set# "+ (" ,".join(str(x) for x in run_sequence[i][1:]))
        data = run_all_algorithm(train_dataset, test_dataset)
        print data.to_string(index=False)

if __name__ == "__main__":
    main()
