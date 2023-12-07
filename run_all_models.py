import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split 
import pickle
import os

def define_all_models():
    # define all models 
    models = []
    
    # simple perceptron
    perceptron_clf = Perceptron()
    models.append(perceptron_clf)

    # MLP
    MLP_clf = MLPClassifier(hidden_layer_sizes = [64],
                    activation = 'logistic',
                    solver = 'sgd',
                    alpha = 0,
                    batch_size = 1,
                    learning_rate_init = 0.01,
                    shuffle = True,
                    momentum = 0,
                    n_iter_no_change = 10)
    models.append(MLP_clf)

    # SMV 
    svm_clf = SVC(kernel='rbf',
              gamma='auto',
              shrinking=True)
    models.append(svm_clf)

    # KNN 
    knn_clf = KNeighborsClassifier(n_neighbors=14, weights="distance", p=1)
    models.append(knn_clf)

    # decision tree 
    decision_tree_clf = DecisionTreeClassifier(criterion="entropy",
                                 max_depth=None,
                                 min_impurity_decrease=0.025,
                                 class_weight="balanced")
    models.append(decision_tree_clf)

    # random forest (ensemble)
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    models.append(random_forest_clf)

    # gradient boost (ensemble)
    gradient_boost_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    models.append(gradient_boost_clf)

    return models 

# Run all the models on the same train_test_split 
#cetacean_type = ('both', 'whale_vs_dolphins', 'only_whales', 'only_dolphins')
def run_all_models(cetacean_type = 'both'):

    models = define_all_models()

    data = []
    for filenames in os.listdir('Datasets'):
        # cetacean_type = ('both', 'whale_vs_dolphins', 'only_whales', 'only_dolphins')
        if cetacean_type == 'both':
            model_folder = 'Models_both/'
        elif cetacean_type == 'whale_vs_dolphins':
            model_folder = 'Models_whale_vs_dolphins'
        elif cetacean_type == 'only_whales':
            model_folder = 'Models_only_whales/'
        elif cetacean_type == 'only_dolphins':
            model_folder = 'Models_only_dolphins/'

        name = 'Datasets/' + filenames
        whale_df = pd.read_csv(name)
        if cetacean_type == 'only_whales':
            # Limit the dataset to just whales, exclude dolphins 
            all_whales = []
            for whale in whale_df['species'].unique():
                if "whal" in whale:
                    all_whales.append(whale)
            whale_df = whale_df[whale_df['species'].isin(all_whales)]
            print(all_whales)
        elif cetacean_type == 'only_dolphins':
            all_dolphins = []
            for cetacean in whale_df['species'].unique():
                if not "whal" in cetacean:
                    all_dolphins.append(cetacean)
            whale_df = whale_df[whale_df['species'].isin(all_dolphins)]
        elif cetacean_type == 'whale_vs_dolphins':
            # Assign all whales to class 1 and all dolphines to class 0. 
            whale_df['species'] = whale_df['species'].str.contains('whale').astype(int)

        X = whale_df.drop(columns='species')
        y = whale_df[['species']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)
        
        model_names = ['Perceptron', 'MLP', 'SVM', 'KNN', 'Decision_Tree', 'Random_Forest', 'Gradient_Boost']
        for i in range(len(models)): 
            model = models[i]
            model.fit(X_train, y_train)

            model_filename = 'Models/' + model_folder + model_names[i] + "_" + filenames.split('.')[0] + '.pkl'
            with open(model_filename, 'wb+') as model_file:
                s = pickle.dump(model, model_file)

            # get our train and test accuracy write to a csv file
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            # model, dataset, train_acc, test_acc
            row = [model_names[i], filenames.split('.')[0], train_acc, test_acc]
            data.append(row)

    data_df = pd.DataFrame(data, columns=['Model', 'Dataset', 'Training_Accuracy', 'Test_Accuracy'])
    # cetacean_type = ('both', 'whale_vs_dolphins', 'only_whales', 'only_dolphins')
    if cetacean_type == 'both':
        results_file_name = "final_results/final_results_both.csv"
    elif cetacean_type == 'whale_vs_dolphins':
        results_file_name = "final_results/final_results_whale_vs_dolphins.csv"
    elif cetacean_type == 'only_whales':
        results_file_name = "final_results/final_results_only_whales.csv"
    elif cetacean_type == 'only_dolphins':
        results_file_name = "final_results/final_results_only_dolphins.csv"

    data_df.to_csv(results_file_name, index=False)


run_all_models(cetacean_type = 'both')
run_all_models(cetacean_type = 'whale_vs_dolphins')
run_all_models(cetacean_type = 'only_whales')
run_all_models(cetacean_type = 'only_dolphins')