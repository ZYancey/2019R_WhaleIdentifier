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


# Run all the models on the same train_test_split 

def run_all_models():

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

    # # kmeans 
    # kmeans_clf = KMeans(n_clusters=3, init='random', n_init=1)
    # models.append(kmeans_clf)
    
    # #HAC 
    # hac_clf = AgglomerativeClustering(n_clusters=3, linkage='complete')
    # models.append(hac_clf)

    # Ensembles 
    # random forest 
    random_forest_clf = RandomForestClassifier(n_estimators=100)
    models.append(random_forest_clf)

    # gradient boost 
    gradient_boost_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    models.append(gradient_boost_clf)

    data = []
    for filenames in os.listdir('Datasets'):
        name = 'Datasets/' + filenames
        whale_df = pd.read_csv(name)
        X = whale_df.drop(columns='species')
        y = whale_df[['species']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)
        
        model_names = ['Perceptron', 'MLP', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boost']
        for i in range(len(models)): 
            model = models[i]
            model.fit(X_train, y_train)

            model_filename = 'Models/' + model_names[i] + filenames.split('.')[0] + '.pkl'
            with open(model_filename, 'wb+') as model_file:
                s = pickle.dump(model, model_file)
                #pickle.loads(s)

            # get our train and test accuracy write to a csv file
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            # model, dataset, train_acc, test_acc
            row = [model_names[i], filenames.split('.')[0], train_acc, test_acc]
            data.append(row)

    data_df = pd.DataFrame(data, columns=['Model', 'Dataset', 'Training Accuracy', 'Test Accuracy'])
    data_df.to_csv("final_results.csv", index=False)



run_all_models()