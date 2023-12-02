import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import normalize


# Run all the models on the same train_test_split 

def run_all_models():

    file_path = 'features.csv'
    whale_df = pd.read_csv(file_path)
    X = whale_df.drop(columns='species')
    y = whale_df[['species']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

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

    
    # KNN 
    # decision tree 

    # Ensembles 
    # random forest 
    # gradient boost 

    

