import pandas as pd
from sklearn.linear_model import Perceptron 
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import normalize

file_path = 'mfcc_features.csv'

whale_df = pd.read_csv(file_path)
#print(whale_df)

whale_np = whale_df.to_numpy()
#print(whale_np)


def doit(X, y, clf, testSize = 0.8, numberTrials = 5, n_jobs = -1):

    num_trials = numberTrials
    train_accuracy = 0
    test_accuracy = 0
    
    
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = splitter(X, y, test_size=testSize)
        
        clf.fit(X_train, y_train)
    
        train_accuracy = train_accuracy+ clf.score(X_train, y_train)
        test_accuracy = test_accuracy + clf.score(X_test, y_test)
    
    train_accuracy = train_accuracy/num_trials
    test_accuracy = test_accuracy/num_trials
    return train_accuracy, test_accuracy
  




X = whale_np[:, 1:]
y = whale_np[:, 0]
X_norm = normalize(X, axis=0)

args = ['l1', 'l2', 'elasticnet']
result_train_accuracies = []
result_test_accuracies = []
for a in args:
    clf = Perceptron(penalty = a)
    train_acc, test_acc = doit(X, y, clf, numberTrials = 500)
    print("a: {}, train: {}, test: {}".format(a, train_acc, test_acc))

# things that don't matter: alpha
# things that do matter: normalization (makes it worse)