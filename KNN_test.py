import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import normalize

file_path = 'mfcc_features.csv'

whale_df = pd.read_csv(file_path)
#print(whale_df)

whale_np = whale_df.to_numpy()
#print(whale_np)


def doit(X, y, neigh, testSize = 0.8, numberTrials = 5):

    num_trials = numberTrials
    train_accuracy = 0
    test_accuracy = 0
    
    
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = splitter(X, y, test_size=testSize)
        
        neigh.fit(X_train, y_train)
    
        train_accuracy = train_accuracy+ neigh.score(X_train, y_train)
        test_accuracy = test_accuracy + neigh.score(X_test, y_test)
    
    train_accuracy = train_accuracy/num_trials
    test_accuracy = test_accuracy/num_trials
    return train_accuracy, test_accuracy, neigh.predict_proba(X_test)
  

neigh = KNeighborsClassifier(n_neighbors=3)


X = whale_np[:, 1:]
y = whale_np[:, 0]
X_norm = normalize(X, axis=0)


train_acc, test_acc, last_prob = doit(X, y, neigh, numberTrials = 100)

print("train accuracy = " + str(train_acc) + ", test accuracy = " + str(test_acc))
# print(last_prob)