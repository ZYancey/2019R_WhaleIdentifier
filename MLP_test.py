import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split

file_path = 'mfcc_features.csv'

whale_df = pd.read_csv(file_path)
#print(whale_df)

whale_np = whale_df.to_numpy()
#print(whale_np)


clf = MLPClassifier(hidden_layer_sizes = [64],
                    activation = 'logistic',
                    solver = 'sgd',
                    alpha = 0,
                    batch_size = 1,
                    learning_rate_init = 0.01,
                    shuffle = True,
                    momentum = 0,
                    n_iter_no_change = 10)


X = whale_df.drop(columns='species')
y = whale_df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
clf.fit(X_train, y_train)


print(clf.score(X_train, y_train)) # Training set accuracy
print(clf.score(X_test, y_test))  # Test set accuracy