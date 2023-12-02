import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def decision_tree_test(file_name, output_folder="models"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(file_name)

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(criterion="entropy",
                                 max_depth=None,
                                 min_impurity_decrease=0.025,
                                 class_weight="balanced")

    clf.fit(X_train, y_train)

    # Save the trained model using pickle
    model_filename = os.path.join(output_folder, f"DT_{os.path.splitext(os.path.basename(file_name))[0]}_model.pkl")
    with open(model_filename, 'wb') as model_file:
        s = pickle.dump(clf, model_file)
        pickle.loads(s)

    y_pred = clf.predict(X_test)

    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Model saved to: {model_filename}")

# Run Decision Tree on all the files
file_names = ["features_mfcc_only.csv",
              "features_mfcc_only_noisegate.csv",
              "features_fulldata_not_normalized.csv",
              "features_fulldata_not_normalized_noisegate.csv",
              "features_fulldata_normalized.csv",
              "features_fulldata_normalized_noisegate.csv"]

for file_name in file_names:
    print(f"File: {file_name}")
    decision_tree_test(file_name)
    print("\n")

# Best accuracies was with 'features_fulldata_normalized.csv'
# Test Accuracy: 65.1367%
# Test Accuracy: 56.2642%

