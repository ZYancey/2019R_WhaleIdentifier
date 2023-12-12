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


def svm_test(file_name, output_folder="models"):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(file_name)

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svm = SVC(kernel='rbf',
              gamma='auto',
              shrinking=True)

    svm.fit(X_train, y_train)

    # Save the trained model using pickle
    model_filename = os.path.join(output_folder, f"SVM_{os.path.splitext(os.path.basename(file_name))[0]}_model.pkl")
    with open(model_filename, 'wb') as model_file:
        pickle.dump(svm, model_file)

    y_pred = svm.predict(X_test)

    train_accuracy = svm.score(X_train, y_train)
    test_accuracy = svm.score(X_test, y_test)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"SVM Model saved to: {model_filename}")

# Run SVM on all the files
file_names = ["features_mfcc_only.csv",
              "features_mfcc_only_noisegate.csv",
              "features_fulldata_not_normalized.csv",
              "features_fulldata_not_normalized_noisegate.csv",
              "features_fulldata_normalized.csv",
              "features_fulldata_normalized_noisegate.csv"]

for file_name in file_names:
    print(f"File: {file_name}")
    svm_test(file_name)
    print("\n")