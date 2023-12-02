import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import pickle

def kmeans_test(file_name, output_folder="models"):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(file_name)

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    kmeans = KMeans(n_clusters=3, init='random', n_init=1)

    kmeans.fit(X_train, y_train)

    # Save the trained model using pickle
    model_filename = os.path.join(output_folder, f"KMeans_{os.path.splitext(os.path.basename(file_name))[0]}_model.pkl")
    with open(model_filename, 'wb') as model_file:
        pickle.dump(kmeans, model_file)

    y_pred = kmeans.predict(X_test)

    train_accuracy = kmeans.score(X_train, y_train)
    test_accuracy = kmeans.score(X_test, y_test)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"KMeans Model saved to: {model_filename}")

# Run K-means on all the files
file_names = ["features_mfcc_only.csv",
              "features_mfcc_only_noisegate.csv",
              "features_fulldata_not_normalized.csv",
              "features_fulldata_not_normalized_noisegate.csv",
              "features_fulldata_normalized.csv",
              "features_fulldata_normalized_noisegate.csv"]
