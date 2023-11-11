import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = 'mfcc_features.csv'

whale_df = pd.read_csv(file_path)

# Just a basic random forest classifier on all species in the df
X = whale_df.drop(columns='species')
y = whale_df[['species']]

clf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # 0.36

# Filtering the data down to two types of whales
filtered_df = whale_df[whale_df['species'].isin(['atlantic_spotted_dolphin', 'whitesided_dolphin'])]
X = filtered_df.drop(columns='species')
y = filtered_df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # 0.74


# Filtering the data down to two types of whales
filtered_df = whale_df[whale_df['species'].isin(['humpback_whale', 'killer_whale'])]
X = filtered_df.drop(columns='species')
y = filtered_df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # 0.88

# Try classifiying whales vs dolphines. Assign all whales to class 1 and all dolphines to class 0. 
whale_df['species'] = whale_df['species'].str.contains('whale').astype(int)
X = whale_df.drop(columns='species')
y = whale_df[['species']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # 0.65, 0.69
