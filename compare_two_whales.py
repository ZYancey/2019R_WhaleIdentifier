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

# Limit the dataset to just whale exclude dolphins 
all_whales = []
for whale in whale_df['species'].unique():
    if "whal" in whale:
        all_whales.append(whale)
    # else:
    #     print(whale)

whale_df = whale_df[whale_df['species'].isin(all_whales)]
all_whale_species = whale_df['species'].unique()
#print(all_whale_species)

#### Create a dataframe with classification between every two types of whales ###
clf = RandomForestClassifier(n_estimators=100, random_state=42)

print(f"The total number of whale species is {len(all_whale_species)}")
data = []
for whale_type in all_whale_species:
    for other_whale_type in all_whale_species:
        if whale_type != other_whale_type:
            #print(f"First whale is: {whale_type}. Second whale is {other_whale_type}")
            row = []
            row.append(whale_type)
            row.append(other_whale_type)

            filtered_df = whale_df[whale_df['species'].isin([whale_type, other_whale_type])]
            X = filtered_df.drop(columns='species')
            y = filtered_df[['species']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
            clf.fit(X_train, np.ravel(y_train))

            y_pred = clf.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_pred)
            row.append(train_accuracy) # train set accuracy

            y_pred = clf.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            row.append(test_accuracy) # test set accuracy

            row.append(train_accuracy - test_accuracy) # difference between training and test accuracy

            if train_accuracy - test_accuracy > 0.3:
                print(f"First whale is: {whale_type}. Second whale is {other_whale_type}")

            data.append(row)

data_df = pd.DataFrame(data, columns=['First Whale', 'Second Whale', 'Training Accuracy', 'Test Accuracy', "Difference"])
print(data_df)

data_df.to_csv("compare_two_whales.csv", index=False)

# I think the algorithm has a hard time with the humpback whales and sperm whales 
all_whales.remove('humpback_whale')
all_whales.remove('sperm_whale')
filtered_df = whale_df[whale_df['species'].isin(all_whales)]

X = filtered_df.drop(columns='species')
y = filtered_df[['species']]

train_acc = []
test_acc = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, np.ravel(y_train))

    y_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    train_acc.append(train_accuracy)
    #print(f'The Training set accuracy is: {train_accuracy}.') 

    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_acc.append(test_accuracy)
    #print(f'The Test set accuracy is: {test_accuracy}.') 

print(train_acc)
print(test_acc)

print(np.mean(train_acc)) # 1.0
print(np.mean(test_acc))  # 0.58