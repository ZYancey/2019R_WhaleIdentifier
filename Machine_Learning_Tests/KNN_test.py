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


def doit(X, y, neigh, testSize = 0.8, numberTrials = 5, n_jobs = -1):

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
  




X = whale_np[:, 1:]
y = whale_np[:, 0]
X_norm = normalize(X, axis=0)


test_ks = [8, 12, 16, 20]
result_train_accuracies = []
result_test_accuracies = []
# for k in test_ks:
#     neigh = KNeighborsClassifier(n_neighbors=k, weights="distance", p=1)
#     train_acc, test_acc, last_prob = doit(X, y, neigh, numberTrials = 100)
#     print("k: {}, train: {}, test: {}".format(k, train_acc, test_acc))

# print("train accuracy = " + str(train_acc) + ", test accuracy = " + str(test_acc))
# print(last_prob)

# things without effect: leaf_size, normalization
# things with effect: distance weighting (yes is good), p = 1 (manhattan distance), no volume cut

file_path = 'mfcc_features.csv'

whale_df = pd.read_csv(file_path)

# Limit the dataset to just whale exclude dolphins 
all_whales = []
for whale in whale_df['species'].unique():
    if "whal" in whale:
        all_whales.append(whale)
    # else:
    #     print(whale)

whale_df = whale_df[whale_df['species'].isin(all_whales)]
all_whale_species = whale_df['species'].unique()


neigh = KNeighborsClassifier(n_neighbors=14, weights="distance", p=1)

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
            y = np.ravel(filtered_df[['species']])

            train_acc, test_acc, last_prob = doit(X, y, neigh, numberTrials = 100)
            row.append(train_acc) # train set accuracy
            row.append(test_acc) # test set accuracy

            row.append(train_acc - test_acc) # difference between training and test accuracy

            if train_acc - test_acc > 0.3:
                print(f"First whale is: {whale_type}. Second whale is {other_whale_type}")

            data.append(row)

data_df = pd.DataFrame(data, columns=['First Whale', 'Second Whale', 'Training Accuracy', 'Test Accuracy', "Difference"])
print(data_df)

data_df.to_csv("compare_two_whales.csv", index=False)


# Really good at telling false killer whales and fin finback whales from the rest of the group, (like 90-95% test accurate)
# and the rest of them are around 80-90% test accuracy  