#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import json
import random, string
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


maindir = "D:/Blood/data" # Directory with files
train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")


# In[19]:


# Define target columns
target_columns = ['hdl_cholesterol_human', 'hemoglobin(hgb)_human', 'cholesterol_ldl_human']

# Separate features and targets
X_train = train.drop(columns=target_columns)
X_train.drop(columns='Reading_ID', axis=1, inplace=True)
y_train = train[target_columns]

X_test = test


# In[25]:


from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in target_columns:
    encoder = LabelEncoder()
    y_train[col] = encoder.fit_transform(y_train[col])
    encoders[col] = encoder


# In[40]:


from sklearn.metrics import mutual_info_score

# Calculate MI for each pair of targets
for i in range(len(target_columns)):
    for j in range(i + 1, len(target_columns)):
        mi = mutual_info_score(y_train[target_columns[i]], y_train[target_columns[j]])
        print(f"Mutual Information between {target_columns[i]} and {target_columns[j]}: {mi:.2f}")

Feature extraction
# In[32]:


# Load mRMR selected features from JSON
with open("selected_features.json", "r") as file:
    selected_features = json.load(file)


# In[57]:


# Apply selected features to train and test data
X_train_target1 = X_train[selected_features["Target1"]]
X_train_target2 = X_train[selected_features["Target2"]]
X_train_target3 = X_train[selected_features["Target3"]]

X_test_target1 = X_test[selected_features["Target1"]]
X_test_target2 = X_test[selected_features["Target2"]]
X_test_target3 = X_test[selected_features["Target3"]]

Baseline modeling
# In[58]:


from sklearn.model_selection import StratifiedKFold

# Map targets to their respective datasets
target_to_dataset = {
    "hdl_cholesterol_human": X_train_target1,
    "hemoglobin(hgb)_human": X_train_target2,
    "cholesterol_ldl_human": X_train_target3,
}

# Redo Stratified K-Fold for each target
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Dictionary to store splits for each target
splits = {}

# Process each target
for target, X_target in target_to_dataset.items():
    
    # Get the corresponding labels
    y_target = y_train[target]
    
    # Compute Stratified K-Fold splits
    splits[target] = list(skf.split(X_target, y_target))


# In[59]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

baseline_model = RandomForestClassifier(max_depth=15, min_samples_leaf=10,
                                       min_samples_split=15, n_estimators=160,
                                       random_state=42)

results = {}

for target, fold_splits in splits.items():
    print(f"\n=== Training Models for Target: {target} ===")

    # Access target-specific dataset and labels
    X_target = target_to_dataset[target]
    y_target = y_train[target]

    results[target] = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        # Split the data
        X_train_fold = X_target.iloc[train_idx]
        X_val_fold = X_target.iloc[val_idx]
        y_train_fold = y_target.iloc[train_idx]
        y_val_fold = y_target.iloc[val_idx]

        # Train the model
        model = baseline_model
        model.fit(X_train_fold, y_train_fold)

        # Validate the model
        y_val_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}")

        # Store fold results
        results[target].append(accuracy)

# Summary of results
for target, accuracies in results.items():
    print(f"\n{target} Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")

Hyperopt for baseline
# In[63]:


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Objective function
def objective(params):
    # Unpack hyperparameters
    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])
    min_samples_split = int(params["min_samples_split"])
    min_samples_leaf = int(params["min_samples_leaf"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    accuracy = cross_val_score(model, X_train, y_train.iloc[:,0], cv=5, scoring="accuracy").mean()

    # Return negative accuracy (because Hyperopt minimizes)
    return {"loss": -accuracy, "status": STATUS_OK}


# In[64]:


# Define the hyperparameter space
search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 10),  
    "max_depth": hp.quniform("max_depth", 5, 50, 5),          
    "min_samples_split": hp.quniform("min_samples_split", 2, 20, 1),  
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1)      
}


# In[65]:


trials = Trials()

best_params = fmin(
    fn=objective,                
    space=search_space,          
    algo=tpe.suggest,            # Tree of Parzen Estimators (TPE) algorithm
    max_evals=10,                
    trials=trials,              
    rstate=np.random.default_rng(42)  
)

print("Best Hyperparameters:", best_params)


# In[66]:


# Convert best_params to integers (if necessary)
best_params = {
    "n_estimators": int(best_params["n_estimators"]),
    "max_depth": int(best_params["max_depth"]),
    "min_samples_split": int(best_params["min_samples_split"]),
    "min_samples_leaf": int(best_params["min_samples_leaf"]),
}


# In[ ]:




