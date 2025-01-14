#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


maindir = "D:/Blood/data" # Directory with files
traincsv = maindir+"/Train.csv"
testcsv = maindir+"/Updated_Test.csv"

Quick look at the data 
# In[33]:


train = pd.read_csv(traincsv)
test = pd.read_csv(testcsv)


# In[7]:


train.describe()


# In[34]:


features = pd.DataFrame(train).iloc[:,1:173]
targets = pd.DataFrame(train).iloc[:, 173:]


# In[46]:


targets

Features interaction analysis
# In[5]:


corr_matrix = train.loc[:, features.columns].corr()
corr_matrix


# In[34]:


[targets.iloc[:, i].value_counts() for i in range(3)]


# In[68]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X = features

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns


vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

vif_data


# In[43]:


# Only the NIR absorbance columns
nir_columns = [col for col in features.columns[:5]]
nir_absorbance_data = features[nir_columns]

# Create boxplots for the selected NIR features
plt.figure(figsize=(15, 8))
nir_absorbance_data.boxplot(rot=90, showfliers=True)
plt.title("Boxplots for NIR Absorbance Features")
plt.ylabel("Absorbance Value")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[44]:


# A subset of NIR features for visualization
nir_columns = [col for col in features.columns[:5]]
selected_data = features[nir_columns]

# Histograms and KDE for each feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(nir_columns):
    plt.subplot(3, 2, i + 1)  
    sns.histplot(selected_data[col], kde=True, bins=30, color='blue', alpha=0.6)
    plt.title(f"Distribution of {col}")
    plt.xlabel("Absorbance Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

Data Preprocessing 
# In[35]:


from scipy.signal import savgol_filter
def apply_savitzky_golay(data, window_length=11, polyorder=2):
    """
    Applies Savitzky-Golay smoothing to each feature (wavelength).
    
    Parameters:
    - data: Pandas DataFrame (rows = samples, columns = spectral features)
    - window_length: Odd integer, the length of the filter window.
    - polyorder: Integer, the order of the polynomial to fit.
    
    Returns:
    - smoothed_data: Pandas DataFrame with smoothed features.
    """
    smoothed_data = data.apply(
        lambda x: savgol_filter(x, window_length=window_length, polyorder=polyorder), axis=0
    )
    return smoothed_data


# In[36]:


def apply_snv(data):
    """
    Applies Standard Normal Variate (SNV) preprocessing to the data.
    
    Parameters:
    - data: Pandas DataFrame (rows = samples, columns = spectral features)
    
    Returns:
    - snv_data: Pandas DataFrame with SNV applied.
    """
    snv_data = data.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)
    return snv_data


# In[37]:


original_spectral_data = train.loc[:, 'absorbance0':'absorbance169']

# Apply Savitzky-Golay smoothing
smoothed_data = apply_savitzky_golay(original_spectral_data, window_length=11, polyorder=2)

# Apply Standard Normal Variate 
preprocessed_data = apply_snv(smoothed_data)

# Replace the original spectral data in train_data with preprocessed data
train.loc[:, 'absorbance0':'absorbance169'] = preprocessed_data


# In[38]:


test_spectrum = test.loc[:, 'absorbance0':'absorbance169']

#Savitzky-Golay smoothing
test_smoothed = apply_savitzky_golay(test_spectrum, window_length=11, polyorder=2)

#Standard Normal Variate 
test_preprocessed = apply_snv(test_smoothed)

test.loc[:, 'absorbance0':'absorbance169'] = test_preprocessed


# In[70]:


# Define wavelengths (170 values from 900 nm to 1700 nm)
wavelengths = np.linspace(900, 1700, 170)

# Plot original vs smoothed vs SNV-corrected for a single sample
sample_id = 0  
original = original_spectral_data.iloc[sample_id, :].values
smoothed = smoothed_data.iloc[sample_id, :].values
snv_corrected = preprocessed_data.iloc[sample_id, :].values

plt.figure(figsize=(12, 6))
plt.plot(wavelengths, original, label="Original", color="blue", alpha=0.7)
plt.plot(wavelengths, smoothed, label="Smoothed (Savitzky-Golay)", color="orange", alpha=0.7)
plt.plot(wavelengths, snv_corrected, label="SNV Corrected", color="green", alpha=0.7)
plt.title("Effect of Preprocessing on Spectral Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.show()


# In[71]:


# A subset of ptrprocessed NIR features for visualization
nir_columns = [col for col in features.columns[:5]]
selected_data = train[nir_columns]

# Histograms and KDE for each feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(nir_columns):
    plt.subplot(3, 2, i + 1)  
    sns.histplot(selected_data[col], kde=True, bins=30, color='blue', alpha=0.6)
    plt.title(f"Distribution of {col}")
    plt.xlabel("Absorbance Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[72]:


corr_matrix = train.loc[:, features.columns].corr()
corr_matrix


# In[13]:


train.to_csv("train_preprocessed.csv", index = False)
test.to_csv("test_preprocessed.csv", index = False)

Feature Extraction
# In[93]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

#only the spectral features for Boruta
X_spectral = X.iloc[:, :-2]
y_target = y['hdl_cholesterol_human']  #one target variable for Boruta

# Use RandomForestClassifier as the base estimator
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)

# Initialize Boruta
boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)

# Fit Boruta on the data
boruta_selector.fit(X_spectral.values, y_target.values)

# Get the selected features
selected_features = X_spectral.columns[boruta_selector.support_]
print("Selected features by Boruta:")
print(selected_features)

# Reduce the dataset to the selected features
X_boruta = X_spectral[selected_features]


# In[100]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Use Random Forest as the estimator for RFE
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rfe = RFE(estimator=rf, n_features_to_select=20)  # Select top 20 features
rfe.fit(X_spectral, y_target)

# Get the selected features
rfe_selected = X_spectral.columns[rfe.support_]
print("Selected features by RFE:")
print(rfe_selected)

# Reduce the dataset to the selected features
X_rfe = X_spectral[rfe_selected]


# In[101]:


correlation_matrix = X_spectral.corr()

# Define a correlation threshold (e.g., 0.85)
threshold = 0.85

# Identify features to keep (those below the threshold)
filtered_features = []
for i in range(len(correlation_matrix)):
    correlated = np.where(correlation_matrix.iloc[i, :].abs() > threshold)[0]
    if len(correlated) == 1:  # If a feature is not highly correlated with others
        filtered_features.append(correlation_matrix.index[i])

print("Filtered features by CFS:")
print(filtered_features)

# Reduce the dataset to the filtered features
X_cfs = X_spectral[filtered_features]


# In[102]:


rf = RandomForestClassifier(random_state=42)
rf.fit(X_spectral, y_target)

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index = X_spectral.columns)
important_features = feature_importances.nlargest(20).index  # Top 20 features
print("Selected features by Random Forest:")
print(important_features)

# Reduce the dataset to the important features
X_rf = X_spectral[important_features]


# In[104]:


methods = ['Boruta', 'RFE', 'Random Forest']
num_features = [
    len(X_boruta.columns),
    len(X_rfe.columns),
    len(X_rf.columns)
]

# Bar plot for comparison
import matplotlib.pyplot as plt

plt.bar(methods, num_features, color=['blue', 'orange', 'red', 'purple'])
plt.title("Number of Features Selected by Each Method")
plt.ylabel("Number of Features")
plt.show()


# In[107]:


# Create sets of selected features
features_boruta = set(X_boruta.columns)
features_rfe = set(X_rfe.columns)
features_rf = set(X_rf.columns)

# Find common features across all sets
common_features = features_boruta  & features_rfe & features_rf
print(f"Common features across all methods: {common_features}")


# In[109]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define model
model = RandomForestClassifier(random_state=42)

# Evaluate model performance on each feature set
methods = ['Boruta', 'RFE', 'Random Forest']
datasets = [X_boruta, X_rfe, X_rf]
scores = []

for X_selected in datasets:
    score = cross_val_score(model, X_selected, y_target, cv=5, scoring='accuracy') 
    scores.append(score.mean())

# Bar plot for comparison
plt.bar(methods, scores, color=['blue', 'red', 'purple'])
plt.title("Model Performance with Different Feature Selection Methods")
plt.ylabel("Accuracy")
plt.show()


# In[40]:


from sklearn.preprocessing import LabelEncoder

# Encode all targets
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()

y_combined = pd.DataFrame({
    'Target1': label_encoder1.fit_transform(y_train[:, 0]),
    'Target2': label_encoder2.fit_transform(y_train[:, 1]),
    'Target3': label_encoder3.fit_transform(y_train[:, 2])
})


# In[43]:


import pymrmr

# Initialize a dictionary to store selected features for each target
selected_features = {}

# Iterate over targets
for target in ['Target1', 'Target2', 'Target3']:
    print(f"Processing {target}...")

    # Combine target with feature data
    data_for_mrmr = pd.concat(
        [y_combined[target], train.drop(columns=['Reading_ID', 'cholesterol_ldl_human', 'hemoglobin(hgb)_human', 'hdl_cholesterol_human'])],
        axis=1
    )

    # Apply mRMR
    selected_features[target] = pymrmr.mRMR(data_for_mrmr, 'MIQ', 10)  # Select top 10 features
    print(f"Selected Features for {target}: {selected_features[target]}")


# In[44]:


# Save selected features to a JSON file
with open("selected_features.json", "w") as file:
    json.dump(selected_features, file)
print("Selected features saved to selected_features.json")


# In[46]:


X_boruta = preprocessed_data.loc[:, ['absorbance0', 'absorbance19', 'absorbance59', 'absorbance88',
       'absorbance93', 'absorbance94', 'absorbance95', 'absorbance97',
       'absorbance98', 'absorbance126', 'absorbance130', 'absorbance138',
       'absorbance141']]


# In[48]:


from sklearn.preprocessing import StandardScaler

# Normalize X_boruta
scaler = StandardScaler()
X_boruta_scaled = scaler.fit_transform(X_boruta)
from sklearn.decomposition import PCA

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
X_boruta_pca = pca.fit_transform(X_boruta_scaled)

# Print the number of components retained
print(f"Number of components to retain 95% variance: {pca.n_components_}")


# In[49]:


import matplotlib.pyplot as plt

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
target_columns = ['hdl_cholesterol_human', 'hemoglobin(hgb)_human', 'cholesterol_ldl_human']

encoders = {}
for col in target_columns:
    encoder = LabelEncoder()
    targets[col] = encoder.fit_transform(targets[col])
    encoders[col] = encoder


# In[85]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state = 42)
# Perform cross-validation
scores = cross_val_score(model, X_boruta_pca, targets.iloc[:, 0], cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


# In[ ]:




