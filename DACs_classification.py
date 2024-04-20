#Step 1:
# Import libraries

import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #to show distributions of classes.
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump #to save models as .sav
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
# Set the option to opt-in to future behavior and silence a warning about future.no_silent dowcasting displaying on console
pd.set_option('future.no_silent_downcasting', True)

#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
data = pd.read_csv('disadvantaged_communities.csv')
print("------------Start of Dataset Dimension---------")
print(f"Dataset Dimensions: {data.shape}")
print("------------End of Dataset Dimension---------")
print()
# 2. Show what column names exist for the 49 attributes in the dataset
#prints all columns in dataset
print("------------Start Column Names---------")
print("Column Names: ", end='')
for col in data: #column names are in 1st row of data
    print(col, end=', ')
    print()
print("----------End of Column Names---------")
# 3. Show the distribution of the target class CES 4.0 Percentile Range column
print()
# Set figure size
print("------------Start of distribution Figure----------")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=data, x='CES 4.0 Percentile Range')
plt.title('Distribution of CES 4.0 Percentile Range')
plt.xlabel('CES 4.0 Percentile Range')
plt.ylabel('Frequency')

# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column

plt.subplot(1, 2, 2)
percentage_distribution = data['CES 4.0 Percentile Range'].value_counts(normalize=True) * 100
sns.barplot(x=percentage_distribution.index, y=percentage_distribution.values)
plt.title('Percent Distribution of CES 4.0 Percentile Range')
plt.xlabel('CES 4.0 Percentile Range')
plt.ylabel('Distribution Percentage')

# Show the plots
plt.tight_layout()
plt.show()
print("------------End of distribution Figure----------")
print()
# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.

clean_data = data.dropna() #drop rows with null values

# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers

# Define categorical columns to be encoded
categorical_cols = [col for col in data if col != 'CES 4.0 Percentile Range']  # append all cols in list except target variable column

# Initialize OrdinalEncoder
encoder = ce.OrdinalEncoder(cols=categorical_cols)

# Fit and transform the encoder on the cleaned dataset
encoded_data = encoder.fit_transform(clean_data)



# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test

X_train = [] # Remove this line after implementing train test split
X_test = [] # Remove this line after implementing train test split

# Separate predictor variables (X) from the target variable (y)
X = encoded_data.drop(columns=['CES 4.0 Percentile Range'])  #target variable
y = encoded_data['CES 4.0 Percentile Range']

# Create train and test splits with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = X_train, X_test, y_train, y_test
# Check the shape of the splits
print("---------Start of shape Splits---------")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print("---------End of shape Splits---------")
print()
# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
from sklearn.preprocessing import StandardScaler
# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on the training set
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set using the scaler fitted on the training set
X_test_scaled = scaler.transform(X_test)

# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps

cols = X_train.columns
X_train = pd.DataFrame(X_train_scaled, columns=cols) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test_scaled, columns=cols) # pd is the imported pandas lirary - Import pandas as pd


# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)

svm_classifier = SVC(kernel='linear',C = 10.0, gamma=0.3)
#Train classifier on training data
svm_classifier.fit(X_train_scaled, y_train)

# Test the above developed SVC on unseen pulsar dataset samples
# compute and print accuracy score
print("---------Start of Accuracy Scores---------")
accuracy = svm_classifier.score(X_test_scaled,y_test)
print(f"Accuracy of SVM classifier on test data: {accuracy * 100}%")

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
dump(svm_classifier, 'svm_classifier.sav')
# You can use the library pickle to save and load your model for this assignment

# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
# Test the above developed Random Forest model on unseen DACs dataset samples

rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
# Test the above developed Random Forest model on unseen DACs dataset samples
rf_classifier.fit(X_rf_train, y_rf_train) 
# compute and print accuracy score
accuracy_rf = rf_classifier.score(X_rf_test, y_rf_test)
print(f"Accuracy of Random Forest classifier on test data: {accuracy_rf * 100}%")

print("---------End of Accuracy Scores---------")
# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
dump(rf_classifier, 'rf_classifier.sav')
# You can use the library pickle to save and load your model for this assignment

