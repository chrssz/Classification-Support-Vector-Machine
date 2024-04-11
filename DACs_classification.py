#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #to show distributions of classes.
import seaborn as sns
#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
data = pd.read_csv('disadvantaged_communities.csv')
# 2. Show what column names exist for the 49 attributes in the dataset
#prints all columns in dataset
print("Column Names: ", end='')
for row in data: #column names are in 1st row of data
    print(row, end=', ')

# 3. Show the distribution of the target class CES 4.0 Percentile Range column


# Set figure size
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

# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.

clean_data = data.dropna() #drop rows with null values

# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers


# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
X_train = [] # Remove this line after implementing train test split
X_test = [] # Remove this line after implementing train test split


# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)



# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
'''
cols = X_train.columns
X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
'''


# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)



# Test the above developed SVC on unseen pulsar dataset samples

# compute and print accuracy score



# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment




# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
'''
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Precision and use the following line to print it
precision = 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))






# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)

# Test the above developed Random Forest model on unseen DACs dataset samples

# compute and print accuracy score

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# Compute Classification Accuracy and use the following line to print it
classification_accuracy = 0
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))


'''