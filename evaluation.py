# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import pandas as pd
import joblib as load  #to load .sav files
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import category_encoders as ce
# Set the option to opt-in to future behavior and silence a warning about future.no_silent dowcasting displaying on console
pd.set_option('future.no_silent_downcasting', True)

# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point
data = pd.read_csv('disadvantaged_communities.csv')
#Clean data
clean_data = data.dropna() #drop rows with null values
#Target Variable to drop when declaring encoder 
categorical_col =  [col for col in data if col != 'CES 4.0 Percentile Range']
#Encode Data
encoder = ce.OrdinalEncoder(cols=categorical_col)
#fit cleaned data
encoded_data = encoder.fit_transform(clean_data)

# Separate predictor variables (X) from the target variable (y)
X = encoded_data.drop(columns=['CES 4.0 Percentile Range'])  #target variable
y = encoded_data['CES 4.0 Percentile Range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# Standardize data for the SVM

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Load your saved model for disadvantaged communities classification 
# that you saved in disadvantaged_communities_classification.py via Pickle
svm = load.load('svm_classifier.sav')  # Support Vector Machine model
rf = load.load('rf_classifier.sav')  # Random Forest model

# 4. Make predictions on the test set created from step 2

svm_predict = svm.predict(X_test_scaled)  # SVM prediction
rf_predict = rf.predict(X_test)  # Random Forest prediction

# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

# SVM Metrics
# Compute Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, svm_predict)
svm_accuracy = accuracy_score(y_test, svm_predict)
# Compute Precision Score for SVM 
svm_precision = precision_score(y_test, svm_predict, average='weighted')
# Compute recall Score for SVM 
svm_recall = recall_score(y_test, svm_predict, average='weighted')
svm_specificity = svm_cm[0, 0] / (svm_cm[0, 0] + svm_cm[0, 1])

# Random Forest Metrics
# Compute confusion Matrix for SVM
rf_cm = confusion_matrix(y_test, rf_predict)
rf_accuracy = accuracy_score(y_test, rf_predict)
# Compute Precision Score for RF 
rf_precision = precision_score(y_test, rf_predict, average='weighted')
# Compute recall Score for RF 
rf_recall = recall_score(y_test, rf_predict, average='weighted')
rf_specificity = rf_cm[0, 0] / (rf_cm[0, 0] + rf_cm[0, 1])

# Print SVM Metrics and Confusion Matrix
print("SVM Metrics:")
print("Confusion Matrix:")
print(svm_cm)
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("Specificity:", svm_specificity)
print()

# Print Random Forest Metrics and Confusion Matrix
print("Random Forest Metrics:")
print("Confusion Matrix:")
print(rf_cm)
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("Specificity:", rf_specificity)