import pandas as pd

df=pd.read_csv('heartdisease.csv')
df.head()
df=df.drop(columns=['education'])
df.isnull().sum()

df.shape

bin_cols=["male", "currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]
for col in bin_cols:
    mode_val=df[col].mode()[0]
    df[col].fillna(mode_val,inplace=True)


missing_values=df.isnull().sum()

import numpy as np

numeric_cols=["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
for col in numeric_cols:
    median_val=df[col].median()
    df[col].fillna(median_val,inplace=True)

missing_values=df.isnull().sum()

df['TenYearCHD'].value_counts()

from sklearn.utils import resample

df_majority = df[df['TenYearCHD']==0]
df_minority = df[df['TenYearCHD']==1]

df_minority_upsampled= resample(df_minority,
                                replace=True,
                                n_samples=len(df_majority),
                                random_state=42)
df_balanced=pd.concat([df_majority,df_minority_upsampled])

df_balanced['TenYearCHD'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X=df_balanced.drop(columns=['TenYearCHD'])
y=df_balanced['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

classifiers = [
     RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    XGBClassifier()
]

results={}

for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test,y_pred)
    print(f"{clf_name} Accuracy:{accuracy}")

    print(f"Classification Report for {clf_name}:")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix for {clf_name}:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define a list of classifiers
classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    XGBClassifier()
]

# Create a DataFrame to store the results
# Create a list to store the results first
results_list = []

# Train and evaluate each classifier
for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_score_ = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    
    # Append results as a dictionary
    results_list.append({'Model': clf_name, 'Accuracy': accuracy, 'F1-Score': f1_score_, 
                         'Precision': precision, 'Recall': recall})

# After loop, create DataFrame once
results_df = pd.DataFrame(results_list)

results_df


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Instantiate the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Train the RandomForestClassifier
rf_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)

# Classification report
print("Classification Report for Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test, y_pred_rf))

print("predcted class ",rf_classifier.predict(X_test_scaled[10].reshape(1,-1))[0])
print("actual class ", y_test.iloc[10])

print("predcted class ",rf_classifier.predict(X_test_scaled[200].reshape(1,-1))[0])
print("actual class ", y_test.iloc[200])

# test 3:
print("predcted class ",rf_classifier.predict(X_test_scaled[110].reshape(1,-1))[0])
print("actual class ", y_test.iloc[110])

import pickle
pickle.dump(rf_classifier,open("rf_classifier.pkl",'wb'))
pickle.dump(scaler,open("scaler.pkl",'wb'))

import pickle

# Load the RandomForestClassifier model
with open("rf_classifier.pkl", "rb") as file:
    rf_classifier = pickle.load(file)

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

import numpy as np

def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    # Encode categorical variables
    male_encoded = 1 if male.lower() == "male" else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == "yes" else 0
    BPMeds_encoded = 1 if BPMeds.lower() == "yes" else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == "yes" else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == "yes" else 0
    diabetes_encoded = 1 if diabetes.lower() == "yes" else 0
    
    # Prepare features array
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded, prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    
    # scalling
    scaled_features = scaler.transform(features)
    
    # predict by model
    result = model.predict(scaled_features)
    
    return result[0]

# test 1:
male = "female"
age = 56.00
currentSmoker = "yes"
cigsPerDay = 3.00
BPMeds = "no"
prevalentStroke = "no"
prevalentHyp = "yes"
diabetes = 'no'
totChol = 285.00
sysBP = 145.00
diaBP = 100.00
BMI = 30.14
heartRate = 80.00
glucose = 86.00


result = predict(rf_classifier, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)


if result == 1:
    print("The Patient has Heart Diseas")
else: 
    print("The Patiennt has No Heart Deseas")

male = 'female'
age = 63.0
currentSmoker = 'yes'
cigsPerDay = 3.0
BPMeds = 'no'
prevalentStroke = 'no'
prevalentHyp = 'yes'
diabetes = 'no'
totChol = 267.0
sysBP = 156.5
diaBP = 92.5
BMI = 27.1
heartRate = 60.0
glucose = 79.0
result = 1.0



result = predict(rf_classifier, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)


if result == 1:
    print("The Patient has Heart Diseas")
else: 
    print("The Patiennt has No Heart Deseas")

import sklearn
sklearn.__version__