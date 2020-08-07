import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# load data
# data set used: https://www.kaggle.com/ronitf/heart-disease-uci
# help with RandomizedSearchCV: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
heart_df = pd.read_csv('heart.csv', sep=',')

# first 7 lines of data
print(heart_df.head(7))

# shape
print(heart_df.shape)

# if there are null values
print(heart_df.isnull().values.any())

# general description
print(heart_df.describe())

# drop null
heart_df = heart_df.dropna()

# split into feature and target data
X = heart_df.iloc[:, :-1].values
Y = heart_df.iloc[:, -1].values

# splitting into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# feature scaling (scale value and data between 0 and 1 inclusive)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create the random grid
random_grid = { 
    'n_estimators': [1,2,3,4,5,6,7,8,9,10],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# use RandomForest Classifier
forest = RandomForestClassifier(n_estimators=10, random_state=1, min_samples_split=7, min_samples_leaf=1, max_features='auto', max_depth=2, criterion='entropy', bootstrap=False)
# forest_random = RandomizedSearchCV(estimator=forest, param_distributions=random_grid, n_iter=100, cv=3, random_state=2, n_jobs=-1)
# forest_random.fit(X_train, Y_train)
forest.fit(X_train, Y_train)

# test accuracy of model on training data set
model = forest
print(model.score(X_train, Y_train))

# test model accuracy on test data set
cm = confusion_matrix(Y_test, model.predict(X_test))

TN = cm[0][0]
TP = cm[0][0]
FN = cm[1][0]
FP = cm[0][1]

# print confusion matrix
print(cm)

# print model accuracy on test data
print('Model Test Accuracy on Confusion Matrix= {}'. format((TP + TN) / (TP + TN + FN + FP)))
print(model.score(X_test, Y_test))