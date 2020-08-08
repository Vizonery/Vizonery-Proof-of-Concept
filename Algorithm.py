# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# load data
# data set used: https://www.kaggle.com/sulianova/cardiovascular-disease-dataset/data
df = pd.read_csv('cardio_train.csv', sep=',')

# print first 7 rows of data
print(df.head(7))

# get shape of data
print(df.shape)

# count null values in each column
print(df.isna().sum())

# another way to check for null or missing values, returns True or False
print(df.isnull().values.any())

# basic stats
print(df.describe())

# count individuals with cardiovascular disease and without
print(df['cardio'].value_counts())

# visual of count
sns.countplot(x='cardio', data=df)
# plt.show()

# look at number people without cardiovascular disease

# create years column
df['years'] = (df['age'] / 365).round(0)
df['years'] = pd.to_numeric(df['years'], downcast='integer')

# visualize data
sns.countplot(x='years', hue='cardio', data=df, palette='colorblind',
              edgecolor=sns.color_palette('dark', n_colors=1))
# plt.show()

# get correlation of columns
print(df.corr())

# visualize data
plt.figure(figsize=(7, 7))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
# plt.show()

# remove years columns
df = df.drop('years', axis=1)

# remove or drop id column
df = df.drop('id', axis=1)

# split data into feature and target data

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# split data again, into 75% training data set and 25% testing data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# feature scaling (scale value and data between 0 and 1 inclusive)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# use RandomForest Classifier
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
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
print('Model Test Accuracy = {}'. format((TP + TN) / (TP + TN + FN + FP)))
