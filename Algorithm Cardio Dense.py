# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load data
# data set used: https://www.kaggle.com/sulianova/cardiovascular-disease-dataset/data
df = pd.read_csv('cardio_train.csv', sep=',')

# print first 7 rows of data
print(df.head(7))

# get shape of data

print(df.shape)

# count null values in each column
print(df.isna().sum())

# another way to check for null or missing values
print(df.isnull().values.any())

# basic stats
print(df.describe())

# count individuals with cardiovascular disease and without
print(df['cardio'].value_counts())

# look at number people without cardiovascular disease

# create years column
df['years'] = (df['age'] / 365).round(0)
df['years'] = pd.to_numeric(df['years'], downcast='integer')


# get correlation of columns
print(df.corr())

# remove years columns
df = df.drop('years', axis=1)

# remove or drop id column
df = df.drop('id', axis=1)

# split data into feature and target data

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

#Dimensionality Reduction
#pca = PCA(n_components=5)
#X=pca.fit_transform(X)

# feature scaling (scale value and data between 0 and 1 inclusive)
sc = StandardScaler()
X = sc.fit_transform(X)

# split data again, into 75% training data set and 25% testing data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)


# change targets to network acceptable format
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

#configure optimizer
SGD = SGD(lr= 0.01, decay=1e-6, momentum= 0.9, nesterov=True)

#create the model
model = Sequential()
model.add(Dense(11, activation = 'relu', input_shape = (11,)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = SGD, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#fit the model & compute accuracy on test data
model.fit(X_train, y_train, epochs = 100, batch_size=100)
print(f'Model Accuracy: {np.round(model.evaluate(X_test, y_test, batch_size=100)[1]*100,2)}%')


