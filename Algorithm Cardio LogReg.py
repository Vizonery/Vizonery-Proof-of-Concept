# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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

# create years column
df['years'] = (df['age'] / 365).round(0)
df['years'] = pd.to_numeric(df['years'], downcast='integer')

# remove years columns
df = df.drop('years', axis=1)

# remove or drop id column
df = df.drop('id', axis=1)

# split data into feature and target data
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
        
# feature scaling (scale value and data between 0 and 1 inclusive)
X = StandardScaler().fit_transform(X)


#EVERYTHING UNTIL THIS POINT SHOULD BE THE SAME

#make gridsearch params
parameters = {
    'penalty':['l1', 'l2', 'elasticnet', 'none'],
    'C':np.logspace(0,4,20),
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 2500]
}
#make gridsearch
clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, verbose=3, n_jobs=-1, cv=4, return_train_score=True)


#THE STEPS BELOW WILL VARY BASED ON YOUR MODELS
#results  = pd.DataFrame(clf.cv_results_)
#results.to_csv('LogReg Cardio Results Full')
#results_slice = results.iloc[:, [0, 2, 4, 5, 6, 7, 8, 13, 14, 15]]
#results_slice.to_csv('LogReg Cardio Results Sliced')

#results = pd.read_csv('LogReg Cardio Results Sliced')
#results.dropna()
#results = results.sort_values(by=['mean_test_score'], ascending=False)
#results['mean_test_score'].value_counts()
#results = results.loc[results['mean_test_score']== results['mean_test_score'].max()]
#results = results.loc[results['param_max_iter'] == 100]
