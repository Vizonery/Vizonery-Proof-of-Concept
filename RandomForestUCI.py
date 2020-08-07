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
    'criterion': ['entropy', 'gini'],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2,3,4,5,6,7,8,9,10],
    'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
    'bootstrap': [True, False],
    'random_state': [0, 1, 2]
}

# use RandomForest Classifier
forest = RandomForestClassifier() # n_estimators=4, criterion='entropy', random_state=2)
forest_random = RandomizedSearchCV(estimator=forest, param_distributions=random_grid, n_iter=200, cv=3, n_jobs=-1)
forest_random.fit(X_train, Y_train)

results = pd.DataFrame(forest_random.cv_results_)

print(results)

results.to_csv(r'./RForestUCIFull.csv')
results_slice = results.iloc[:, [0, 2, 4, 5, 6, 7, 8, 13, 14, 15]]
results_slice.to_csv(r'./RForestUCISliced.csv')

results = pd.read_csv('RForestUCISliced.csv')
results.dropna()
results = results.sort_values(by=['mean_test_score'], ascending=False)
results['mean_test_score'].value_counts()
results = results.loc[results['mean_test_score']== results['mean_test_score'].max()]
# results.to_csv(r'./RForestUCISliced.csv')

best_random = forest_random.best_params_
print(best_random)
# results = results.loc[results['param_max_iter'] == 100]



'''
# Evaluate fuction (from link)
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestClassifier(n_estimators=13, criterion='gini', random_state=1)
base_model.fit(X_train, Y_train)
base_accuracy = evaluate(base_model, X_test, Y_test)

best_random = forest_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, Y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# test accuracy of model on training data set
model = forest_random
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
'''