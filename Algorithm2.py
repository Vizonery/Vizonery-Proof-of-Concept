# Source: https://www.kaggle.com/dileep070/logistic-regression
# import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.mlab as mlab
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.tools import add_constant as add_constant
import sklearn
from sklearn import metrics
# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# import dataset
heart_df = pd.read_csv('framingham.csv')
heart_df.drop(['education'], axis=1, inplace=True)
print(heart_df.head())

# rename column name
heart_df.rename(columns={'male': 'Sex_male'}, inplace=True)
print(heart_df.head())

# check for missing values
print(heart_df.isnull().sum())

# count missing values and drop them
count = 0
for i in heart_df.isnull().sum(axis=1):
    if i > 0:
        count = count+1
print('Total number of rows with missing values is ', count)
print('since it is only', round((count/len(heart_df.index))*100),
      'percent of the entire dataset the rows with missing values are excluded.')

heart_df.dropna(axis=0, inplace=True)

print(heart_df.describe())

# check null values again
print(heart_df.isnull().sum())

# add constant
heart_df_constant = add_constant(heart_df)
print(heart_df_constant.head())

# logistic regression

# Chi-square Test
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols = heart_df_constant.columns[:-1]
model = sm.Logit(heart_df.TenYearCHD, heart_df_constant[cols])
result = model.fit()
print(result.summary())

# P-value approach


def back_feature_elem(data_frame, dep_var, col_list):
    while len(col_list) > 0:
        model = sm.Logit(dep_var, data_frame[col_list])
        result = model.fit(disp=0)
        largest_pvalue = round(result.pvalues, 3).nlargest(1)
        if largest_pvalue[0] < (0.05):
            return result
            break
        else:
            col_list = col_list.drop(largest_pvalue.index)


result = back_feature_elem(heart_df_constant, heart_df.TenYearCHD, cols)

# interpreting results: Odds Ration, Confidence Intervals and Pvalues
params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue = round(result.pvalues, 3)
conf['pvalue'] = pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio', 'pvalue']
print((conf))

# splitting data to train and test split
new_features = heart_df[['age', 'Sex_male', 'cigsPerDay',
                         'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = new_features.iloc[:, :-1]
y = new_features.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=5)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# model accuracy
print(sklearn.metrics.accuracy_score(y_test, y_pred))
