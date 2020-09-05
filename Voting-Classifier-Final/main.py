import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.utils import shuffle

heart_df = pd.read_csv('heart.csv')

heart_df = heart_df[['age', 'sex', 'cp', 'trestbps', 'thalach', 'target']]

heart_df = shuffle(heart_df, random_state=722)

X = heart_df.iloc[:250, :-1]
X_test = heart_df.iloc[250:, :-1]
y = heart_df.iloc[:250, -1]
y_test = heart_df.iloc[250:, -1]

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

eclf1.fit(X, y)

y_pred = eclf1.predict_proba(X_test)

print(y_pred)

y_pred = eclf1.predict(X_test)

print(y_pred)

print(metrics.accuracy_score(y_test, y_pred))

filename = 'model.sav'
pickle.dump(eclf1, open(filename, 'wb'))
