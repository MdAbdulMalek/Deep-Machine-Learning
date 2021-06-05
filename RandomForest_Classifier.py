
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


df = pd.read_csv("experiment.csv")
print(df.head())


df.drop(['experiment-1'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


Y = df["Productivity"].values  
Y=Y.astype('int')

X = df.drop(labels = ["Productivity"], axis=1)  
#print(X.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
model.fit(X_train, y_train)


prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

from sklearn import metrics
#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)



