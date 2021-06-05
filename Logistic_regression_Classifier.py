
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("experiment.csv")
print(df.head())

df.drop(['experiment-1'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2


Y = df["Productivity"].values 
Y=Y.astype('int')

X = df.drop(labels = ["Productivity"], axis=1)  


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

from sklearn.linear_model import LogisticRegression   
model = LogisticRegression() 

model.fit(X_train, y_train)  

prediction_test = model.predict(X_test)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


print(model.coef_) 
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("Weights for each variables is a follows...")
print(weights)
