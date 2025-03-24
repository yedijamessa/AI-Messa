# In this program we will estimte whether a person was able to survive or not on titanic.
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv('titanic.csv')
print(data.head())

# Unamed columns and passenger id can be removed
del data['Unnamed: 0'], data['PassengerId']

data['Age'] = round(data['Age']*100)
data['Fare'] = data['Fare']*100

print(data.head())

# Data information
print(data.info())

# Getting statistics of the data
print(data.describe())

# Correlation plotting
correlation_matrix = data.corr()
for col in data.columns:
  val = correlation_matrix['Survived'][col]
  if(val<0 and val>-0.1) or (val>0 and val<0.1):
    del data[col]

print(data.columns)

# Feature and label creations
features = data.drop('Survived',axis=1)
labels = data['Survived']

params={'penalty':['l1','l2','elasticnet'],
        'alpha':[0.0001,0.001,0.01,0.1,1,2,5,10],
        'max_iter':range(100,1100,100)
        }

model = Perceptron()
# Train-test split
xTrain,xTest,yTrain,yTest = train_test_split(features,labels,test_size=0.2)

# Scaling data
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

search = GridSearchCV(model,param_grid=params,scoring='accuracy')
search.fit(xTrainScale,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model Creation
model = Perceptron(alpha=0.01, max_iter=100, penalty='elasticnet')
model.fit(xTrainScale,yTrain)

yPred = model.predict(xTestScale)

# MOdel Evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))
