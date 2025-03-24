# In this program we will predict if a person have sleep disorder or not
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print(data.head())

# Person id won't be useful so we remove it
del data['Person ID']

# Data informtion
print(data.info())

# Getting data statistics
print(data.describe())

# Label Encoding
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# features label split
features = data.drop('Sleep Disorder',axis=1)
labels = data['Sleep Disorder']

#  Modelling correlation
correlation_matrix = data.corr()

# Removing weakly correlted features
for col in data.columns:
  val = correlation_matrix['Sleep Disorder'][col]
  if(val>0 and val<0.1) or (val<0 and val>-0.1):
    del data[col]

model = keras.models.Sequential([
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dense(3,activation="sigmoid")
])

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

# Train-test split
xTrain,xTest,yTrain,yTest = train_test_split(features,labels,test_size=0.2)

# Scaling data
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

model.fit(xTrainScale,yTrain,epochs=75)

# Model Evaluation
model.evaluate(xTestScale,yTest)