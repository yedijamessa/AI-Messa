# In this program we will predict sweetner used in the food item
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('food_ingredients_and_allergens.csv')
print(data.head())

# Getting data information
print(data.info())

# In prediction  we have 1 nan value
# We check for food product
print(data[data['Food Product']=='Baked Ziti'])

# we find it consists of prediction "Contains" so we impute it with nan value
data.loc[data['Prediction'].isna(),'Prediction'] = data.loc[data['Prediction'].isna(),'Prediction'].fillna('Contains')

# Label Encoding
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=="object":
    data[col] = encoder.fit_transform(data[col])

# Feature and label seperation
features = data.drop('Sweetener',axis=1)
labels = data['Sweetener']

# Getting correlation with label
correlation_matrix= data.corr()
for col in data.columns:
  val = correlation_matrix['Sweetener'][col]
  if(val>0 and val<0.1) or (val<0 and val>-0.1):
    del data[col]

# train-test split
xTrain,xTest,yTrain,yTest = train_test_split(features,labels,test_size=0.2)

# Scaling features
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

# Model Building
model = keras.models.Sequential([
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
model.fit(xTrainScale,yTrain,epochs=75,batch_size=22)

model.evaluate(xTestScale,yTest)