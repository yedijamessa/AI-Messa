# In this program we will estimate the price of the car
# Dataset link : https://www.kaggle.com/datasets/alpertemel/turkey-car-market-2020?resource=download

import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


# Loading dataset
data = pd.read_csv('turkey_car_market.csv')
print(data.head())

# As date won't be useful so we remove it
del data['Date']

# Getting data information
print(data.info())

# Checking for outliers
print(data.describe())

# Potential outliers are price and Km
sns.scatterplot(data['Km'])

# Km outlier treatment
maxPercentile = np.percentile(data['Km'], [99])

 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['Km']>3*upperValue,'Km']=3*upperValue

# Checking outliers for price
sns.scatterplot(data['Price'])

# Price outlier treatment
maxPercentile = np.percentile(data['Price'], [99])
 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['Price']>3*upperValue,'Price']=3*upperValue

# Label Encoding
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Generating correlation
correlation_matrix = data.corr()
for col in data.columns:
  val = correlation_matrix['Price'][col]
  if(val<0 and val>-0.1) or (val>0 and val<0.1):
    del data[col]

# Feature and label creation 
features = data.drop('Price',axis=1)
label = data['Price']

# Train-Test split
xTrain,xTest,yTrain,yTest = train_test_split(features,label,test_size=0.3)

# Scaling features
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

# Model Building
model = keras.models.Sequential([
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dense(1,activation="linear")
])

model.compile(loss="mean_squared_error",optimizer="adam",metrics=['mae'])
model.fit(xTrainScale,yTrain,epochs=100)

# Model Evaluation
model.evaluate(xTestScale,yTest)

# r2_score
yPred = model.predict(xTestScale)
print(r2_score(yTest,yPred))