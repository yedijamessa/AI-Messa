# In this program we will estimate the temperature of the Delhi
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv('delhi_weather.csv',parse_dates=[0])
print(data.head())

print(data.tail())

# Since only 1 2017 year value is there we remove it
data= data[data['date'].dt.year!=2017]

# checking data information
print(data.info())

# Univariate data analysis
print(data.describe())

# Plotting the relations with the meantemp
sns.pairplot(data,x_vars=['humidity','wind_speed','meanpressure'],y_vars='meantemp')

# pressure is not at all related to temperature so we can remove them
del data['meanpressure']
correlation_matrix = data.corr()
print(correlation_matrix['meantemp'])

df = data.copy()
df.index = df['date']
df['meantemp'].plot()

# There is a seasonality in the data
# plotting the trend
sns.regplot(x=data.index.values, y= data['meantemp'])

# There is a growing trend of mean temperature 
# lets decompose the data
result = seasonal_decompose(df['meantemp'],model='multiplicative')
result.plot()
plt.show()

# resampling data to get the seasonality
quarterlyTemp = data.resample('Q',on='date').mean()
yearlyTemp = data.resample('A',on='date').sum()

yearlyTemp['meantemp'].plot()
# There is a sharp growth in temperature

quarterlyTemp['meantemp'].plot()
# There is seasonlaity in graph

# Scaling data
scaler = MinMaxScaler()
data = data[['meantemp','humidity','wind_speed']]
dataScale = scaler.fit_transform(data)

import numpy as np
# Split the data into training and test sets
train_size = int(data.shape[0]*0.8)
train_data = dataScaled[:train_size]
test_data = dataScaled[train_size:]

# Prepare the training data
X_train, y_train = [], []
for i in range(3, len(train_data)):
    X_train.append(train_data[i-3:i])
    y_train.append(train_data[i, 0]) # here 0th index is passed as it is the target variable index in our data
X_train, y_train = np.array(X_train), np.array(y_train)

# Prepare the test data
X_test, y_test = [], []
for i in range(3, len(test_data)):
    X_test.append(test_data[i-3:i])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

from keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
#  Build the LSTM model
model = Sequential([
LSTM(50, return_sequences=True, input_shape=(3, 3),activation='relu'),
LSTM(50,activation='relu'),
Dense(1,activation='linear')]
)
model.compile(optimizer='adam', loss='mse',metrics='mae')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=22)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse scale the predictions
inverse_train_predictions = scaler.inverse_transform(np.concatenate((X_train[:, -1, 1:], train_predictions), axis=1))[:, -1]
inverse_test_predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, 1:], test_predictions), axis=1))[:, -1]

# Print the predictions
print('Train Predictions:', inverse_train_predictions)
print('Test Predictions:', inverse_test_predictions)


print(r2_score(y_test,test_predictions))
# 91.3%
