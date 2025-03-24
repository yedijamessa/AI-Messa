# In this program we will estimate the temperature of the delhi and forecast for next 15 days
import yfinance as y
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math

data = y.download('MSFT',start="2011-01-01",end="2023-01-01")
data=data[['Close']]
print(data.head())

# Scaling data
scaler = MinMaxScaler()
dataScaled = scaler.fit_transform(data)

# Reshape the dta
sData = dataScaled.reshape(-1,1)

#splitting dataset into train and test split
training_size=int(len(sData)*0.7)
test_size=len(sData)-training_size
train_data,test_data=sData[0:training_size,:],sData[training_size:, :1]

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 90
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# Create the Stacked LSTM model
model=Sequential([
LSTM(50,return_sequences=True,input_shape=(90,1)),
LSTM(50,return_sequences=True),
LSTM(50),
Dense(1)])
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

# Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
print(train_predict, test_predict)

# Calculate RMSE performance metrics
print(math.sqrt(mean_squared_error(ytest,test_predict)))

# Forecast for next 15 days
x_input = test_data[test_data.shape[0]-time_step:].reshape(1,-1)
print(x_input.shape)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=90
i=0
while(i<15):
    if(len(temp_input)>90): # since we require 90 days data to predict forward
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1

# Printing the forecast values
print(scaler.inverse_transform(lst_output))
