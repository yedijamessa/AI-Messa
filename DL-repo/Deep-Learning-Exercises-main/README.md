# Deep Learning Portfolio

Repository consists of programs that use various concepts and models that does classification, regression and image classification like tasks.The repository consists of folders such as ANN, CNN and RNN which defines the type of neural network used to carry out various tasks.

## Contents
+ [About](#intro) 
+ [Applications](#applications) 
+ [Terms](#terms)
+ [Steps Involved](#steps)
+ [Libraries Used](#library)
+ [Programs Description](#program)
+ [LICENSE](LICENSE)


<a id="intro"></a><h2>About</h2>

Deep learning is a subfield of machine learning that focuses on building and training artificial neural networks to mimic the functioning of the human brain. It involves creating models with multiple layers of interconnected neurons, known as artificial neural networks, to process and learn from large volumes of data.Deep learning is preferred many a times than general algorithms because it uses neural networks and various attributes such as activation functions, loss function and optimizers which enhances the model performance by error correction with every iteration of training.


Generally normal machine learning algorithms can be used for the tasks of such as classifications, regression etc. Deep learning comes into play when data is in form of speech,image or video etc in which data can't be computed directly.

<a id="applications"></a><h2>Applications</h2>
There are a wide range of applications of deep learning few are mentioned below:

+ Computer Vision(image related works)
+ Natural Language Processing(speech related works)
+ Speech Recognition
+ Autonomous Vehicles
+ Recommender Systems
+ Virtual Assistants


<a id="terms"></a><h2>Terms</h2>
To get started with the deep learning we need to have knowledege about several terms used.The terms and their meaning are described below:

| TERMS        | MEANING       
| ------------- |:-------------:|
**Artificial Neural Network (ANN)** | A computational model inspired by the structure and function of the human brain, composed of interconnected artificial neurons.
**Neuron** | The basic building block of an artificial neural network, also known as a node or unit. It receives inputs, applies an activation function, and produces an output.
**Activation Function** | A function that determines the output of a neuron based on its weighted inputs.
**Loss Function** | A function that measures the difference between predicted and actual outputs, used to guide the training process.
**Feedforward Neural Network** | A neural network where the information flows only in one direction, from the input layer to the output layer.
**Backpropagation** |An algorithm used to update the weights of a neural network by minimizing the error between predicted and actual outputs during training.
**Gradient Descent** | An optimization algorithm used in backpropagation to find the optimal weights that minimize the loss function.
**Vanishing Gradient Problem** | A challenge in training deep neural networks where gradients become very small, making it difficult to update early layers effectively.

**Note:** The general machine learning terms are not discussed here they can be found [here](https://github.com/Sandy0002/Machine-Learning-Exercises/blob/main/README.md#terms).

<a id="steps"></a><h2>Steps Involved</h2>
Since deep learning is subset of machine learning the steps involved are same. You can look for them [here](https://github.com/Sandy0002/Machine-Learning-Exercises/blob/main/README.md#steps)


<a id="library"></a><h2>Libraries Used</h2>
+ **Numpy** : Used for numerical computations in python
+ **Pandas** : Used for file reading and other operations when working with large data.
+ **Sklearn(Scikit-learn)** : This is a machine learning library for python.
+ **Matplotlib** : Visualization library
+ **Seaborn** : Interactive visualizations are made using these library.
+ **Tensorflow** : The deep learning framework made by Google for the lower level computations.
+ **Keras** : Model level library for carrying out modelling tasks.


<a id="program"></a><h2>Programs Description</h2>

The datasets used for these program are downloaded from **kaggle**.Datasets can be found [here](https://github.com/Sandy0002/Deep-Learning-Exercises/tree/main/Datasets).
For some programs which have heavy datasets their links are in respective programs. 


+ ### ANN
  ANN stands for Artificial Neural Networks.They are used for general classification or regression problems where normal machine learning algorithms won't work then to get better results these are used.
  All the programs below uses artificial neural networks for solving problems.
  + **Car Prices** : This program uses turkey's car specifications dataset and does estimation of the prices. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/ANN/car_price.py)
  + **Have Sleep Disorder** : Predicting if a person have a sleep disorder or not. If a person have sleep disorder then what kind of disorder it is. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/ANN/have_sleep_disorder.py)
  + **Sweetner Type** : Here we have various food items and trying to identify the type of sweetner add in the food. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/ANN/sweetener_type.py)
  + **Titanic Survival** : The famous tragedy of Titanic 1912. We are trying to identify if a person was able to survive or not. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/ANN/titanic.py)


+ ### CNN
  CNN stands for Convolutional neural networks. They are used for  image classification tasks.The programs below demonstrate the same.
  + **Sentiments Analysis** : In this program we are trying to estimate if the mood in the image is happy or sad. This program demonstrates binary classification problem. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/CNN/planets__identification%20copy.py)
  + **Planets and Moon Identification** : In this program we are estimating if the used is of a planet or moon. If its a planet then which planet it is. This program demonstrates multi-label classification. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/CNN/sentiments__analysis.py)

+ ### RNN
  RNN stands for Recurrent Neural network. Uses concept of back propogation which increses the accuracy of the results. They are preferred in the problems such as natural language processing, time series analysis etc.

  + **Microsoft Stock Prices**: This program demonstrtes time series forecasting using a RNN model LSTM. LSTM stands for Long Short Term memory model specifically used for time series analysis. In this program stock prices of microsoft are forecasted. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/RNN/ms_stocks_price.py)
  + **Delhi Temperature Forecast** : In this program delhi temperature is being forecasted using LSTM model. Program [link](https://github.com/Sandy0002/Deep-Learning-Exercises/blob/main/RNN/temperature_forecast_lstm.py)


## LICENSE
[MIT LICENSE](LICENSE)
