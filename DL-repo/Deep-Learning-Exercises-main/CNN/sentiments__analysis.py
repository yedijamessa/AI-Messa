# In this program we will estimate if a image is  of happiness or sadness

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainDir = r'Data\Sentiments\train'
testDir = r'Data\Sentiments\test'

# Creating data generator objects
trainDataGen = ImageDataGenerator(rescale=1./255)
testDataGen = ImageDataGenerator(rescale=1./255)

# Creating generator
trainGenerator = trainDataGen.flow_from_directory(
    trainDir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

testGenerator = testDataGen.flow_from_directory(
    testDir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

 # Model creation
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape=(150,150,3)), # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=64,kernel_size=(3,3), activation='relu'), # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=128,kernel_size=(3,3), activation='relu'), # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=256,kernel_size=(3,3), activation='relu'), # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])


# passing model learning parameters categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit_generator(trainGenerator, epochs=20, validation_data=testGenerator)

# Evaluate the model
print(model.evaluate_generator(testGenerator))