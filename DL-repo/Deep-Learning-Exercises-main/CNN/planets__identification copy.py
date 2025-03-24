# Dataset link: https://www.kaggle.com/datasets/emirhanai/planets-and-moons-dataset-ai-in-space?resource=download

from tensorflow import keras
from tensorflow.keras.preprocessing.image import  ImageDataGenerator

trainDir = r'C:\Datasets\Planets\train'
testDir = r'C:\Datasets\Planets\test'

# Creating generator objects
trainGenerator = ImageDataGenerator(rescale=1./255)
testGenerator = ImageDataGenerator(rescale=1./255)

# Creating generator
trainGenerator = trainGenerator.flow_from_directory(
    trainDir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

testGenerator = testGenerator.flow_from_directory(
    testDir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Model Building
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
    # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    # as we have rgb value here so we have 3 channels,
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')]
)

# passing model learning parameters categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit_generator(trainGenerator, epochs=20, validation_data=testGenerator)

# Evaluate the model
print(model.evaluate_generator(testGenerator))