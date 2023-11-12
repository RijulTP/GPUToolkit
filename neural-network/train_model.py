import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Initializing the CNN
classifier = Sequential()

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating training set
training_set = train_datagen.flow_from_directory(
    './training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Creating the Test set
test_set = test_datagen.flow_from_directory(
    './test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model using fit method
classifier.fit(training_set, epochs=25, validation_data=test_set, validation_steps=2000)

# Save the trained model
classifier.save('trained_model.h5')
