
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import os

def create_deepconvnet_model(input_shape=(400, 400, 19), num_classes=2):
    model = Sequential([
        Conv2D(32, kernel_size=(1, 5), strides=(1, 2), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),

        Conv2D(64, kernel_size=(1, 5), strides=(1, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),

        Conv2D(128, kernel_size=(1, 5), strides=(1, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
