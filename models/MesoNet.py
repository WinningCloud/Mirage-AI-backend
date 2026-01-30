'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines a lightweight custom Convolutional Neural Network (CNN)
// for binary image classification (e.g., fake vs real). It avoids using pretrained networks,
// making it suitable for smaller datasets or controlled environments.
// 1) Builds a sequential CNN architecture with increasing depth
// 2) Applies Batch Normalization and MaxPooling after each convolution block
// 3) Adds dropout regularization and dense classification head
// 4) Outputs a probability for binary classification via sigmoid activation
//
// Remarks:
// - The input shape is expected to be (256, 256, 3)
// - Model is uncompiled and must be compiled before training
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, MaxPooling2D, Dropout

def build_model(input_shape=(256, 256, 3)):
    model = Sequential()

    model.add(Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(8, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(16, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(16, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model
