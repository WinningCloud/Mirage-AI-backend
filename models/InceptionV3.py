'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines a binary classification model using InceptionV3 as the
// convolutional base with custom top layers for deepfake or general image classification.
// 1) Loads pretrained InceptionV3 (without top classifier layer)
// 2) Freezes base layers to prevent weight updates in early training
// 3) Adds global average pooling and a dropout regularized dense output
// 4) Produces binary output using sigmoid activation
//
// Remarks:
// - Input shape expected is (224, 224, 3)
// - Returned model is uncompiled and ready for training or inference
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=output)
