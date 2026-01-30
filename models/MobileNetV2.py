'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines a binary classification model using MobileNetV2 as the
// convolutional base, with a custom top classification layer.
// 1) Loads pretrained MobileNetV2 without the top classification layer
// 2) Adds global average pooling, dropout, and a dense sigmoid output layer
// 3) Outputs binary classification (e.g., FAKE vs REAL)
//
// Remarks:
// - The model expects input images of shape (224, 224, 3)
// - The final output is a probability via sigmoid activation
// - Weights for the base model are loaded from ImageNet
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)  # ✅ Correct for binary classification
    model = Model(inputs=base_model.input, outputs=output)
    return model
