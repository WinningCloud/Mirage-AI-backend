'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines a binary classification model using ResNet50 as the base
// convolutional feature extractor, combined with a custom classification head.
// 1) Loads pretrained ResNet50 without its top classification layer
// 2) Freezes the base layers to preserve learned features from ImageNet
// 3) Adds global average pooling and dense layers for binary classification
// 4) Final output is a sigmoid probability for fake/real classification tasks
//
// Remarks:
// - Input shape is expected to be (224, 224, 3)
// - The returned model is uncompiled and can be trained using binary_crossentropy
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def build_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    return model
