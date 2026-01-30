'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines the architecture for a binary classification model using EfficientNetB4
// as the backbone, with custom classification layers added on top.
// 1) Loads EfficientNetB4 pretrained on ImageNet as the base model
// 2) Freezes base layers to preserve learned features
// 3) Adds Global Average Pooling, BatchNormalization, Dense and Dropout layers
// 4) Outputs a binary classification prediction using sigmoid activation
//
// Remarks:
// - The model expects input images of shape (224, 224, 3)
// - Returns a compiled-ready Keras model instance
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_shape)

    # ✅ Freeze base model to prevent large gradient flow initially
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)                  # ✅ Normalize features before Dense
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
