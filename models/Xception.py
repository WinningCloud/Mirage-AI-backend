'''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) GAHRANOX INFOSEC 2025
//
// @Author: Mohammed Faadil
//
// Purpose: This module defines a custom binary classification model using Xception as the
// base architecture for feature extraction. Custom fully connected layers are added on top.
// 1) Loads Xception without the top classification layer, using ImageNet weights
// 2) Freezes the base layers to retain pretrained feature maps
// 3) Adds global average pooling, normalization, dropout, and dense layers
// 4) Produces binary classification output via sigmoid activation
//
// Remarks:
// - Input image shape: (224, 224, 3)
// - The model is returned uncompiled for flexibility in usage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False  # ✅ Freeze base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
