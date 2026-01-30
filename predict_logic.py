import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Ensure your models folder is in the same directory or adjust the import path
from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

MODEL_BUILDERS = {
    "ResNet50": ResNet50.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "MobileNetV2": MobileNetV2.build_model,
    "Xception": Xception.build_model
}

IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")

def load_all_models_once():
    loaded_models = {}
    print("\n--- 🧠 Loading AI Models ---")
    
    for name, builder in MODEL_BUILDERS.items():
        weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
        
        if not os.path.exists(weight_path):
            print(f"⚠️ Missing: {name} (Checked {weight_path})")
            continue

        try:
            # 1. Build the architecture
            model = builder(input_shape=(224, 224, 3))
            
            # 2. Try loading weights with legacy support
            # We use skip_mismatch=True to avoid the 'vars' error in newer Keras versions
            model.load_weights(weight_path, skip_mismatch=True)
            
            loaded_models[name] = model
            print(f"✅ Loaded: {name}")
            
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            # Final attempt: Loading using the native H5PY interface if Keras fails
            try:
                import h5py
                model = builder(input_shape=(224, 224, 3))
                with h5py.File(weight_path, 'r') as f:
                    model.load_weights(weight_path)
                loaded_models[name] = model
                print(f"✅ Loaded: {name} (via H5PY fallback)")
            except Exception as e2:
                print(f"🔥 Critical Failure for {name}: {e2}")

    return loaded_models

def predict_deepfake(image_path, models):
    # Preprocess the image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    results = {}
    for name, model in models.items():
        try:
            # Run prediction
            prob = model.predict(img_array, verbose=0)[0][0]
            results[name] = {
                "label": "FAKE" if prob >= 0.5 else "REAL",
                "confidence": float(prob)
            }
        except Exception as e:
            print(f"Prediction error on {name}: {e}")
            
    return results