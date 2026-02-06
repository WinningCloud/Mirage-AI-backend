import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import your model architectures
from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

MODEL_BUILDERS = {
    "Xception": Xception.build_model,      # Industry standard for deepfakes
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "ResNet50": ResNet50.build_model,
    "MobileNetV2": MobileNetV2.build_model
}

# Industry standard for most Deepfake models is 299x299, 
# but we will keep 224 if your weights were trained on that.
IMG_SIZE = (224, 224) 
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")

def load_all_models_once():
    loaded_models = {}
    print("\n--- 🧠 Loading Forensic Engines ---")
    
    for name, builder in MODEL_BUILDERS.items():
        weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
        if not os.path.exists(weight_path):
            continue

        try:
            model = builder(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
            # IMPORTANT: Try to avoid skip_mismatch=True if possible. 
            # If weights don't match exactly, the model predicts random noise.
            model.load_weights(weight_path, skip_mismatch=False) 
            loaded_models[name] = model
            print(f"✅ Ready: {name}")
        except Exception as e:
            print(f"⚠️ {name} load failed: {e}. Attempting fallback...")
            try:
                model.load_weights(weight_path, skip_mismatch=True)
                loaded_models[name] = model
            except:
                pass
    return loaded_models

def preprocess_for_forensics(image_path):
    """
    Improved Preprocessing: 
    Deepfake models perform best when scaled between -1 and 1 
    rather than just 0 and 1.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    
    # Scale from [0, 255] to [-1, 1] (Standard for Xception/Inception)
    img_array /= 127.5
    img_array -= 1.0
    
    return np.expand_dims(img_array, axis=0)

def predict_deepfake(image_path, models):
    img_tensor = preprocess_for_forensics(image_path)
    results = {}
    
    # Adjust this if your model was trained 0=Fake, 1=Real
    # Usually: 1 = FAKE, 0 = REAL
    FAKE_THRESHOLD = 0.5 

    for name, model in models.items():
        try:
            # Get raw probability
            prob = float(model.predict(img_tensor, verbose=0)[0][0])
            
            # Logic: If prob is closer to 1, it's FAKE.
            label = "FAKE" if prob >= FAKE_THRESHOLD else "REAL"
            
            results[name] = {
                "label": label,
                "confidence": prob
            }
        except Exception as e:
            print(f"Prediction error on {name}: {e}")
            
    return results