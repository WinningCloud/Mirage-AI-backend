import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import your model architectures
from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

MODEL_BUILDERS = {
    "Xception": Xception.build_model,
    "EfficientNetB4": EfficientNetB4.build_model,
    "InceptionV3": InceptionV3.build_model,
    "ResNet50": ResNet50.build_model,
    "MobileNetV2": MobileNetV2.build_model,
}

# Input sizes per model (industry standard)
MODEL_INPUT_SIZES: Dict[str, Tuple[int, int]] = {
    "Xception": (299, 299),
    "InceptionV3": (299, 299),
    "EfficientNetB4": (224, 224),
    "ResNet50": (224, 224),
    "MobileNetV2": (224, 224),
}

# Preprocessing per model. Override with env var if your weights used a different scale.
# Allowed values: "-1_1", "0_1"
MODEL_PREPROCESSING: Dict[str, str] = {
    "Xception": "-1_1",
    "InceptionV3": "-1_1",
    "EfficientNetB4": "0_1",
    "ResNet50": "0_1",
    "MobileNetV2": "0_1",
}

# Output polarity per model. If your model outputs probability of REAL, set to "real".
# Allowed values: "fake", "real"
MODEL_OUTPUT_POLARITY: Dict[str, str] = {
    "Xception": "fake",
    "InceptionV3": "fake",
    "EfficientNetB4": "fake",
    "ResNet50": "fake",
    "MobileNetV2": "fake",
}

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")


def _preprocess_image(image_path: str, target_size: Tuple[int, int], mode: str) -> np.ndarray:
    """
    Preprocess image for deepfake forensic models.
    Uses [-1, 1] scaling which is standard for Xception/Inception-based models
    and works well for most custom CNNs in this project.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype("float32")

    if mode == "-1_1":
        img_array = (img_array / 127.5) - 1.0
    else:
        img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def load_all_models_once() -> Dict[str, tf.keras.Model]:
    loaded_models: Dict[str, tf.keras.Model] = {}
    print("\n--- 🧠 Loading Forensic Engines ---")

    for name, builder in MODEL_BUILDERS.items():
        weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
        if not os.path.exists(weight_path):
            continue

        try:
            input_size = MODEL_INPUT_SIZES.get(name, (224, 224))
            model = builder(input_shape=(input_size[0], input_size[1], 3))
            model.load_weights(weight_path, skip_mismatch=False)
            loaded_models[name] = model
            print(f"✅ Ready: {name}")
        except Exception as e:
            print(f"⚠️ {name} load failed: {e}. Attempting fallback...")
            try:
                model.load_weights(weight_path, skip_mismatch=True)
                loaded_models[name] = model
                print(f"✅ Ready (fallback): {name}")
            except Exception as fallback_error:
                print(f"❌ {name} fallback failed: {fallback_error}")

    return loaded_models


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _predict_single_model(
    model: tf.keras.Model,
    image_path: str,
    input_size: Tuple[int, int],
    preprocess_mode: str,
    output_polarity: str,
) -> float:
    img_tensor = _preprocess_image(image_path, input_size, preprocess_mode)
    raw = float(model.predict(img_tensor, verbose=0)[0][0])

    # If output looks like logits, convert to probability
    if raw < 0.0 or raw > 1.0:
        prob = _sigmoid(raw)
    else:
        prob = raw

    prob = float(np.clip(prob, 0.0, 1.0))

    # Convert to probability of FAKE if model outputs REAL
    if output_polarity == "real":
        prob = 1.0 - prob

    return float(np.clip(prob, 0.0, 1.0))


def _aggregate_predictions(predictions: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregates model probabilities using a trimmed mean to reduce outliers.
    Returns aggregate probability and per-model mean for debug.
    """
    probs = np.array(list(predictions.values()), dtype="float32")
    if probs.size == 0:
        return {"ensemble": 0.0, "mean": 0.0}

    if probs.size <= 2:
        return {"ensemble": float(np.mean(probs)), "mean": float(np.mean(probs))}

    probs_sorted = np.sort(probs)
    trimmed = probs_sorted[1:-1]  # trim min & max
    ensemble_prob = float(np.mean(trimmed))

    return {"ensemble": ensemble_prob, "mean": float(np.mean(probs))}


def predict_deepfake(image_path: str, models: Dict[str, tf.keras.Model]) -> Dict[str, Dict[str, float]]:
    """
    Returns a dictionary with per-model predictions and an ensemble decision.
    The output includes:
      - label: "FAKE" or "REAL"
      - confidence: probability of FAKE (0..1)
    """
    results: Dict[str, Dict[str, float]] = {}
    per_model_probs: Dict[str, float] = {}

    # Empirically chosen threshold for higher precision on FAKE detection
    # You can tune this based on your validation set or environment.
    FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.6"))

    for name, model in models.items():
        try:
            input_size = MODEL_INPUT_SIZES.get(name, (224, 224))
            preprocess_mode = os.getenv(f"PREPROCESS_{name}", MODEL_PREPROCESSING.get(name, "-1_1"))
            output_polarity = os.getenv(f"OUTPUT_{name}", MODEL_OUTPUT_POLARITY.get(name, "fake"))
            prob_fake = _predict_single_model(
                model,
                image_path,
                input_size,
                preprocess_mode,
                output_polarity,
            )
            per_model_probs[name] = prob_fake

            label = "FAKE" if prob_fake >= FAKE_THRESHOLD else "REAL"
            results[name] = {
                "label": label,
                "confidence": prob_fake,
            }
        except Exception as e:
            print(f"Prediction error on {name}: {e}")

    # Ensemble aggregation
    agg = _aggregate_predictions(per_model_probs)
    ensemble_prob = agg["ensemble"]
    ensemble_label = "FAKE" if ensemble_prob >= FAKE_THRESHOLD else "REAL"

    results["ENSEMBLE"] = {
        "label": ensemble_label,
        "confidence": ensemble_prob,
    }

    return results
# import os
# from typing import Dict, Tuple

# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.preprocessing.image import img_to_array

# # Import your model architectures
# from models import ResNet50, EfficientNetB4, InceptionV3, MobileNetV2, Xception

# # ================= MODEL CONFIG ================= #

# MODEL_BUILDERS = {
#     "Xception": Xception.build_model,
#     "EfficientNetB4": EfficientNetB4.build_model,
#     "InceptionV3": InceptionV3.build_model,
#     "ResNet50": ResNet50.build_model,
#     "MobileNetV2": MobileNetV2.build_model,
# }

# MODEL_INPUT_SIZES: Dict[str, Tuple[int, int]] = {
#     "Xception": (299, 299),
#     "InceptionV3": (299, 299),
#     "EfficientNetB4": (224, 224),
#     "ResNet50": (224, 224),
#     "MobileNetV2": (224, 224),
# }

# MODEL_PREPROCESSING: Dict[str, str] = {
#     "Xception": "-1_1",
#     "InceptionV3": "-1_1",
#     "EfficientNetB4": "0_1",
#     "ResNet50": "0_1",
#     "MobileNetV2": "0_1",
# }

# MODEL_OUTPUT_POLARITY: Dict[str, str] = {
#     "Xception": "fake",
#     "InceptionV3": "fake",
#     "EfficientNetB4": "fake",
#     "ResNet50": "fake",
#     "MobileNetV2": "fake",
# }

# MODEL_WEIGHTS: Dict[str, float] = {
#     "Xception": 1.2,
#     "EfficientNetB4": 1.1,
#     "InceptionV3": 1.0,
#     "ResNet50": 1.0,
#     "MobileNetV2": 0.9,
# }

# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "trained_models")

# FACE_CASCADE = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # ================= IMAGE VALIDATION ================= #

# def _is_low_information_image(img: np.ndarray) -> bool:
#     """Detects blank or near-blank images."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return gray.std() < 5  # Very low texture/detail


# def _detect_and_crop_face(image_path: str):
#     img_bgr = cv2.imread(image_path)
#     if img_bgr is None:
#         raise ValueError(f"Could not read image: {image_path}")

#     if _is_low_information_image(img_bgr):
#         return None, "LOW_INFORMATION"

#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     if len(faces) == 0:
#         return None, "NO_FACE"

#     x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
#     face_crop = img_bgr[y:y + h, x:x + w]

#     if _is_low_information_image(face_crop):
#         return None, "LOW_INFORMATION"

#     return face_crop, None

# # ================= PREPROCESS ================= #

# def _preprocess_image_from_array(img_bgr: np.ndarray, target_size: Tuple[int, int], mode: str) -> np.ndarray:
#     img_resized = cv2.resize(img_bgr, target_size)
#     img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
#     img_array = img_to_array(img_rgb).astype("float32")

#     if mode == "-1_1":
#         img_array = (img_array / 127.5) - 1.0
#     else:
#         img_array = img_array / 255.0

#     return np.expand_dims(img_array, axis=0)

# # ================= MODEL LOADING ================= #

# def load_all_models_once() -> Dict[str, tf.keras.Model]:
#     loaded_models: Dict[str, tf.keras.Model] = {}
#     print("\n--- 🧠 Loading Forensic Engines ---")

#     for name, builder in MODEL_BUILDERS.items():
#         weight_path = os.path.join(MODEL_DIR, f"{name}.weights.h5")
#         if not os.path.exists(weight_path):
#             continue

#         try:
#             input_size = MODEL_INPUT_SIZES.get(name, (224, 224))
#             model = builder(input_shape=(input_size[0], input_size[1], 3))
#             model.load_weights(weight_path, skip_mismatch=False)
#             loaded_models[name] = model
#             print(f"✅ Ready: {name}")
#         except Exception as e:
#             print(f"⚠️ {name} load failed: {e}. Attempting fallback...")
#             try:
#                 model.load_weights(weight_path, skip_mismatch=True)
#                 loaded_models[name] = model
#                 print(f"✅ Ready (fallback): {name}")
#             except Exception as fallback_error:
#                 print(f"❌ {name} fallback failed: {fallback_error}")

#     return loaded_models

# # ================= PREDICTION ================= #

# def _sigmoid(x: float) -> float:
#     return float(1.0 / (1.0 + np.exp(-x)))


# def _predict_single_model(model, face_img, input_size, preprocess_mode, output_polarity):
#     img_tensor = _preprocess_image_from_array(face_img, input_size, preprocess_mode)
#     raw = float(model.predict(img_tensor, verbose=0)[0][0])

#     prob = _sigmoid(raw) if (raw < 0.0 or raw > 1.0) else raw
#     prob = float(np.clip(prob, 0.0, 1.0))

#     if output_polarity == "real":
#         prob = 1.0 - prob

#     return float(np.clip(prob, 0.0, 1.0))

# # ================= ENSEMBLE ================= #

# def _aggregate_predictions(predictions: Dict[str, float]) -> Dict[str, float]:
#     names = list(predictions.keys())
#     probs = np.array([predictions[n] for n in names], dtype="float32")
#     weights = np.array([MODEL_WEIGHTS.get(n, 1.0) for n in names], dtype="float32")

#     if len(probs) > 2:
#         sorted_idx = np.argsort(probs)
#         probs = probs[sorted_idx][1:-1]
#         weights = weights[sorted_idx][1:-1]

#     return {
#         "ensemble": float(np.average(probs, weights=weights)),
#         "mean": float(np.mean(probs)),
#         "std": float(np.std(probs)),
#     }

# # ================= MAIN ================= #

# def predict_deepfake(image_path: str, models: Dict[str, tf.keras.Model]) -> Dict[str, Dict[str, float]]:
#     FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.6"))
#     UNCERTAINTY_STD_THRESHOLD = float(os.getenv("UNCERTAINTY_STD_THRESHOLD", "0.25"))

#     face_img, error_type = _detect_and_crop_face(image_path)

#     # 🚨 EARLY EXIT CONDITIONS
#     if error_type == "LOW_INFORMATION":
#         return {"ENSEMBLE": {"label": "INVALID_IMAGE", "confidence": 0.0, "agreement_std": 0.0}}

#     if error_type == "NO_FACE":
#         return {"ENSEMBLE": {"label": "NO_FACE_DETECTED", "confidence": 0.0, "agreement_std": 0.0}}

#     results: Dict[str, Dict[str, float]] = {}
#     per_model_probs: Dict[str, float] = {}

#     for name, model in models.items():
#         try:
#             input_size = MODEL_INPUT_SIZES[name]
#             preprocess_mode = os.getenv(f"PREPROCESS_{name}", MODEL_PREPROCESSING[name])
#             output_polarity = os.getenv(f"OUTPUT_{name}", MODEL_OUTPUT_POLARITY[name])

#             prob_fake = _predict_single_model(model, face_img, input_size, preprocess_mode, output_polarity)
#             per_model_probs[name] = prob_fake

#             results[name] = {
#                 "label": "FAKE" if prob_fake >= FAKE_THRESHOLD else "REAL",
#                 "confidence": prob_fake,
#             }
#         except Exception as e:
#             print(f"Prediction error on {name}: {e}")

#     agg = _aggregate_predictions(per_model_probs)
#     ensemble_prob = agg["ensemble"]
#     std_dev = agg["std"]

#     if std_dev > UNCERTAINTY_STD_THRESHOLD:
#         ensemble_label = "UNCERTAIN"
#     else:
#         ensemble_label = "FAKE" if ensemble_prob >= FAKE_THRESHOLD else "REAL"

#     results["ENSEMBLE"] = {
#         "label": ensemble_label,
#         "confidence": ensemble_prob,
#         "agreement_std": std_dev,
#     }

#     return results
