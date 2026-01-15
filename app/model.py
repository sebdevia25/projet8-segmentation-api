import os
import requests
import tensorflow as tf

MODEL_PATH = "model/segmentation.keras"
MODEL_URL = "https://huggingface.co/Jabb/projet8-segmentation-model/resolve/main/segmentation.keras"

def download_model_if_needed():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, timeout=120)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

def load_model():
    download_model_if_needed()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

MODEL = load_model()  # charge le modèle au démarrage
