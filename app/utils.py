import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

# même nombre de classes que pendant ton entrainement
NUM_CLASSES = 8

# palette de couleurs fixe (lisible)
PALETTE = np.array([
    [0, 0, 0],         # void
    [128, 64, 128],    # flat
    [220, 20, 60],     # human
    [0, 0, 142],       # vehicule
    [70, 70, 70],      # construction
    [153, 153, 153],   # object
    [107, 142, 35],    # nature
    [70, 130, 180],    # sky
], dtype=np.uint8)

IMG_H = 256
IMG_W = 512


def preprocess_image(img_bytes):
    """ Charge l'image, la convertit en RGB, la resize """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_W, IMG_H))
    img = np.array(img) / 255.0
    return img.astype(np.float32), img  # (entrée réseau, version RGB pour affichage)


def colorize_mask(mask):
    """ mask : (H, W) avec valeurs 0..7 """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = PALETTE[c]
    return out


def encode_image_to_base64(img):
    """ Convertit une image numpy en base64 """
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
