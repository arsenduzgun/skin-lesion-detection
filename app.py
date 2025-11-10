from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import numpy as np
import json
import os
import tensorflow as tf

app = Flask(__name__)

# --- Paths (env-overridable) ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/HAM10000_efficientnetB0.keras")
META_PATH  = os.getenv("META_PATH",  "models/HAM10000_meta.json")

# --- Load model + meta ---
model = tf.keras.models.load_model(MODEL_PATH)

with open(META_PATH, "r") as f:
    meta = json.load(f)

IMG_SIZE     = int(meta.get("img_size", 256))
CLASS_NAMES  = meta.get("class_names", ["akiec","bcc","bkl","df","mel","nv","vasc"])
THRESHOLDS   = np.array(meta.get("thresholds", [0.5]*len(CLASS_NAMES)), dtype=np.float32)

# --- Preprocess must MATCH training ---
# Your saved model expects inputs in [0,1] because inside the graph you have:
#   x = preprocess_input(inputs * 255.0)
# So here we: center-crop square -> resize -> scale to [0,1] -> float32
def preprocess_image(image: Image.Image) -> np.ndarray:
    # honor EXIF orientation & ensure RGB
    image = ImageOps.exif_transpose(image).convert("RGB")

    # center-crop to square
    w, h = image.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))

    # resize to model’s expected square
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # to [0,1] float32, add batch dim
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def apply_thresholds(probs_row: np.ndarray, thresholds: np.ndarray) -> int:
    """
    Multi-class thresholding with argmax fallback.
    - If one or more classes pass their threshold, pick the passing class with highest prob.
    - If none pass, return argmax.
    """
    passing = np.where(probs_row >= thresholds)[0]
    if passing.size > 0:
        # choose the passing class with highest probability
        return passing[np.argmax(probs_row[passing])]
    return int(np.argmax(probs_row))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field name 'image'."}), 400

    file = request.files["image"]
    image = Image.open(file)
    x = preprocess_image(image)  # (1, IMG, IMG, 3) in [0,1]

    # model outputs probs because final layer is softmax
    probs = model.predict(x, verbose=0)[0]  # shape (7,)
    idx_thresholded = apply_thresholds(probs, THRESHOLDS)
    label_thresholded = CLASS_NAMES[idx_thresholded]

    # also provide plain argmax for reference
    idx_argmax = int(np.argmax(probs))
    label_argmax = CLASS_NAMES[idx_argmax]

    # top-3 for UI
    top3_idx = probs.argsort()[-3:][::-1].tolist()
    top3 = [
        {"label": CLASS_NAMES[i], "prob": float(probs[i]), "threshold": float(THRESHOLDS[i])}
        for i in top3_idx
    ]

    return jsonify({
        "prediction": label_thresholded,
        "prediction_index": int(idx_thresholded),
        "used_thresholds": True,
        "argmax_label": label_argmax,
        "probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "thresholds": {CLASS_NAMES[i]: float(t) for i, t in enumerate(THRESHOLDS)},
        "top3": top3,
        "img_size": IMG_SIZE
    })

if __name__ == "__main__":
    # optional: make TF quieter / GPU memory growth
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    app.run(host="0.0.0.0", port=5000, debug=True)
