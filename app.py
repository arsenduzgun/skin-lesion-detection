from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import json
import os
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.getenv('MODEL_PATH', 'model-dev/models/model.keras')
CLASSES_PATH = os.getenv('CLASSES_PATH', 'model-dev/dataset/classes.json')

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

def preprocess_image(image: Image.Image) -> np.ndarray:
    width, height = image.size
    if width > height:
        new_width = height
        left = (width - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, height))
    elif height > width:
        new_height = width
        top = (height - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, width, bottom))
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_idx = int(np.argmax(prediction, axis=1)[0])
    predicted_label = classes[predicted_idx]
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)