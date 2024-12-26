from flask import Flask, request, jsonify, render_template
from PIL import Image
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')

def imageArray(image):
    
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
    
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    image = Image.open(file)

    image_array = imageArray(image)

    prediction = model.predict(image_array)
    prediction = np.argmax(prediction, axis=1)[0]
    classes = pd.read_csv('classes.csv')
    prediction = classes['class'][prediction]
    
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)