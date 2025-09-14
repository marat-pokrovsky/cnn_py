import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

from model import create_model, create_another_model

app = Flask(__name__)

# --- Model Loading ---
models = {}

# Model 1
MODEL_PATH = "cnn_model.h5"
if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    models["mnist_model"] = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Trained model not found. Creating a dummy model.")
    model = create_model(num_classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    models["mnist_model"] = model

# Model 2
ANOTHER_MODEL_PATH = "another_cnn_model.h5"
if os.path.exists(ANOTHER_MODEL_PATH):
    print("Loading another trained model...")
    models["vgg_model"] = tf.keras.models.load_model(ANOTHER_MODEL_PATH)
else:
    print("Another trained model not found. Creating a dummy model.")
    another_model = create_another_model(num_classes=10)
    another_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    models["vgg_model"] = another_model

def preprocess_image(image_bytes, model_name):
    """
    Preprocesses the image for the selected CNN model.
    """
    if model_name == "mnist_model":
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        return img_array
    elif model_name == "vgg_model":
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    else:
        raise ValueError("Unknown model name")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint. Expects an image file and a model name.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    model_name = request.form.get("model")
    if not model_name or model_name not in models:
        return jsonify({'error': 'invalid model name provided'}), 400

    file = request.files['file']
    selected_model = models[model_name]

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes, model_name)
        prediction = selected_model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/game2')
def game2():
    return render_template('game2.html')

@app.route('/predict_game', methods=['POST'])
def predict_game():
    """
    Prediction endpoint for the digit recognition game.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    model_name = "mnist_model"
    selected_model = models[model_name]

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes, model_name)
        prediction = selected_model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify_digit', methods=['POST'])
def verify_digit():
    """
    Verification endpoint for the digit drawing challenge.
    """
    if 'file' not in request.files or 'digit' not in request.form:
        return jsonify({'error': 'file or digit not provided'}), 400

    file = request.files['file']
    target_digit = int(request.form['digit'])
    model_name = "mnist_model"
    selected_model = models[model_name]

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes, model_name)
        prediction = selected_model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        if int(predicted_class) == target_digit:
            return jsonify({'correct': True})
        else:
            return jsonify({'correct': False, 'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
