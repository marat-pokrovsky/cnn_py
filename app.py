import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

from model import create_model

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = "cnn_model.h5"
model = None

if os.path.exists(MODEL_PATH):
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Trained model not found. Creating a dummy model.")
    # This is a fallback for testing the server without a trained model.
    # Replace num_classes with the actual number of classes in your dataset.
    model = create_model(num_classes=10) 
    # You might want to compile it as well if you do something with it.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def preprocess_image(image_bytes):
    """
    Preprocesses the image for the CNN model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint. Expects an image file in the request.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    
    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
