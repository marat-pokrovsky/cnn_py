import numpy as np
from PIL import Image
import os

def load_data(data_dir):
    """
    Loads data from the given directory.
    Assumes that the directory contains subdirectories for each class,
    and each subdirectory contains images for that class.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = Image.open(img_path).convert('L') # Convert to grayscale
                    img = img.resize((28, 28)) # Resize to 28x28
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(i)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

def preprocess_data(images):
    """
    Preprocesses the image data.
    """
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, -1)
    return images
