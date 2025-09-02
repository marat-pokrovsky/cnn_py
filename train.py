import tensorflow as tf
from data_loader import load_data, preprocess_data
from model import create_model

# Load and preprocess data
data_dir = "data"
images, labels = load_data(data_dir)
images = preprocess_data(images)

# Get the number of classes from the data
num_classes = len(set(labels))

# Create the model
model = create_model(num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save the trained model
model.save("cnn_model.h5")
