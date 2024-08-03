import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Define the web app layout
st.title("MNIST Digit Recognizer")
st.sidebar.title("Upload Image")

uploaded_file = st.sidebar.file_uploader("Upload an image of a digit (0-9)", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize to [0, 1] range
    image = image.reshape(1, 28, 28, 1)  # Add batch dimension

    # Make predictions
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted Digit:", predicted_digit)
