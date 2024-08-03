import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Define the web app layout
st.title(":red[MNIST Digit Recognizer]")
st.header("Upload Image")

uploaded_file = st.file_uploader("Upload an image of a digit (0-9)", type=["jpg", "png", 'webp'])

if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = image.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img)  # Convert to numpy array
    img = img / 255.0  # Normalize to [0, 1] range
    img = img.reshape(1, 28, 28)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.balloons()
    # Display the results
    st.image(uploaded_file, caption="Uploaded Image")
    st.write('-------------------------------------------------------------')
    st.write('-------------------------------------------------------------')
    st.write("Predicted Digit:", digit)


