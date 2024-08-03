import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Define the web app layout
st.title(":red[MNIST Digit Recognizer]")
st.write('[click to get Sample Images of digits](https://www.google.com/imgres?q=mnist%20images&imgurl=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F321174607%2Ffigure%2Ffig3%2FAS%3A806993333850113%401569413612260%2FExample-of-a-MNIST-input-An-image-is-passed-to-the-network-as-a-matrix-of-28-by-28.png&imgrefurl=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FExample-of-a-MNIST-input-An-image-is-passed-to-the-network-as-a-matrix-of-28-by-28_fig3_321174607&docid=SscvN6rlLtzgyM&tbnid=yEDu2ks1PlxS8M&vet=12ahUKEwip7-OfsdiHAxUwyDgGHRYECmAQM3oECCoQAA..i&w=176&h=176&hcb=2&ved=2ahUKEwip7-OfsdiHAxUwyDgGHRYECmAQM3oECCoQAA)')
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


