import pickle
import tensorflow as tf
import cv2
from tensorflow import keras
import streamlit as st
import numpy as np

# Load the trained model
file = 'mnist_model.sav'
model = pickle.load(open(file,'rb'))

# Define the web app layout

st.title("MNIST Digit Recognizer")
st.sidebar.title("Upload Image")

uploaded_file = st.sidebar.file_uploader("Upload an image of a digit (0-9)", type=["jpg", "png"])

if uploaded_file is not None:
    image = cv2.imread(uploaded_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    
    # Make predictions
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted Digit:", predicted_digit)