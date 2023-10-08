import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load class mapping from JSON file
class_mapping = {}  # Initialize an empty dictionary
try:
    with open("classes.json", "r") as json_file:
        class_mapping = json.load(json_file)
except FileNotFoundError:
    st.error("Error: classes.json not found. Make sure the file exists.")

# Load the model architecture from the JSON file
try:
    with open("model_architecture.json", "r") as json_file:
        model_json = json_file.read()
except FileNotFoundError:
    st.error("Error: model_architecture.json not found. Make sure the file exists.")

# Load the model weights from the HDF5 file
try:
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("model_weights.h5")
except (FileNotFoundError, tf.errors.NotFoundError):
    st.error("Error: Model files not found. Make sure the model and weights files exist.")

# Define a function to make predictions
def predict_image(image):
    image = np.asarray(image)
    image = tf.image.resize(image, (331, 331))  # Resize to match model input shape
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

# Create the Streamlit app
st.title('Dog Breed Identifier App')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(image)
        predicted_class_index = np.argmax(prediction)
        
        # Use the class mapping to get the predicted class label
        predicted_class_label = class_mapping.get(str(predicted_class_index), 'Unknown')
        
        # Style the result text with larger font size and bold
        result_text = f'<p style="font-size:24px; font-weight:bold;">Result   :   {predicted_class_label}</p>'
        
        st.markdown(result_text, unsafe_allow_html=True)  # Render HTML
