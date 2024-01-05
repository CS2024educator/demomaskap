import streamlit as st
import tensorflowjs as tfjs
import numpy as np

# Load the Teachable Machine model using TensorFlow.js
model_path = ''  # Replace this with the path to your model directory
model = tfjs.converters.load_keras_model(model_path)

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Add your preprocessing steps here (e.g., resizing, normalizing)
    return image

# Function to make predictions using the loaded model
def predict(image):
    processed_image = preprocess_image(image)
    # Perform prediction using the loaded model
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    return predictions

# Streamlit app UI
st.title('Teachable Machine Image Classifier')
st.write('Upload an image for classification')

# File uploader to get user input image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make predictions when the user clicks the button
    if st.button('Predict'):
        # Perform prediction
        predictions = predict(image)
        # Display predictions (customize based on your model output)
        st.write(f'Predictions: {predictions}')
