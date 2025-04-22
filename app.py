import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Face Mask Detection App")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Mask_Model.h5")  # Update filename if needed
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (adjust according to your model's input size)
    img = image.resize((124, 124))  # Resize to match training data
    img_array = np.array(img)
    img_array = img_array / 255.0   # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    # Prediction
    prediction = model.predict(img_array)

    # Assuming binary classification [Mask, No Mask]
    label = "Mask" if prediction[0][0] > 0.03 else "No Mask"
    confidence = 1 - prediction[0][0] if label == "Mask" else prediction[0][0]

    st.subheader("Prediction")
    print(prediction)
    st.write(f"Label: {label}")

    # streamlit run app.py  command for opening interface