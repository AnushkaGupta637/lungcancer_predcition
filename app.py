import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('LungcancerPrediction.h5')  

st.title("Lung Cancer Image Classification")
st.write("Upload a lung X-ray or CT scan image to predict lung condition (Cancer or Normal).")

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        st.write("Predicting...")

        # Preprocess the image
        img = img.resize((256, 256))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        prediction = model.predict(img_array)[0][0]

        pred_label = 'Cancer detected' if prediction > 0.5 else 'No Lung Cancer detected'

        st.success(f"Prediction: {pred_label}")
