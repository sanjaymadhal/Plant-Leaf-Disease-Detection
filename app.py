import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model once and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')  # Update with correct path
    return model


# Class labels (Modify based on your dataset)
CLASS_NAMES = ['Potato__Healthy', 'Potato_Early_blight', 'Potato__Late_blight']

# Streamlit UI
def main():
    st.title("🍂 Potato Leaf Disease Detector")
    st.write("Upload an image of a potato leaf to detect diseases.")

    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB format
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict Disease"):
            if load_model is not None:
                result, confidence = predict(image)
                st.success(f"🩺 **Prediction:** {result} ({confidence:.2f}% Confidence)")
                
                # Display confidence as a progress bar
                st.progress(int(confidence))
            else:
                st.error("⚠️ Model not loaded. Please check the model file.")

# Image Preprocessing and Prediction
def predict(image):
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Resize image to match model input size (128x128)
        img_resized = cv2.resize(img_array, (128, 128))

        # Normalize pixel values to range [0,1]
        img_resized = img_resized / 255.0

        # Expand dimensions to create a batch of size 1
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Make a prediction
        predictions = load_model.predict(img_expanded)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        return predicted_class, confidence
    except Exception as e:
        return f"Error during prediction: {e}", 0

if __name__ == "__main__":
    main()
