import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Streamlit application
st.title("Image Component Detection with YOLO v8")
st.write("Upload an image and click 'Analyse Image' to detect components.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Analyze button
     # Analyze button
    if st.button('Analyse Image'):
        with st.spinner('Analyzing...'):
            # Perform detection
            results = model(image)
            # Extract predicted classes
            detected_classes = set()
            for result in results:
                for pred in result.boxes.data.tolist():
                    class_id = int(pred[5])
                    detected_classes.add(result.names[class_id])
            
            detected_classes = list(detected_classes)
            
            # Display the results
            st.write("Detected components:")
            if detected_classes:
                st.write(detected_classes)
            else:
                st.write("No components detected.")