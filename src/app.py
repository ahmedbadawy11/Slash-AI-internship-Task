import streamlit as st
from PIL import Image
from model.YOLO_v8 import get_image_components


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
    if st.button('Analyse Image'):
        with st.spinner('Analyzing...'):
            # Analyse Image using YOLOv8 
            detected_classes=get_image_components(image)
            
            # Display the results
            st.write("Detected components:")
            if detected_classes:
                st.write(detected_classes)
            else:
                st.write("No components detected.")