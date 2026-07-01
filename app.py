import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="PoultryAI", layout="wide")

model = load_model("poultry_model.keras")

class_names = ['Coccidiosis', 'Healthy', 'Newcastle_Disease', 'Salmonella']

st.title("PoultryAI: Real-Time Poultry Disease Classification System")
st.markdown("### EfficientNetB0 Based Disease Detection")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Poultry Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Input Image", use_container_width=True)

with col2:
    if uploaded_file:
        img = img.resize((224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Run Analysis"):
            prediction = model.predict(img_array)
            result = class_names[np.argmax(prediction)]

            st.subheader("Prediction Result")
            st.success(result)

st.markdown("---")
st.write("Department of CSE (AI) | DSATM")