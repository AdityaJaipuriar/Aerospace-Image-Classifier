import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# Page config, CSS, title, sidebar 
st.set_page_config(page_title="Aerospace Classifier", layout="wide")

# Model loading (keep this early, it's cached) 
@st.cache_resource(show_spinner="Loading models...")
def load_experts():
    try:
        model_xcp  = tf.keras.models.load_model('xception_clean.keras',  compile=False)
        model_eff  = tf.keras.models.load_model('efficientnet_clean.keras', compile=False)
        model_conv = tf.keras.models.load_model('convnext_clean.keras',  compile=False)

        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)

        labels = {v: k for k, v in class_indices.items()}  # index → name

        st.success("Models loaded successfully ✓")  # temporary debug
        return model_xcp, model_eff, model_conv, labels

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, None

model_xcp, model_eff, model_conv, class_labels = load_experts()

if model_xcp is None:
    st.stop()

st.subheader("Upload Aircraft Image")
uploaded_file = st.file_uploader(
    "Choose an image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

# Only now use uploaded_file 
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify button
    if st.button("Classify Aircraft", type="primary"):
        with st.spinner("Running ensemble prediction..."):
            img_resized = image.resize((299, 299))
            img_array = np.array(img_resized).astype('float32')
            img_batch = np.expand_dims(img_array, axis=0)  # (1, 299, 299, 3)

            # Predictions with correct preprocessing
            pred_xcp  = model_xcp.predict(img_batch / 255.0, verbose=0)
            pred_eff  = model_eff.predict(eff_preprocess(img_batch), verbose=0)
            pred_conv = model_conv.predict(img_batch / 255.0, verbose=0)

            final_pred = 0.2 * pred_xcp + 0.4 * pred_eff + 0.4 * pred_conv
            pred_idx   = np.argmax(final_pred[0])
            confidence = final_pred[0][pred_idx]

            predicted_class = class_labels.get(pred_idx, "Unknown")

            st.success(f"**Predicted:** {predicted_class}")
            st.metric("Confidence", f"{confidence:.1%}")

            # Optional: show probabilities
            probs = final_pred[0] * 100
            st.bar_chart(
                {class_labels[i]: p for i, p in enumerate(probs)},
                x_label="Class",
                y_label="Probability (%)"
            )

else:
    st.info("Please upload an aircraft image to begin classification.")