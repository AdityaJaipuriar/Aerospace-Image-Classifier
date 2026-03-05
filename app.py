'''import streamlit as st
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
    st.info("Please upload an aircraft image to begin classification.")'''

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception, EfficientNetB0, ConvNeXtTiny
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Aerospace Ensemble Classifier", layout="wide")

# --- 2. ARCHITECTURE RECONSTRUCTION ---
def build_expert_skeleton(model_type):
    """
    Manually defines the architecture to force a single input stream.
    This fixes the '2 input tensors' error by explicitly wiring the GAP layer.
    """
    if model_type == 'xcp':
        base = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))
    elif model_type == 'eff':
        base = EfficientNetB0(weights=None, include_top=False, input_shape=(299, 299, 3))
    elif model_type == 'conv':
        base = ConvNeXtTiny(weights=None, include_top=False, input_shape=(299, 299, 3))
    
    # FORCE SINGLE INPUT: Pull only the first tensor if multiple are detected
    base_output = base.output
    if isinstance(base_output, list):
        base_output = base_output[0]

    x = layers.GlobalAveragePooling2D()(base_output)
    
    # These names MUST match the 'dense' and 'dense_1' from your training
    x = layers.Dense(128, activation='relu', name='dense')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(5, activation='softmax', name='dense_1')(x)
    
    return models.Model(inputs=base.input, outputs=outputs)

# --- 3. CACHED ASSET LOADING ---
@st.cache_resource
def load_experts():
    try:
        # Clear session to stay under Streamlit's 1GB RAM limit
        tf.keras.backend.clear_session()

        # Rebuild local skeletons
        m1 = build_expert_skeleton('xcp')
        m2 = build_expert_skeleton('eff')
        m3 = build_expert_skeleton('conv')

        # Load weights from your .keras files (Keras 3 handles this internally)
        # Ensure these filenames match your GitHub exactly
        m1.load_weights('xception_clean.keras')
        m2.load_weights('efficientnet_clean.keras')
        m3.load_weights('convnext_clean.keras')

        with open('class_indices.json', 'r') as f:
            class_map = json.load(f)
        
        labels = {int(v): k for k, v in class_map.items()}
        return m1, m2, m3, labels
    
    except Exception as e:
        st.error(f"Inference Initialization Error: {e}")
        return None, None, None, None

m_xcp, m_eff, m_conv, class_labels = load_experts()

# --- 4. MAIN INTERFACE ---
st.title("✈️ Aerospace Image Classifier")
st.markdown("Automated aircraft identification via **Triple-Expert Neural Ensemble**.")
st.divider()

if m_xcp is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload Aircraft Imagery")
        uploaded_file = st.file_uploader("Drop image here (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_container_width=True, caption="Target Specimen")

    with col2:
        st.subheader("Ensemble Inference")
        if uploaded_file:
            with st.spinner('Calculating neural features...'):
                img_resized = img.resize((299, 299))
                img_array = np.array(img_resized).astype('float32')
                
                # Preprocessing paths from your Kaggle logic
                in_xcp = np.expand_dims(img_array / 255.0, axis=0) 
                in_eff = np.expand_dims(eff_preprocess(img_array.copy()), axis=0)
                in_conv = np.expand_dims(img_array, axis=0) 

                # Prediction with .numpy() for memory efficiency
                p1 = m_xcp(in_xcp, training=False).numpy()
                p2 = m_eff(in_eff, training=False).numpy()
                p3 = m_conv(in_conv, training=False).numpy()

                # Weighted Ensemble (20/40/40) for 0.87 accuracy
                final_probs = (0.2 * p1) + (0.4 * p2) + (0.4 * p3)
                idx = np.argmax(final_probs[0])
                confidence = np.max(final_probs[0])

                st.success(f"### Predicted: **{class_labels[idx].upper()}**")
                st.write(f"Aggregate Confidence: **{confidence*100:.2f}%**")
                st.progress(float(confidence))
else:
    st.error("Assets offline. Ensure your .keras files are properly tracked via Git LFS.")