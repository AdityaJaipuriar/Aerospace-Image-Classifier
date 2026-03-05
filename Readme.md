# Aerospace Image Classification System

Automated aircraft identification from visual data using a **Triple-Expert Neural Ensemble**

## Problem Statement

Manual identification of aircraft from visual data is a critical but time-consuming task in aerospace monitoring and air traffic management.  
Challenges include:

- Distinguishing between similar aerodynamic profiles  
- Handling varying lighting conditions, angles, and backgrounds  
- Need for high-precision, automated computer vision solution

## Solution

A robust **Aerospace Image Classification System** that uses a **Triple-Expert Neural Ensemble** combining three state-of-the-art convolutional neural network architectures.  
The system provides highly accurate classification across five aircraft categories and includes a clean, real-time web interface built with **Streamlit**.

### Key Features

- Weighted soft-voting ensemble of three strong backbones  
- Real-time inference via browser-based upload  
- Optimized for low-latency CPU execution  
- Model caching for fast repeated predictions  
- Clear probability visualization

### Model Ensemble Architecture

| Model          | Weight | Backbone              | Strength                              |
|----------------|--------|-----------------------|---------------------------------------|
| Xception       | 20%    | Xception              | Excellent at capturing fine details   |
| EfficientNetB0 | 40%    | EfficientNet-B0       | Strong efficiency–accuracy trade-off  |
| ConvNeXtTiny   | 40%    | ConvNeXt-Tiny         | Modern transformer-inspired design    |

### Technical Implementation

- **Input Pipeline**  
  - Images resized to 299 × 299  
  - Model-specific preprocessing:  
    - Xception & ConvNeXtTiny → `/ 255.0` scaling  
    - EfficientNet → native `preprocess_input`

- **Inference Engine**  
  - Streamlit-based interactive dashboard  
  - `@st.cache_resource` for singleton model loading  
  - Weighted ensemble soft voting

- **Deployment Assets**  
  - Native Keras v3 `.keras` format (recommended)  
  - Legacy `.h5` weights & full models for compatibility

### Performance

- **Ensemble Accuracy**: **≥ 0.87** on the evaluation set  
- **Classes** (5-class problem):  
  - commercial  
  - glider  
  - helicopter  
  - mig-29  
  - sukhoi  
- **Inference Speed**: real-time on standard CPU

### Project Structure

```text
Aerospace-Image-Classification/
├── app.py                        # Streamlit web interface
├── aerospace-image-classification.ipynb   # Training & evaluation notebook
├── class_indices.json            # Label ↔ index mapping
├── requirements.txt              # Dependencies
├── xception_clean.keras          # Cleaned saved models
├── efficientnet_clean.keras
├── convnext_clean.keras
└── README.md