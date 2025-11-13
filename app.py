import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
import traceback

# ---------------------------
# CONFIG
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = "models/best_brain_tumor_model.h5"
IMG_SIZE = (224, 224)
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# ---------------------------
# Load and Build Model Properly
# ---------------------------
st.title("🧠 Brain Tumor MRI Classifier")

@st.cache_resource
def load_and_build_model():
    try:
        model = load_model(MODEL_PATH)
        
        # Ensure the model is built by creating a dummy input with the correct shape
        dummy_input = tf.keras.Input(shape=(*IMG_SIZE, 3))
        
        # For Sequential models, we need to rebuild them properly
        if isinstance(model, tf.keras.Sequential):
            st.sidebar.info("Model is Sequential - rebuilding with proper input shape")
            # Rebuild the sequential model with proper input shape
            rebuilt_model = tf.keras.Sequential()
            rebuilt_model.add(tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3)))
            
            # Add all layers from the original model
            for layer in model.layers:
                rebuilt_model.add(layer)
            
            # Rebuild the model
            rebuilt_model.build(input_shape=(None, *IMG_SIZE, 3))
            model = rebuilt_model
        
        # Test the model with a dummy prediction to ensure it's built
        dummy_batch = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        _ = model.predict(dummy_batch, verbose=0)
        
        st.sidebar.success("✅ Model loaded and built successfully")
        return model
        
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}")
        st.exception(e)
        st.stop()

try:
    model = load_and_build_model()
except Exception as e:
    st.error("Failed to load or initialize the model. See details below.")
    st.exception(e)
    st.stop()

# ---------------------------
# Model Diagnostics
# ---------------------------
st.sidebar.subheader("Model Diagnostics")

def diagnose_model(model):
    """Comprehensive model analysis"""
    st.sidebar.write("**Model Type:**", type(model).__name__)
    
    # Get input shape
    try:
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
        else:
            input_shape = "N/A"
    except:
        input_shape = "N/A"
    
    # Get output shape  
    try:
        if hasattr(model, 'output_shape'):
            output_shape = model.output_shape
        else:
            output_shape = "N/A"
    except:
        output_shape = "N/A"
    
    st.sidebar.write("**Input Shape:**", input_shape)
    st.sidebar.write("**Output Shape:**", output_shape)
    st.sidebar.write("**Model Built:**", getattr(model, 'built', 'N/A'))

    # Analyze layers
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D) or 'conv' in getattr(layer, 'name', '').lower():
            conv_layers.append({
                'index': i,
                'name': getattr(layer, 'name', str(i)),
                'type': type(layer).__name__,
                'built': getattr(layer, 'built', False),
                'output_shape': getattr(layer, 'output_shape', 'N/A')
            })

    st.sidebar.write(f"**Convolutional Layers Found:** {len(conv_layers)}")
    return conv_layers

conv_layers = diagnose_model(model)

# ---------------------------
# FIXED GRAD-CAM IMPLEMENTATION
# ---------------------------

def build_gradcam_model(model, layer_name=None):
    """Build a Grad-CAM model that will definitely work"""
    try:
        # If no specific layer provided, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, Conv2D):
                    target_layer = layer
                    break
            else:
                # If no Conv2D found, try to find any layer with 'conv' in name
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        target_layer = layer
                        break
                else:
                    st.warning("No convolutional layers found for Grad-CAM")
                    return None, None
        else:
            # Find layer by name
            target_layer = None
            for layer in model.layers:
                if layer.name == layer_name:
                    target_layer = layer
                    break
            if target_layer is None:
                st.warning(f"Layer {layer_name} not found")
                return None, None

        st.sidebar.info(f"Using layer: {target_layer.name}")

        # Create a model that outputs the target layer's activations and the final predictions
        grad_model = Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        
        return grad_model, target_layer
        
    except Exception as e:
        st.error(f"Error building Grad-CAM model: {str(e)}")
        return None, None

def simple_gradcam(model, img_array, pred_index=None):
    """Working Grad-CAM implementation"""
    try:
        # Build the Grad-CAM model
        grad_model, target_layer = build_gradcam_model(model)
        if grad_model is None:
            return None

        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            st.warning("Gradients are None - cannot compute Grad-CAM")
            return None

        # Global average pooling of gradients
        guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * guided_grads, axis=-1)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap.numpy()

    except Exception as e:
        st.error(f"Grad-CAM failed: {str(e)}")
        return None

def direct_gradcam_approach(model, img_array, pred_index=None):
    """Alternative direct approach that avoids model rebuilding"""
    try:
        # Find the last convolutional layer
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                conv_layer = layer
                break
        
        if conv_layer is None:
            st.warning("No Conv2D layer found")
            return None

        # Create a model that goes from input to the convolutional layer output
        conv_output_model = Model(inputs=model.input, outputs=conv_layer.output)
        
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            # Get convolutional outputs
            conv_outputs = conv_output_model(img_tensor)
            
            # Get predictions
            predictions = model(img_tensor)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            target_output = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(target_output, conv_outputs)
        if grads is None:
            return None
            
        # Pool gradients and create heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"Direct Grad-CAM approach failed: {str(e)}")
        return None

def alternative_heatmap(img_array):
    """Fallback heatmap based on image intensity"""
    try:
        src = img_array[0]
        if src.dtype != np.uint8:
            src_uint8 = np.clip(src * 255.0, 0, 255).astype(np.uint8)
        else:
            src_uint8 = src

        # Convert to grayscale if needed
        if src_uint8.ndim == 3:
            img_gray = cv2.cvtColor(src_uint8, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = src_uint8

        heatmap = cv2.resize(img_gray, IMG_SIZE).astype(np.float32) / 255.0
        
        # Enhance contrast
        heatmap = heatmap ** 2
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap

    except Exception as e:
        st.error(f"Alternative heatmap failed: {str(e)}")
        return None

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    if heatmap is None:
        return img

    try:
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert image to BGR for OpenCV
        if len(img.shape) == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Ensure same data type
        img_bgr = img_bgr.astype(np.uint8)
        
        # Blend images
        overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    except Exception as e:
        st.error(f"Overlay failed: {str(e)}")
        return img

# ---------------------------
# Layer Selection
# ---------------------------
st.sidebar.subheader("Grad-CAM Settings")

# Let user select which layer to use for Grad-CAM
if conv_layers:
    layer_names = [layer['name'] for layer in conv_layers]
    selected_layer = st.sidebar.selectbox(
        "Choose layer for Grad-CAM:",
        layer_names,
        index=len(layer_names)-1  # Default to last layer
    )
else:
    selected_layer = None
    st.sidebar.warning("No convolutional layers available for Grad-CAM")

# ---------------------------
# File Processing
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload MRI Images", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"Image: {uploaded_file.name}")

        try:
            # Load and preprocess image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMG_SIZE)
            img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

            # Make prediction
            pred = model.predict(img_array, verbose=0)
            pred_class = int(np.argmax(pred, axis=1)[0])
            label = CLASSES[pred_class]
            confidence = float(pred[0][pred_class] * 100)

            # Try different visualization methods
            heatmap = None
            visualization_method = "None"

            # Method 1: Try direct Grad-CAM approach
            heatmap = direct_gradcam_approach(model, img_array, pred_class)
            if heatmap is not None:
                visualization_method = "Grad-CAM"

            # Method 2: Try simple Grad-CAM
            if heatmap is None:
                heatmap = simple_gradcam(model, img_array, pred_class)
                if heatmap is not None:
                    visualization_method = "Grad-CAM (Simple)"

            # Method 3: Fallback to alternative
            if heatmap is None:
                heatmap = alternative_heatmap(img_array)
                if heatmap is not None:
                    visualization_method = "Intensity-based"

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.image(img_rgb, caption="Original MRI", use_container_width=True)

            with col2:
                if heatmap is not None:
                    overlay = overlay_heatmap(img_resized, heatmap)
                    caption = f"{visualization_method} → {label} ({confidence:.1f}%)"
                    st.image(overlay, caption=caption, use_container_width=True)
                else:
                    st.image(img_rgb, caption="No visualization available", use_container_width=True)

            # Prediction results
            st.success(f"**Prediction:** {label} (Confidence: {confidence:.2f}%)")

            # Confidence scores
            with st.expander("View detailed confidence scores"):
                for i, class_name in enumerate(CLASSES):
                    conf = float(pred[0][i] * 100)
                    st.write(f"- **{class_name}:** {conf:.2f}%")
                    st.progress(float(pred[0][i]))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}:")
            st.code(traceback.format_exc())

# ---------------------------
# Model Information
# ---------------------------
with st.sidebar.expander("Model Information"):
    st.write(f"**Type:** {type(model).__name__}")
    st.write(f"**Input Shape:** {getattr(model, 'input_shape', 'N/A')}")
    st.write(f"**Output Shape:** {getattr(model, 'output_shape', 'N/A')}")
    st.write(f"**Layers:** {len(model.layers)}")
    
    if conv_layers:
        st.write("**Convolutional Layers:**")
        for layer in conv_layers:
            st.write(f"- {layer['name']} ({layer['type']})")

with st.sidebar.expander("Troubleshooting"):
    st.write("""
    **If Grad-CAM fails:**
    1. The model will still provide predictions
    2. Alternative visualization methods will be used
    3. Check that the model has convolutional layers
    4. Ensure the model is properly built with input shape (224, 224, 3)
    """)