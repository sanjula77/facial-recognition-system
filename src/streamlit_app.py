# src/streamlit_app.py

import streamlit as st
import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tempfile
from PIL import Image
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Facial Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_DIR = "models"
EMBEDDINGS_DIR = "embeddings"
DATA_DIR = "data"

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler with caching."""
    try:
        model_path = os.path.join(MODEL_DIR, "face_classifier.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            return None, None
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ Scaler not found: {scaler_path}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_resource
def load_face_cascade():
    """Load OpenCV face detection cascade with caching."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            st.error("❌ Failed to load face cascade")
            return None
        return cascade
    except Exception as e:
        st.error(f"❌ Error loading face cascade: {e}")
        return None

def extract_simple_features(face_image):
    """Extract the same features used during training."""
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to a standard size for feature extraction
    gray_resized = cv2.resize(gray, (64, 64))
    
    # Extract HOG-like features (same as training)
    features = []
    
    # Divide image into 8x8 blocks
    block_size = 8
    for i in range(0, 64, block_size):
        for j in range(0, 64, block_size):
            block = gray_resized[i:i+block_size, j:j+block_size]
            
            # Calculate simple statistics for each block
            mean_val = np.mean(block)
            std_val = np.std(block)
            
            features.extend([mean_val, std_val])
    
    # Add some global features
    features.append(np.mean(gray_resized))
    features.append(np.std(gray_resized))
    
    return np.array(features, dtype=np.float32)

def detect_and_crop_faces(image, cascade, target_size=(112, 112)):
    """Detect faces and crop them to target size."""
    if cascade is None:
        return [], []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    cropped_faces = []
    face_locations = []
    
    for (x, y, w, h) in faces:
        # Expand the bounding box
        expand = 0.3
        x1 = max(0, int(x - w * expand))
        y1 = max(0, int(y - h * expand))
        x2 = min(image.shape[1], int(x + w * (1 + expand)))
        y2 = min(image.shape[0], int(y + h * (1 + expand)))
        
        # Create square crop
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        crop_size = max(x2 - x1, y2 - y1)
        
        # Ensure crop size is reasonable
        crop_size = min(crop_size, min(image.shape[0], image.shape[1]))
        
        # Calculate crop boundaries
        crop_x1 = max(0, center_x - crop_size // 2)
        crop_y1 = max(0, center_y - crop_size // 2)
        crop_x2 = min(image.shape[1], crop_x1 + crop_size)
        crop_y2 = min(image.shape[0], crop_y1 + crop_size)
        
        # Adjust if we hit image boundaries
        if crop_x2 - crop_x1 < crop_size:
            if crop_x1 == 0:
                crop_x2 = min(image.shape[1], crop_x1 + crop_size)
            else:
                crop_x1 = max(0, crop_x2 - crop_size)
        
        if crop_y2 - crop_y1 < crop_size:
            if crop_y1 == 0:
                crop_y2 = min(image.shape[0], crop_y1 + crop_size)
            else:
                crop_y1 = max(0, crop_y2 - crop_size)
        
        # Create the crop
        face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
        
        cropped_faces.append(face_resized)
        face_locations.append((crop_x1, crop_y1, crop_x2, crop_y2))
    
    return cropped_faces, face_locations

def recognize_faces_in_image(image, model, scaler, cascade):
    """Recognize faces in an image."""
    try:
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect and crop faces
        cropped_faces, face_locations = detect_and_crop_faces(image, cascade)
        
        if len(cropped_faces) == 0:
            return []
        
        # Process each detected face
        results = []
        for i, (face, location) in enumerate(zip(cropped_faces, face_locations)):
            try:
                # Extract features
                features = extract_simple_features(face)
                
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled).max()
                
                results.append({
                    'face_id': i,
                    'person': prediction,
                    'confidence': confidence,
                    'location': location,
                    'face_crop': face
                })
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return []

def get_system_stats():
    """Get system statistics."""
    stats = {}
    
    # Check model files
    model_path = os.path.join(MODEL_DIR, "face_classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    stats['model_loaded'] = os.path.exists(model_path)
    stats['scaler_loaded'] = os.path.exists(scaler_path)
    
    # Check data directory
    if os.path.exists(DATA_DIR):
        person_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        stats['total_persons'] = len(person_dirs)
        stats['person_names'] = person_dirs
    else:
        stats['total_persons'] = 0
        stats['person_names'] = []
    
    # Check processed data
    processed_dir = "processed_data"
    if os.path.exists(processed_dir):
        person_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        stats['processed_persons'] = len(person_dirs)
    else:
        stats['processed_persons'] = 0
    
    return stats

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">👤 Facial Recognition System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Load model and scaler
        model, scaler = load_model_and_scaler()
        cascade = load_face_cascade()
        
        # System statistics
        stats = get_system_stats()
        
        if stats['model_loaded'] and stats['scaler_loaded']:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
        
        if cascade is not None:
            st.success("✅ Face Detector Loaded")
        else:
            st.error("❌ Face Detector Not Found")
        
        st.markdown("---")
        st.subheader("📊 System Info")
        st.write(f"**Total Persons:** {stats['total_persons']}")
        st.write(f"**Processed Persons:** {stats['processed_persons']}")
        
        if stats['person_names']:
            st.write("**Known Persons:**")
            for person in stats['person_names']:
                st.write(f"• {person}")
        
        st.markdown("---")
        st.subheader("💡 How to Use")
        st.write("1. Upload an image with faces")
        st.write("2. The system will detect faces")
        st.write("3. View recognition results")
        st.write("4. Check confidence scores")
    
    # Main content
    if model is None or scaler is None or cascade is None:
        st.error("❌ System not ready. Please ensure the model files exist and are properly loaded.")
        st.info("💡 Make sure you have run the training pipeline first: `python working_pipeline.py`")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔍 Face Recognition", "📊 System Overview", "⚙️ Settings"])
    
    with tab1:
        st.header("🔍 Face Recognition")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing faces to recognize"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            with st.spinner("🔍 Processing image and recognizing faces..."):
                results = recognize_faces_in_image(image, model, scaler, cascade)
            
            with col2:
                st.subheader("🎯 Recognition Results")
                
                if results:
                    st.success(f"✅ Detected {len(results)} face(s)")
                    
                    # Display results
                    for i, result in enumerate(results):
                        with st.expander(f"Face {i+1}: {result['person']} (Confidence: {result['confidence']:.3f})"):
                            col_a, col_b = st.columns([1, 1])
                            
                            with col_a:
                                # Convert BGR to RGB for display
                                face_rgb = cv2.cvtColor(result['face_crop'], cv2.COLOR_BGR2RGB)
                                st.image(face_rgb, caption=f"Detected Face {i+1}", use_column_width=True)
                            
                            with col_b:
                                st.write(f"**Person:** {result['person']}")
                                st.write(f"**Confidence:** {result['confidence']:.3f}")
                                
                                # Confidence bar
                                st.progress(result['confidence'])
                                
                                # Confidence level
                                if result['confidence'] > 0.9:
                                    st.success("High Confidence")
                                elif result['confidence'] > 0.7:
                                    st.warning("Medium Confidence")
                                else:
                                    st.error("Low Confidence")
                    
                    # Summary metrics
                    st.markdown("---")
                    st.subheader("📊 Summary")
                    
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        st.metric("Faces Detected", len(results))
                    
                    with col_metrics2:
                        avg_confidence = np.mean([r['confidence'] for r in results])
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    with col_metrics3:
                        unique_persons = len(set([r['person'] for r in results]))
                        st.metric("Unique Persons", unique_persons)
                
                else:
                    st.warning("⚠️ No faces detected in the image")
                    st.info("💡 Try uploading an image with clearer, front-facing faces")
    
    with tab2:
        st.header("📊 System Overview")
        
        # System status
        col_status1, col_status2, col_status3 = st.columns(3)
        
        with col_status1:
            st.metric("Model Status", "✅ Loaded" if stats['model_loaded'] else "❌ Missing")
        
        with col_status2:
            st.metric("Face Detector", "✅ Loaded" if cascade is not None else "❌ Missing")
        
        with col_status3:
            st.metric("Total Persons", stats['total_persons'])
        
        # Detailed information
        st.markdown("---")
        
        if stats['person_names']:
            st.subheader("👥 Known Persons")
            for person in stats['person_names']:
                person_dir = os.path.join(DATA_DIR, person)
                if os.path.exists(person_dir):
                    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    st.write(f"**{person}**: {len(images)} training images")
        
        # Model information
        st.markdown("---")
        st.subheader("🤖 Model Information")
        
        if model is not None:
            st.write(f"**Model Type:** {type(model).__name__}")
            st.write(f"**Model Parameters:** {model.get_params()}")
            
            # Check if model has been fitted
            if hasattr(model, 'classes_'):
                st.write(f"**Classes:** {list(model.classes_)}")
                st.write(f"**Number of Classes:** {len(model.classes_)}")
    
    with tab3:
        st.header("⚙️ Settings")
        
        st.subheader("🔧 Model Configuration")
        
        # Model reload button
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()
        
        st.info("💡 The model is automatically cached for better performance. Use the reload button if you've updated the model files.")
        
        # System information
        st.markdown("---")
        st.subheader("ℹ️ System Information")
        
        st.write(f"**Python Version:** {st.get_option('server.enableCORS')}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        st.write(f"**OpenCV Version:** {cv2.__version__}")
        st.write(f"**NumPy Version:** {np.__version__}")
        
        # File paths
        st.markdown("---")
        st.subheader("📁 File Paths")
        
        st.write(f"**Model Directory:** `{os.path.abspath(MODEL_DIR)}`")
        st.write(f"**Data Directory:** `{os.path.abspath(DATA_DIR)}`")
        st.write(f"**Embeddings Directory:** `{os.path.abspath(EMBEDDINGS_DIR)}`")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, OpenCV, and Scikit-learn"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
