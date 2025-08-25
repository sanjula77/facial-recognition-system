# src/recognize_face.py

import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = "../models"
EMBEDDINGS_DIR = "../embeddings"

def load_model():
    """Load the trained model and scaler."""
    try:
        model_path = os.path.join(MODEL_DIR, "face_classifier.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            return None, None
        
        if not os.path.exists(scaler_path):
            logger.error(f"❌ Scaler not found: {scaler_path}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info("✅ Model and scaler loaded successfully")
        return model, scaler
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None, None

def load_face_cascade():
    """Load OpenCV face detection cascade."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.error("❌ Failed to load face cascade")
            return None
        return cascade
    except Exception as e:
        logger.error(f"❌ Error loading face cascade: {e}")
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
        return []
    
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

def recognize_faces_in_image(image_path, model, scaler, cascade):
    """Recognize faces in a single image."""
    try:
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"❌ Could not read image: {image_path}")
            return None
        
        # Detect and crop faces
        cropped_faces, face_locations = detect_and_crop_faces(image, cascade)
        
        if len(cropped_faces) == 0:
            logger.info("❌ No faces detected in the image")
            return None
        
        logger.info(f"✅ Detected {len(cropped_faces)} face(s)")
        
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
                    'location': location
                })
                
                logger.info(f"Face {i+1}: {prediction} (confidence: {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"🔥 Error processing face {i}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"🔥 Error processing image: {e}")
        return None

def main():
    """Main function to demonstrate face recognition."""
    logger.info("🚀 Starting Face Recognition Demo")
    logger.info("=" * 50)
    
    # Load model and scaler
    model, scaler = load_model()
    if model is None or scaler is None:
        logger.error("❌ Cannot proceed without model")
        return
    
    # Load face cascade
    cascade = load_face_cascade()
    if cascade is None:
        logger.error("❌ Cannot proceed without face cascade")
        return
    
    # Example: recognize faces in a test image
    # You can change this path to any image you want to test
    test_image_path = "../data/pasindu/1.jpg"  # Change this to test different images
    
    if os.path.exists(test_image_path):
        logger.info(f"🔍 Testing recognition on: {test_image_path}")
        results = recognize_faces_in_image(test_image_path, model, scaler, cascade)
        
        if results:
            logger.info("\n📊 Recognition Results:")
            for result in results:
                logger.info(f"   Face {result['face_id']+1}: {result['person']} "
                          f"(confidence: {result['confidence']:.3f})")
        else:
            logger.info("❌ No faces were recognized")
    else:
        logger.warning(f"⚠️ Test image not found: {test_image_path}")
        logger.info("💡 To test recognition, place an image in the data directory and update the path above")
    
    logger.info("\n✅ Face recognition demo completed!")
    logger.info("💡 You can now use this script to recognize faces in any image!")

if __name__ == "__main__":
    main()
