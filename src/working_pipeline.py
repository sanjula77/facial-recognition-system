# src/working_pipeline.py

import os
import cv2
import numpy as np
from PIL import Image
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR = "../data"
OUTPUT_DIR = "../processed_data"
EMBEDDINGS_DIR = "../embeddings"
MODEL_DIR = "../models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_face_cascade():
    """Load OpenCV face detection cascade."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            logger.error(f"❌ Cascade file not found at {cascade_path}")
            return None
        
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.error("❌ Failed to load face cascade")
            return None
        
        logger.info("✅ Face cascade loaded successfully")
        return cascade
    except Exception as e:
        logger.error(f"❌ Error loading face cascade: {e}")
        return None

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
    
    return cropped_faces

def extract_simple_features(face_image):
    """Extract simple features from face image."""
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to a standard size for feature extraction
    gray_resized = cv2.resize(gray, (64, 64))
    
    # Extract HOG-like features (simplified)
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

def preprocess_faces():
    """Preprocess faces and extract features."""
    cascade = load_face_cascade()
    if cascade is None:
        return False
    
    features = []
    labels = []
    
    person_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for person in tqdm(person_dirs, desc="Processing persons"):
        src_dir = os.path.join(INPUT_DIR, person)
        dst_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(src_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc=f"Processing {person}", leave=False):
            img_path = os.path.join(src_dir, img_name)
            
            try:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                # Detect and crop faces
                cropped_faces = detect_and_crop_faces(img, cascade)
                
                if len(cropped_faces) == 0:
                    continue
                
                # Process each detected face
                for i, face in enumerate(cropped_faces):
                    try:
                        # Save the cropped face
                        save_path = os.path.join(dst_dir, f"{os.path.splitext(img_name)[0]}_{i}.jpg")
                        cv2.imwrite(save_path, face)
                        
                        # Extract features
                        face_features = extract_simple_features(face)
                        features.append(face_features)
                        labels.append(person)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing face {i} in {img_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"🔥 Error processing {img_path}: {e}")
                continue
    
    if not features:
        logger.error("❌ No features extracted")
        return False
    
    # Convert to numpy arrays
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)
    
    # Save features and labels
    np.save(os.path.join(EMBEDDINGS_DIR, "features.npy"), features)
    np.save(os.path.join(EMBEDDINGS_DIR, "labels.npy"), labels)
    
    logger.info(f"✅ Extracted {len(features)} features from {len(set(labels))} persons")
    return True

def train_classifier():
    """Train a simple classifier on the extracted features."""
    try:
        # Load features and labels
        features = np.load(os.path.join(EMBEDDINGS_DIR, "features.npy"))
        labels = np.load(os.path.join(EMBEDDINGS_DIR, "labels.npy"))
        
        if len(features) == 0:
            logger.error("❌ No features found")
            return False
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train SVM classifier
        svm = SVC(kernel='linear', probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        logger.info(f"✅ Training completed! Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred))
        
        # Save model and scaler
        joblib.dump(svm, os.path.join(MODEL_DIR, "face_classifier.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        
        logger.info("✅ Model and scaler saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error training classifier: {e}")
        return False

def main():
    """Main pipeline function."""
    logger.info("🚀 Starting Working Facial Recognition Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Preprocess faces and extract features
    logger.info("🔍 Step 1: Preprocessing faces and extracting features...")
    if not preprocess_faces():
        logger.error("❌ Face preprocessing failed")
        return False
    
    # Step 2: Train classifier
    logger.info("🔍 Step 2: Training classifier...")
    if not train_classifier():
        logger.error("❌ Classifier training failed")
        return False
    
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("📁 Output files:")
    logger.info(f"   - Processed images: {OUTPUT_DIR}")
    logger.info(f"   - Features: {EMBEDDINGS_DIR}")
    logger.info(f"   - Trained model: {MODEL_DIR}")
    
    return True

if __name__ == "__main__":
    main()
