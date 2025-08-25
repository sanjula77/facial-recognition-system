# src/simple_preprocessing.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input and output paths
INPUT_DIR = "../data"
OUTPUT_DIR = "../processed_data"
IMAGE_SIZE = (112, 112)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_face_cascade():
    """Load OpenCV face detection cascade."""
    try:
        # Try to load the cascade file
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            logger.warning(f"⚠️ Cascade file not found at {cascade_path}")
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

def detect_faces_opencv(image, cascade):
    """Detect faces using OpenCV cascade classifier."""
    if cascade is None:
        return []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

def crop_and_resize_face(image, face_rect, target_size=(112, 112)):
    """Crop and resize a face region."""
    x, y, w, h = face_rect
    
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
    
    return face_resized

def preprocess_faces_simple():
    """Simple face preprocessing using OpenCV."""
    # Load face cascade
    cascade = load_face_cascade()
    if cascade is None:
        logger.error("❌ Cannot proceed without face cascade")
        return False
    
    total_imgs, saved, noface, errors = 0, 0, 0, 0
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for person in tqdm(person_dirs, desc="Processing persons"):
        src_dir = os.path.join(INPUT_DIR, person)
        dst_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get all image files for this person
        image_files = [f for f in os.listdir(src_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc=f"Processing {person}", leave=False):
            total_imgs += 1
            img_path = os.path.join(src_dir, img_name)
            
            try:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"⚠️ Skip (read failed): {img_path}")
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces using OpenCV
                faces = detect_faces_opencv(img_rgb, cascade)
                
                if len(faces) == 0:
                    logger.warning(f"❌ No face detected in {img_name}")
                    noface += 1
                    continue
                
                base = os.path.splitext(img_name)[0]
                
                for i, face_rect in enumerate(faces):
                    try:
                        # Crop and resize face
                        face_resized = crop_and_resize_face(img_rgb, face_rect, IMAGE_SIZE)
                        
                        # Check if face is valid
                        if face_resized.shape != (112, 112, 3):
                            logger.warning(f"⚠️ Skipping {img_name}, bad shape {face_resized.shape}")
                            continue
                        
                        # Save the processed face
                        save_path = os.path.join(dst_dir, f"{base}_{i}_opencv.jpg")
                        if os.path.exists(save_path):
                            continue
                        
                        # Convert back to BGR for OpenCV save
                        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved += 1
                        
                    except Exception as e:
                        logger.error(f"🔥 Error processing face {i} in {img_name}: {e}")
                        errors += 1
                        
            except Exception as e:
                errors += 1
                logger.error(f"🔥 Error processing {img_path}: {e}")
    
    logger.info(f"\n✅ Simple face preprocessing completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   Total images processed: {total_imgs}")
    logger.info(f"   Faces saved: {saved}")
    logger.info(f"   Images with no faces: {noface}")
    logger.info(f"   Errors: {errors}")
    
    return saved > 0

def main():
    """Main function."""
    logger.info("🚀 Starting simple face preprocessing...")
    
    success = preprocess_faces_simple()
    
    if success:
        logger.info(f"✅ Final processed images saved in: {OUTPUT_DIR}")
        return True
    else:
        logger.warning("⚠️ No faces were processed successfully")
        return False

if __name__ == "__main__":
    main()
