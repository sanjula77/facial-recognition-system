# src/data_preprocessing.py 

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input and output paths
INPUT_DIR = "../data"
OUTPUT_DIR = "../processed_data"
IMAGE_SIZE = (112, 112)  # Standard for ArcFace

os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_face_analyzer():
    """Initialize InsightFace analyzer with optimized parameters."""
    try:
        app = FaceAnalysis(name="buffalo_l")
        # Use larger detection size for better face detection and lower threshold
        app.prepare(ctx_id=0, det_size=(640, 640))
        # Set lower detection threshold for better face detection
        if hasattr(app, 'det_model'):
            app.det_model.threshold = 0.5  # Lower threshold for more lenient detection
        logger.info("✅ Face analyzer initialized successfully")
        return app
    except Exception as e:
        logger.error(f"❌ Failed to initialize face analyzer: {e}")
        return None

def preprocess_and_save_faces(app, expand=0.3, skip_existing=True, jpeg_quality=95,
                              allowed_ext=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """Detect, align, crop, resize, and save faces using InsightFace."""
    def clamp(v, lo, hi): 
        return max(lo, min(hi, v))

    total_imgs, saved, noface, errors = 0, 0, 0, 0
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for person in tqdm(person_dirs, desc="Processing persons"):
        src_dir = os.path.join(INPUT_DIR, person)
        dst_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)

        # Get all image files for this person
        image_files = [f for f in os.listdir(src_dir) 
                      if os.path.splitext(f)[1].lower() in allowed_ext]
        
        for img_name in tqdm(image_files, desc=f"Processing {person}", leave=False):
            total_imgs += 1
            img_path = os.path.join(src_dir, img_name)

            try:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"⚠️ Skip (read failed): {img_path}")
                    continue

                # Convert BGR to RGB for InsightFace
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces using InsightFace
                faces = app.get(img_rgb)
                
                if not faces:
                    logger.warning(f"❌ No face detected in {img_name}")
                    noface += 1
                    continue

                base = os.path.splitext(img_name)[0]
                
                for i, face in enumerate(faces):
                    try:
                        # Get aligned face (already cropped and aligned by InsightFace)
                        aligned_face = face.aligned_face
                        
                        if aligned_face is None or aligned_face.size == 0:
                            # Try to get the bounding box and create a better crop
                            logger.warning(f"⚠️ Standard alignment failed for {img_name}, trying improved crop...")
                            
                            # Get face bounding box and landmarks
                            bbox = face.bbox
                            kps = face.kps if hasattr(face, 'kps') else None
                            
                            if bbox is not None and len(bbox) >= 4:
                                x1, y1, x2, y2 = bbox
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Expand the bounding box to include more context
                                w, h = x2 - x1, y2 - y1
                                expand = 0.3
                                x1 = max(0, int(x1 - w * expand))
                                y1 = max(0, int(y1 - h * expand))
                                x2 = min(img.shape[1], int(x2 + w * expand))
                                y2 = min(img.shape[0], int(y2 + h * expand))
                                
                                if x2 > x1 and y2 > y1:
                                    # Create a square crop centered on the face
                                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                    crop_size = max(x2 - x1, y2 - y1)
                                    
                                    # Ensure crop size is reasonable
                                    crop_size = min(crop_size, min(img.shape[0], img.shape[1]))
                                    
                                    # Calculate crop boundaries
                                    crop_x1 = max(0, center_x - crop_size // 2)
                                    crop_y1 = max(0, center_y - crop_size // 2)
                                    crop_x2 = min(img.shape[1], crop_x1 + crop_size)
                                    crop_y2 = min(img.shape[0], crop_y1 + crop_size)
                                    
                                    # Adjust if we hit image boundaries
                                    if crop_x2 - crop_x1 < crop_size:
                                        if crop_x1 == 0:
                                            crop_x2 = min(img.shape[1], crop_x1 + crop_size)
                                        else:
                                            crop_x1 = max(0, crop_x2 - crop_size)
                                    
                                    if crop_y2 - crop_y1 < crop_size:
                                        if crop_y1 == 0:
                                            crop_y2 = min(img.shape[0], crop_y1 + crop_size)
                                        else:
                                            crop_y1 = max(0, crop_y2 - crop_size)
                                    
                                    # Create the crop
                                    face_crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                                    
                                    # Check if crop is valid
                                    if face_crop.size > 0 and face_crop.shape[0] >= 80 and face_crop.shape[1] >= 80:
                                        # Resize to target size
                                        face_resized = cv2.resize(face_crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                                        
                                        # Ensure RGB format
                                        if face_resized.shape == (112, 112, 3):
                                            # Save the improved crop
                                            save_path = os.path.join(dst_dir, f"{base}_{i}_improved.jpg")
                                            if skip_existing and os.path.exists(save_path):
                                                continue
                                                
                                            # Convert back to BGR for OpenCV save
                                            face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                                            cv2.imwrite(save_path, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                                            saved += 1
                                            logger.info(f"✅ Improved crop successful for {img_name}")
                                            continue
                            
                            logger.warning(f"⚠️ Skipping {img_name}, all cropping methods failed")
                            continue
                        
                        # Check if face is too small
                        if aligned_face.shape[0] < 80 or aligned_face.shape[1] < 80:
                            logger.warning(f"⚠️ Skipping {img_name}, face too small: {aligned_face.shape}")
                            continue
                        
                        # Resize to target size
                        face_resized = cv2.resize(aligned_face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                        
                        # Ensure RGB format
                        if face_resized.shape != (112, 112, 3):
                            logger.warning(f"⚠️ Skipping {img_name}, bad shape {face_resized.shape}")
                            continue
                        
                        # Save the processed face
                        save_path = os.path.join(dst_dir, f"{base}_{i}.jpg")
                        if skip_existing and os.path.exists(save_path):
                            continue
                            
                        # Convert back to BGR for OpenCV save
                        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        saved += 1
                        
                    except Exception as e:
                        logger.error(f"🔥 Error processing face {i} in {img_name}: {e}")
                        errors += 1

            except Exception as e:
                errors += 1
                logger.error(f"🔥 Error processing {img_path}: {e}")

    logger.info(f"\n✅ Face preprocessing completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   Total images processed: {total_imgs}")
    logger.info(f"   Faces saved: {saved}")
    logger.info(f"   Images with no faces: {noface}")
    logger.info(f"   Errors: {errors}")
    
    return saved > 0

def main():
    """Main function to run the preprocessing pipeline."""
    logger.info("🚀 Starting face preprocessing pipeline...")
    
    # Initialize face analyzer
    app = initialize_face_analyzer()
    if app is None:
        logger.error("❌ Cannot proceed without face analyzer")
        return False
    
    # Run preprocessing
    success = preprocess_and_save_faces(app)
    
    if success:
        logger.info(f"✅ Final processed images saved in: {OUTPUT_DIR}")
        return True
    else:
        logger.warning("⚠️ No faces were processed successfully")
        return False

if __name__ == "__main__":
    main()