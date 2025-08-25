# src/simple_embedding_extraction.py

import os
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROCESSED_DIR = "../processed_data"
EMBEDDINGS_DIR = "../embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(EMBEDDINGS_DIR, "labels.npy")

def initialize_face_analyzer():
    """Initialize InsightFace analyzer for embedding extraction only."""
    try:
        app = FaceAnalysis(name='buffalo_l')
        # We only need the recognition model, not detection
        app.prepare(ctx_id=0, det_size=(112, 112))
        logger.info("✅ Face analyzer initialized successfully")
        return app
    except Exception as e:
        logger.error(f"❌ Failed to initialize face analyzer: {e}")
        return None

def check_processed_data():
    """Check if processed data exists and has valid structure."""
    if not os.path.exists(PROCESSED_DIR):
        logger.error(f"❌ Processed data directory not found: {PROCESSED_DIR}")
        return False
    
    person_dirs = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    if not person_dirs:
        logger.error(f"❌ No person directories found in {PROCESSED_DIR}")
        return False
    
    total_images = 0
    for person in person_dirs:
        person_dir = os.path.join(PROCESSED_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        logger.info(f"📁 {person}: {len(images)} images")
    
    if total_images == 0:
        logger.error(f"❌ No processed images found in {PROCESSED_DIR}")
        return False
    
    logger.info(f"✅ Found {total_images} processed images across {len(person_dirs)} persons")
    return True

def extract_embeddings_simple(app):
    """Extract face embeddings from processed images without face detection."""
    embeddings = []
    labels = []
    
    person_dirs = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    
    for person in tqdm(person_dirs, desc="Processing persons"):
        person_dir = os.path.join(PROCESSED_DIR, person)
        
        # Get all image files for this person
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc=f"Processing {person}", leave=False):
            img_path = os.path.join(person_dir, img_name)
            
            try:
                # Load image using PIL for better compatibility
                img_pil = Image.open(img_path).convert('RGB')
                img = np.array(img_pil)
                
                # Ensure image is the right size
                if img.shape != (112, 112, 3):
                    logger.warning(f"⚠️ Skipping {img_name}: unexpected shape {img.shape}")
                    continue
                
                # Since these are already cropped faces, we can try to extract embeddings directly
                # We'll use the recognition model without detection
                try:
                    # Create a fake face object with the image data
                    # This is a workaround since we already have cropped faces
                    face_embedding = app.models['recognition'].get_feat(img)
                    
                    if face_embedding is not None and face_embedding.shape == (512,):
                        # Normalize embedding
                        face_embedding = face_embedding / np.linalg.norm(face_embedding)
                        
                        embeddings.append(face_embedding)
                        labels.append(person)
                        logger.info(f"✅ Extracted embedding from {img_name}")
                    else:
                        logger.warning(f"⚠️ Skipping {img_name}: invalid embedding shape")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract embedding from {img_name}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"🔥 Error processing {img_path}: {e}")
                continue
    
    return embeddings, labels

def save_embeddings(embeddings, labels):
    """Save embeddings and labels to files."""
    try:
        # Convert to numpy arrays
        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels)
        
        # Save embeddings and labels
        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(LABELS_FILE, labels)
        
        logger.info(f"✅ Embeddings saved in: {EMBEDDINGS_FILE}")
        logger.info(f"✅ Labels saved in: {LABELS_FILE}")
        logger.info(f"📊 Final statistics: {len(embeddings)} embeddings, {len(set(labels))} unique persons")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save embeddings: {e}")
        return False

def main():
    """Main function to run the simple embedding extraction pipeline."""
    logger.info("🚀 Starting simple face embedding extraction pipeline...")
    
    # Check if processed data exists
    if not check_processed_data():
        logger.error("❌ Cannot proceed without valid processed data")
        return False
    
    # Initialize face analyzer
    app = initialize_face_analyzer()
    if app is None:
        logger.error("❌ Cannot proceed without face analyzer")
        return False
    
    # Extract embeddings
    logger.info("🔍 Extracting face embeddings...")
    embeddings, labels = extract_embeddings_simple(app)
    
    if not embeddings:
        logger.error("❌ No embeddings were extracted successfully")
        return False
    
    # Save embeddings
    logger.info("💾 Saving embeddings...")
    if save_embeddings(embeddings, labels):
        logger.info("✅ Embedding extraction completed successfully!")
        return True
    else:
        logger.error("❌ Failed to save embeddings")
        return False

if __name__ == "__main__":
    main()
