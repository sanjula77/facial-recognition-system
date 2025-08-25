# src/test_face_detection.py

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_face_detection():
    """Test face detection on a few sample images."""
    
    # Initialize face analyzer
    try:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✅ Face analyzer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize face analyzer: {e}")
        return False
    
    # Test on a few images from the data directory
    data_dir = "../data"
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Find some test images
    test_images = []
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path)[:2]:  # Test first 2 images per person
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(person_path, img_file))
                    if len(test_images) >= 6:  # Limit to 6 test images
                        break
            if len(test_images) >= 6:
                break
    
    if not test_images:
        logger.error("❌ No test images found")
        return False
    
    logger.info(f"🔍 Testing face detection on {len(test_images)} images...")
    
    success_count = 0
    for img_path in test_images:
        try:
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"⚠️ Could not read image: {img_path}")
                continue
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = app.get(img_rgb)
            
            if faces:
                logger.info(f"✅ {os.path.basename(img_path)}: {len(faces)} face(s) detected")
                success_count += 1
                
                # Test face alignment
                for i, face in enumerate(faces):
                    aligned_face = face.aligned_face
                    if aligned_face is not None and aligned_face.size > 0:
                        logger.info(f"   Face {i+1}: aligned shape {aligned_face.shape}")
                    else:
                        logger.warning(f"   Face {i+1}: alignment failed")
            else:
                logger.warning(f"❌ {os.path.basename(img_path)}: No faces detected")
                
        except Exception as e:
            logger.error(f"🔥 Error processing {img_path}: {e}")
    
    logger.info(f"\n📊 Test Results: {success_count}/{len(test_images)} images had faces detected")
    
    if success_count > 0:
        logger.info("✅ Face detection is working!")
        return True
    else:
        logger.error("❌ Face detection failed on all test images")
        return False

def test_face_embedding():
    """Test face embedding extraction."""
    try:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create a simple test image (you can replace this with a real image path)
        test_img_path = None
        for person_dir in os.listdir("../data"):
            person_path = os.path.join("../data", person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_img_path = os.path.join(person_path, img_file)
                        break
                if test_img_path:
                    break
        
        if not test_img_path:
            logger.error("❌ No test image found for embedding test")
            return False
        
        # Load and process image
        img = cv2.imread(test_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = app.get(img_rgb)
        if not faces:
            logger.error("❌ No faces found for embedding test")
            return False
        
        # Extract embedding
        face = faces[0]
        embedding = face.embedding
        
        if embedding is not None and embedding.shape == (512,):
            logger.info(f"✅ Face embedding extracted successfully: shape {embedding.shape}")
            logger.info(f"   Embedding norm: {np.linalg.norm(embedding):.6f}")
            return True
        else:
            logger.error(f"❌ Invalid embedding shape: {embedding.shape if embedding is not None else 'None'}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing face embedding: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting Face Detection Tests")
    logger.info("=" * 50)
    
    # Test 1: Face detection
    logger.info("🔍 Test 1: Face Detection")
    detection_success = test_face_detection()
    
    # Test 2: Face embedding
    logger.info("\n🔍 Test 2: Face Embedding")
    embedding_success = test_face_embedding()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Face Detection: {'✅ PASS' if detection_success else '❌ FAIL'}")
    logger.info(f"Face Embedding: {'✅ PASS' if embedding_success else '❌ FAIL'}")
    
    if detection_success and embedding_success:
        logger.info("\n🎉 All tests passed! The system is ready to use.")
        return True
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()
