# src/run_pipeline.py

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_preprocessing import main as run_preprocessing
from embedding_extraction import main as run_embedding_extraction
from train_classifier import main as run_training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'cv2', 'numpy', 'insightface', 'sklearn', 'joblib', 'tqdm', 'PIL'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required modules: {', '.join(missing_modules)}")
        logger.error("Please install missing dependencies: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are available")
    return True

def check_data_structure():
    """Check if the data directory structure is correct."""
    data_dir = Path("../data")
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Check for person subdirectories
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not person_dirs:
        logger.error(f"❌ No person directories found in {data_dir}")
        return False
    
    total_images = 0
    for person_dir in person_dirs:
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
        total_images += len(images)
        logger.info(f"📁 {person_dir.name}: {len(images)} images")
    
    if total_images == 0:
        logger.error(f"❌ No images found in data directories")
        return False
    
    logger.info(f"✅ Found {total_images} images across {len(person_dirs)} persons")
    return True

def run_pipeline():
    """Run the complete facial recognition pipeline."""
    start_time = time.time()
    
    logger.info("🚀 Starting Facial Recognition Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Check dependencies
    logger.info("🔍 Step 1: Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Step 2: Check data structure
    logger.info("🔍 Step 2: Checking data structure...")
    if not check_data_structure():
        return False
    
    # Step 3: Data preprocessing
    logger.info("🔍 Step 3: Running data preprocessing...")
    logger.info("⏳ This may take a while depending on the number of images...")
    
    if not run_preprocessing():
        logger.error("❌ Data preprocessing failed")
        return False
    
    # Step 4: Embedding extraction
    logger.info("🔍 Step 4: Extracting face embeddings...")
    if not run_embedding_extraction():
        logger.error("❌ Embedding extraction failed")
        return False
    
    # Step 5: Model training
    logger.info("🔍 Step 5: Training classification model...")
    if not run_training():
        logger.error("❌ Model training failed")
        return False
    
    # Pipeline completed
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"⏱️ Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info("📁 Output files:")
    logger.info(f"   - Processed images: ../processed_data/")
    logger.info(f"   - Face embeddings: ../embeddings/")
    logger.info(f"   - Trained model: ../models/")
    
    return True

def main():
    """Main entry point."""
    try:
        success = run_pipeline()
        if success:
            logger.info("✅ Pipeline completed successfully!")
            return 0
        else:
            logger.error("❌ Pipeline failed!")
            return 1
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
