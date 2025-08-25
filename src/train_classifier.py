# src/train_classifier.py

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
EMBEDDINGS_FILE = "../embeddings/embeddings.npy"
LABELS_FILE = "../embeddings/labels.npy"
MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load embeddings and labels with error checking."""
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            logger.error(f"❌ Embeddings file not found: {EMBEDDINGS_FILE}")
            return None, None
        
        if not os.path.exists(LABELS_FILE):
            logger.error(f"❌ Labels file not found: {LABELS_FILE}")
            return None, None
        
        embeddings = np.load(EMBEDDINGS_FILE)
        labels = np.load(LABELS_FILE)
        
        if len(embeddings) == 0:
            logger.error("❌ No embeddings found in file")
            return None, None
        
        if len(labels) == 0:
            logger.error("❌ No labels found in file")
            return None, None
        
        if len(embeddings) != len(labels):
            logger.error(f"❌ Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
            return None, None
        
        logger.info(f"✅ Loaded {len(embeddings)} embeddings with {len(set(labels))} unique classes")
        return embeddings, labels
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None, None

def prepare_data(embeddings, labels):
    """Prepare data for training with scaling and validation."""
    try:
        # Check data quality
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logger.error(f"❌ Need at least 2 classes for classification, got {len(unique_labels)}")
            return None, None, None, None, None
        
        # Check minimum samples per class
        min_samples = min([list(labels).count(label) for label in unique_labels])
        if min_samples < 3:
            logger.warning(f"⚠️ Some classes have very few samples (minimum: {min_samples})")
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        logger.info(f"✅ Data prepared: {len(X_train)} train, {len(X_val)} validation samples")
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
        
    except Exception as e:
        logger.error(f"❌ Error preparing data: {e}")
        return None, None, None, None, None

def train_models(X_train, y_train, X_val, y_val):
    """Train multiple models and select the best one."""
    models = {
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    logger.info("🔍 Training and evaluating models...")
    
    for name, model in tqdm(models.items(), desc="Training models"):
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'val_score': val_score
            }
            
            logger.info(f"📊 {name}: Train={train_score:.3f}, Val={val_score:.3f}")
            
            # Update best model
            if val_score > best_score:
                best_score = val_score
                best_model = model
                
        except Exception as e:
            logger.error(f"❌ Error training {name}: {e}")
            continue
    
    if best_model is None:
        logger.error("❌ No models were trained successfully")
        return None, results
    
    logger.info(f"🏆 Best model: {best_score:.3f} validation score")
    return best_model, results

def evaluate_model(model, X_val, y_val, scaler):
    """Comprehensive model evaluation."""
    try:
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        logger.info("\n" + "="*50)
        logger.info("📊 MODEL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Validation samples: {len(y_val)}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        return None

def save_model(model, scaler, accuracy):
    """Save the trained model and scaler."""
    try:
        # Save model
        joblib.dump(model, MODEL_FILE)
        logger.info(f"✅ Model saved to: {MODEL_FILE}")
        
        # Save scaler
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"✅ Scaler saved to: {SCALER_FILE}")
        
        # Save model info
        model_info = {
            'model_type': type(model).__name__,
            'accuracy': accuracy,
            'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown',
            'classes': list(model.classes_) if hasattr(model, 'classes_') else 'Unknown'
        }
        
        info_file = os.path.join(MODEL_DIR, "model_info.txt")
        with open(info_file, 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"✅ Model info saved to: {info_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        return False

def main():
    """Main training pipeline."""
    logger.info("🚀 Starting facial recognition model training...")
    
    # Load data
    embeddings, labels = load_data()
    if embeddings is None:
        logger.error("❌ Cannot proceed without valid data")
        return False
    
    # Prepare data
    X_train, X_val, y_train, y_val, scaler = prepare_data(embeddings, labels)
    if X_train is None:
        logger.error("❌ Cannot proceed without prepared data")
        return False
    
    # Train models
    best_model, all_results = train_models(X_train, y_train, X_val, y_val)
    if best_model is None:
        logger.error("❌ No models were trained successfully")
        return False
    
    # Evaluate best model
    accuracy = evaluate_model(best_model, X_val, y_val, scaler)
    if accuracy is None:
        logger.error("❌ Model evaluation failed")
        return False
    
    # Save model
    if save_model(best_model, scaler, accuracy):
        logger.info("✅ Training pipeline completed successfully!")
        return True
    else:
        logger.error("❌ Failed to save model")
        return False

if __name__ == "__main__":
    main()
