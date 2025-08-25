# Facial Recognition System

A robust and optimized facial recognition system built with Python, InsightFace, and scikit-learn.

## 🚀 Features

- **Advanced Face Detection**: Uses InsightFace for reliable face detection and alignment
- **High-Quality Embeddings**: Extracts 512-dimensional face embeddings using ArcFace
- **Multiple Classifiers**: Supports SVM, Random Forest, and K-Nearest Neighbors
- **Robust Pipeline**: Comprehensive error handling and validation
- **Progress Tracking**: Visual progress bars and detailed logging
- **Easy to Use**: Simple command-line interface for the complete pipeline

## 📋 Requirements

- Python 3.8+
- Windows/Linux/macOS
- CUDA support (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd facial_recognition_system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv face-recog-venv
   
   # On Windows:
   face-recog-venv\Scripts\activate
   
   # On Linux/macOS:
   source face-recog-venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Data Structure

Organize your images in the following structure:
```
data/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

**Important Notes:**
- Each person should have their own folder
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- Images should contain clear, front-facing faces
- Recommended: 10-50 images per person for best results

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd src
python run_pipeline.py
```

This will automatically run all steps:
1. Data preprocessing and face detection
2. Face embedding extraction
3. Model training and evaluation

### Option 2: Run Individual Steps

1. **Data Preprocessing:**
   ```bash
   cd src
   python data_preprocessing.py
   ```

2. **Extract Embeddings:**
   ```bash
   python embedding_extraction.py
   ```

3. **Train Model:**
   ```bash
   python train_classifier.py
   ```

### Option 3: Test Face Detection
```bash
cd src
python test_face_detection.py
```

## 📊 Output Files

After running the pipeline, you'll find:

- **`processed_data/`**: Cropped and aligned face images
- **`embeddings/`**: Face embeddings and labels
- **`models/`**: Trained classifier and scaler
- **`pipeline.log`**: Detailed execution log

## 🔧 Configuration

### Face Detection Parameters
- **Detection Size**: 640x640 (configurable in `data_preprocessing.py`)
- **Face Size**: Minimum 80x80 pixels
- **Output Size**: 112x112 pixels (ArcFace standard)

### Model Parameters
- **SVM**: Linear kernel with probability estimation
- **Random Forest**: 100 estimators
- **KNN**: 3 neighbors
- **Validation Split**: 80% train, 20% validation

## 🐛 Troubleshooting

### Common Issues

1. **"No face detected" errors:**
   - Ensure images contain clear, front-facing faces
   - Check image quality and lighting
   - Try increasing detection size in `initialize_face_analyzer()`

2. **CUDA/GPU errors:**
   - Install `onnxruntime-gpu` for GPU support
   - Set `ctx_id=-1` for CPU-only mode

3. **Memory issues:**
   - Reduce batch size or detection size
   - Process fewer images at once

4. **Import errors:**
   - Ensure virtual environment is activated
   - Check all dependencies are installed: `pip list`

### Performance Tips

- **GPU Acceleration**: Use CUDA for 2-5x faster processing
- **Batch Processing**: Process multiple images together
- **Image Quality**: Use high-resolution images (minimum 640x640)
- **Face Orientation**: Ensure faces are relatively front-facing

## 📈 Performance Metrics

The system provides comprehensive evaluation:
- **Accuracy**: Overall classification accuracy
- **Per-class Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Detailed error analysis
- **Cross-validation**: Robust performance estimation

## 🔮 Future Enhancements

- Real-time face recognition
- Web interface
- Database integration
- Multi-face detection
- Age/gender estimation
- Emotion recognition

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the `pipeline.log` file for detailed error messages
3. Ensure your data structure matches the requirements
4. Verify all dependencies are properly installed

---

**Happy Face Recognition! 🎭✨**
