# AI Clinic Project - Sign Numbers Recognition

A machine learning system for recognizing hand sign numbers 0-9 using computer vision and Random Forest classification, designed to support sign language communication accessibility.


## 📋 Abstract

This project develops a comprehensive system to recognize hand signs for numbers 0–9 using machine learning techniques. The system processes images through preprocessing pipelines, trains and evaluates various models, and provides a menu-driven interface for algorithm selection, model evaluation, and real-time webcam predictions. The primary goal is to support sign language recognition for improved communication accessibility.

## 🎯 Project Overview

### Key Features:
- **Real-time hand sign recognition** for digits 0-9
- **Custom data collection and preprocessing** pipeline
- **Feature extraction** using MediaPipe hand tracking
- **Machine learning model training** with Random Forest
- **Interactive webcam interface** with visual feedback
- **Comprehensive evaluation metrics** and model analysis

## 📊 Data Collection

### Collection Method:
- **Manual capture** using smartphone camera (960 x 1280 px resolution)
- **Multiple participants** to ensure variation in hand shape and size
- **Consistent hand positioning** across all captures
- **Controlled lighting conditions** to minimize variations

### Dataset Structure:
```
dataset/
├── train/          # 5 images per digit (50 total original)
├── validation/     # 2 images per digit (20 total original)
└── test/          # 1 image per digit (10 total original)
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

### Dataset Statistics:
- **Original**: 8 images per class (80 total)
- **After augmentation**: 136 images per class (1,360 total)
- **Balanced distribution** across all digits to avoid class bias

## 🔧 Data Preprocessing

### Objective:
Preprocess and augment hand gesture images for robust machine learning model training.

### Key Components:

#### 1. Hand Detection
- Uses custom `HandTrackingModule` for accurate hand detection
- Skips unreadable or blank images
- Ignores images where no hand is detected

#### 2. Data Augmentation
Transforms each original image into 17 variations using:
- **Brightness adjustment** (different alpha values)
- **Rotation** (various specified angles)
- **Flipping** (horizontal, vertical)
- **16 additional augmented versions** per original image

#### Results:
- **From**: 8 images → **To**: 136 images per class
- Clean, uniform, and enriched dataset ready for training

## 🏗️ Core Components

The system follows a **modular design pattern** with the following components:

### File Structure:
```
src/
├── main.py                    # Main application entry point
├── preprocess.py             # Data preprocessing pipeline
├── HandTrackingModule.py     # Hand detection and tracking
├── feature_extractor.py     # Feature extraction from hand landmarks
└── recognizer.py            # Number sign recognition system
```

## 🤝 Hand Tracking Module

### Technology:
- **MediaPipe**: Powerful computer vision library for real-time hand tracking
- **21 key points detection**: Identifies joints and fingertips
- **Hand skeleton visualization**: Draws connections between landmarks
- **Real-time processing**: Provides position data for feature extraction

### Features:
- Robust hand detection in various lighting conditions
- Accurate landmark identification
- Real-time performance optimization

## 🔍 Feature Extractor

### HandFeatureExtractor Class:
Extracts **84 meaningful features** from hand gestures:

#### Feature Categories:
1. **Raw Features (42)**:
   - X, Y coordinates of all 21 landmarks

2. **Engineered Features (42)**:
   - **Palm-to-fingertip distances** (5 features)
   - **Adjacent fingertip distances** (4 features)
   - **Finger joint angles** (15 angles)
   - **Additional geometric relationships** (18 features)

#### Preprocessing:
- **Position and scale invariance**: Normalizes landmarks
- **Robust feature engineering**: Creates meaningful geometric relationships
- **Dimensionality optimization**: Balances information content with processing speed

## 🎯 Number Sign Recognizer

### Core Functionality:

#### 1. Dataset Management
- **Organized dataset handling**: Supports train/validation/test splits
- **Directory processing**: Handles images from each number directory (0-9)
- **Feature extraction and storage**: Prepares data for model training

#### 2. Model Training
- **Algorithm**: Random Forest Classifier
- **Default configuration**: 200 trees, maximum depth of 20
- **Hyperparameter optimization**: GridSearchCV support for:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- **Feature importance analysis**: Identifies most relevant features
- **Model persistence**: Saves trained models for future use

#### 3. Why Random Forest?
After testing multiple algorithms, Random Forest was chosen because:
- **High accuracy** on validation data
- **Robust real-time performance** with webcam input
- **Reliable classification** in practical scenarios
- **Balance between complexity and performance**

## 📈 Evaluation Metrics

### Comprehensive Model Assessment:

#### 1. IoU (Intersection over Union)
- Calculates IoU for each class
- Formula: `TP / (TP + FP + FN)`
- Measures prediction accuracy per digit

#### 2. F1 Score
- Harmonic mean of precision and recall
- Provides balanced performance measure
- Accounts for both false positives and false negatives

#### 3. ROC Curve Analysis
- Creates ROC curves for each class
- Uses one-vs-rest approach for multi-class classification
- Evaluates classifier performance across all thresholds

#### 4. Confusion Matrix
- Visual representation of classification results
- Identifies common misclassification patterns
- Helps optimize model performance

## 🎮 Real-time Detection

### Interactive Features:
- **Live webcam feed**: Real-time hand sign recognition
- **Visual feedback system**:
  - 🟢 **Green**: High confidence predictions
  - 🟡 **Yellow**: Medium confidence predictions
  - 🔴 **Red**: Low confidence predictions
- **FPS counter**: Performance monitoring
- **Debug mode**: Detailed prediction information display

### User Interface:
```
Normal Mode: Clean interface with predictions
Debug Mode: Detailed confidence scores and feature information
```

## 🚀 Usage

### Installation:
```bash
# Install required dependencies
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install numpy
pip install matplotlib
```

### Running the Application:
```bash
# Start the main application
python main.py

# Preprocess new data
python preprocess.py

# Train new models
python recognizer.py
```

### Menu Options:
1. **Train Model**: Train new Random Forest classifier
2. **Evaluate Model**: Run comprehensive evaluation metrics
3. **Real-time Recognition**: Start webcam-based recognition
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Feature Analysis**: Analyze feature importance

## 🌟 Applications and Extensions

### Current Applications:
- **Educational tools** for sign language learning
- **Accessibility interfaces** for communication support
- **Gesture-controlled applications**
- **Human-computer interaction research**

### Future Extensions:
- **Full sign language alphabet** recognition
- **Word and phrase recognition**
- **Multi-hand gesture support**
- **Integration with smart home systems**
- **Mobile application development**

### Integration Possibilities:
The modular design enables easy integration with:
- Web applications
- Mobile apps
- IoT devices
- Educational platforms
- Accessibility software

## 📊 Performance Metrics

### Model Performance:
- **Training Accuracy**: High accuracy on augmented dataset
- **Real-time Performance**: Optimized for webcam input
- **Robustness**: Handles various lighting conditions and hand sizes
- **Speed**: Real-time processing with FPS monitoring

### System Requirements:
- **Camera**: Standard webcam or smartphone camera
- **Processing**: CPU-based inference (no GPU required)
- **Memory**: Lightweight model suitable for edge deployment

## 🔧 Technical Architecture

### Data Flow:
```
Input Image → Hand Detection → Landmark Extraction → 
Feature Engineering → Model Prediction → Confidence Assessment → 
Visual Feedback
```

### Module Dependencies:
```
main.py
├── HandTrackingModule.py (MediaPipe integration)
├── feature_extractor.py (Feature engineering)
├── recognizer.py (ML model management)
└── preprocess.py (Data pipeline)
```

## 🎯 Conclusion

This project represents not just a technical solution, but a commitment to creating technology that serves human values of connection, communication, and inclusion. By providing an accessible and robust sign language recognition system, we contribute to breaking down communication barriers and fostering inclusive digital environments.

### Impact Areas:
- **Accessibility**: Supporting deaf and hard-of-hearing communities
- **Education**: Facilitating sign language learning
- **Technology**: Advancing computer vision and ML applications
- **Inclusion**: Creating more accessible digital interfaces

---

*This system demonstrates the practical application of machine learning in solving real-world accessibility challenges while maintaining high technical standards and user-friendly interfaces.*
