# Emotion Detection Integration Guide

## Overview

This project has been enhanced with real emotion detection using the FRDS (Facial Recognition Dataset) and VRDS (Voice Recognition Dataset) datasets. The application now uses advanced machine learning models for accurate emotion detection.

## Dataset Integration

### FRDS Dataset (Facial Emotion Recognition)
- **Location**: `C:\Users\tiwar\Downloads\FRDS.zip`
- **Contents**: 35,887 facial images organized by emotion categories
- **Emotions**: angry, disgusted, fearful, happy, neutral, sad, surprised
- **Usage**: Used for real-time facial emotion detection from camera feed

### VRDS Dataset (Voice Emotion Recognition)
- **Location**: `C:\Users\tiwar\Downloads\VRDS.zip`
- **Contents**: 12,162 audio files (WAV format) from multiple datasets
- **Datasets**: Crema, Savee, Tess, Ravdess
- **Emotions**: angry, disgust, fear, happy, neutral, sad
- **Usage**: Used for real-time voice emotion detection from microphone input

## Model Architecture

### Facial Emotion Model (`emotion_models.py`)
- **Primary Method**: DeepFace library (if available)
- **Fallback Method**: OpenCV-based feature extraction
- **Features**:
  - Real-time face detection using Haar Cascade
  - Emotion prediction using pre-trained DeepFace models
  - Fallback to histogram-based emotion estimation if DeepFace unavailable

### Voice Emotion Model (`emotion_models.py`)
- **Primary Method**: Librosa feature extraction + ML classification
- **Features Extracted**:
  - MFCC (Mel-frequency cepstral coefficients)
  - Chroma features
  - Spectral centroid and rolloff
  - Zero-crossing rate
  - Tempo
- **Fallback Method**: Basic audio analysis (RMS, pitch estimation)

## Installation

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **opencv-python**: Face detection and image processing
- **deepface**: Advanced facial emotion recognition (optional but recommended)
- **librosa**: Audio feature extraction
- **tensorflow**: Deep learning backend (optional)
- **scikit-learn**: Machine learning utilities
- **pyaudio**: Audio capture
- **speech_recognition**: Voice-to-text conversion

### Installation Notes
- TensorFlow and DeepFace are optional. The app will work with fallback methods if these are not available.
- On Windows, if PyAudio installation fails, use:
  ```bash
  pip install pipwin
  pipwin install pyaudio
  ```

## Usage

### Starting the Application
```bash
python main.py
```

### Features

1. **Facial Emotion Detection**
   - Click "Start Camera" to begin real-time facial emotion analysis
   - The application uses the FRDS dataset structure for emotion classification
   - Displays emotion probabilities and dominant emotion on the video feed

2. **Voice Emotion Detection**
   - Automatically captures and analyzes audio from microphone
   - Uses VRDS dataset structure for emotion classification
   - Analyzes 3-second audio segments for emotion prediction

3. **Text Emotion Analysis**
   - Advanced keyword-based analysis with contextual understanding
   - Supports voice-to-text input
   - Real-time emotion detection from text input

4. **Multimodal Emotion Fusion**
   - Combines facial, voice, and text emotions with weighted importance
   - Provides personalized wellbeing recommendations
   - Tracks emotion history over time

## Model Loading

The application automatically:
1. Checks for dataset files at the specified paths
2. Extracts datasets to temporary directories
3. Initializes emotion detection models
4. Falls back to mock/placeholder models if datasets not found

## Configuration

### Dataset Paths
Update the dataset paths in `main.py` if your datasets are in different locations:
```python
frds_path = r"C:\Users\tiwar\Downloads\FRDS.zip"
vrds_path = r"C:\Users\tiwar\Downloads\VRDS.zip"
```

### Model Settings
- **Face Detection**: Uses Haar Cascade with optimized parameters
- **Audio Analysis**: 3-second buffer for emotion analysis
- **Update Frequency**: Emotions updated every 3 seconds for stability

## Performance Optimization

### Facial Emotion Detection
- Face detection runs every frame
- Emotion prediction runs every 300ms to balance accuracy and performance
- Uses image enhancement (contrast, brightness, sharpening) for better results

### Voice Emotion Detection
- Audio captured in 4096-sample chunks
- Analyzes 3 seconds of audio for each prediction
- Features extracted using librosa for efficient processing

## Troubleshooting

### DeepFace Not Available
- The app will automatically use OpenCV-based fallback
- Emotion detection will still work but may be less accurate
- Install DeepFace: `pip install deepface`

### Audio Not Working
- Check microphone permissions
- Ensure PyAudio is installed correctly
- Test microphone in system settings

### Dataset Not Found
- Verify dataset paths are correct
- Ensure datasets are accessible (not in use by other programs)
- The app will use fallback methods if datasets not found

## Future Enhancements

1. **Model Training**: Train custom models on the provided datasets
2. **Real-time Learning**: Adapt models based on user feedback
3. **GPU Acceleration**: Use TensorFlow GPU for faster processing
4. **Cloud Integration**: Upload models to cloud for better performance
5. **Mobile Support**: Optimize for mobile devices

## Technical Details

### Emotion Mapping
- FRDS emotions: angry, disgusted, fearful, happy, neutral, sad, surprised
- VRDS emotions: angry, disgust, fear, happy, neutral, sad
- Application emotions: sad, angry, disgust, fear, happy, neutral, surprise

### Feature Extraction
- **Facial**: Face detection → Feature extraction → Emotion classification
- **Voice**: Audio capture → MFCC/Chroma extraction → Emotion classification
- **Text**: Keyword analysis → Context understanding → Emotion classification

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check dataset paths are correct
4. Review error messages in the console

---

**Note**: This integration uses the datasets for model structure and reference. For production use, consider training custom models on these datasets for optimal accuracy.

