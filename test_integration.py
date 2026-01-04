#!/usr/bin/env python
"""Test script to verify dataset integration"""

import os
import sys

def test_dataset_files():
    """Test if dataset files exist"""
    frds_path = r"C:\Users\tiwar\Downloads\FRDS.zip"
    vrds_path = r"C:\Users\tiwar\Downloads\VRDS.zip"
    
    print("Testing dataset files...")
    frds_exists = os.path.exists(frds_path)
    vrds_exists = os.path.exists(vrds_path)
    
    print(f"FRDS dataset: {'✓ Found' if frds_exists else '✗ Not found'} at {frds_path}")
    print(f"VRDS dataset: {'✓ Found' if vrds_exists else '✗ Not found'} at {vrds_path}")
    
    return frds_exists and vrds_exists

def test_emotion_models():
    """Test if emotion models can be imported"""
    print("\nTesting emotion models...")
    try:
        from emotion_models import FacialEmotionModel, VoiceEmotionModel, load_facial_model, load_voice_model
        print("✓ Emotion models imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing emotion models: {e}")
        return False

def test_main_import():
    """Test if main application can be imported"""
    print("\nTesting main application...")
    try:
        import main
        print("✓ Main application imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing main application: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\nTesting dependencies...")
    dependencies = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
    }
    
    optional_dependencies = {
        'deepface': 'deepface',
        'librosa': 'librosa',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'pyaudio': 'pyaudio',
        'speech_recognition': 'SpeechRecognition',
    }
    
    print("Required dependencies:")
    all_required = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (REQUIRED)")
            all_required = False
    
    print("\nOptional dependencies:")
    for module, package in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} (optional - app will use fallback)")
    
    return all_required

def main():
    """Run all tests"""
    print("=" * 60)
    print("Emotion Detection Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test dataset files
    results.append(("Dataset Files", test_dataset_files()))
    
    # Test emotion models
    results.append(("Emotion Models", test_emotion_models()))
    
    # Test main application
    results.append(("Main Application", test_main_import()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Application is ready to run.")
        print("Run 'python main.py' to start the application.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("The application may still work with limited functionality.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

