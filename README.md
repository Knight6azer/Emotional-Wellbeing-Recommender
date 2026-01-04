# Emotionally Aware Wellbeing Recommender

A streamlined multimodal emotion app that combines visual, textual, and audio inputs to provide personalized wellbeing recommendations. The current build runs in safe fallback mode for stability.

## Features

- Visual emotion detection using camera input (OpenCV-based, lightweight)
- Text emotion analysis with contextual keyword parsing
- Optional audio capture and basic emotion estimation
- Optional voice-to-text input
- Personalized wellbeing recommendations
- Photo capture with emotion overlay

## Current Mode

- Real ML models are disabled for stability (`TensorFlow`, `DeepFace`, etc.).
- Fallback methods are used for all emotion analyses.
- The app works without any external datasets.

## Requirements

- Python 3.8+ on Windows
- `opencv-python`, `numpy`, `matplotlib`, `Pillow` (installed via `requirements.txt`)
- Optional: `pyaudio`, `SpeechRecognition`, `pyttsx3` for audio and voice features

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Heavy ML packages listed in `requirements.txt` are optional and not used by the current build.
- If `PyAudio` installation fails on Windows:
  ```bash
  pip install pipwin
  pipwin install pyaudio
  ```

## Usage

Start the app:

```bash
python main.py
```

Tips:

- Click "Start Camera" to begin facial detection; use "Capture" to save a photo.
- Enter text in the input area or use "[MIC] Start Voice Input" if available.
- Use "[AUDIO] Capture Voice Emotion" to estimate emotion from recent audio.

## File Structure

```
WBRM.py/
├── main.py          # Main application (unchanged logic)
├── requirements.txt # Dependencies
└── README.md        # Project overview and usage
```

The `captured_photos/` folder is created automatically when you capture photos.

## Troubleshooting

- Ensure dependencies are installed: `pip install -r requirements.txt`.
- If camera/audio/voice features are unavailable, the app degrades gracefully.
- On Windows consoles with encoding quirks, the app uses ASCII-safe messages.

## Roadmap

- Re-enable real models behind a configuration flag when stability allows.
- Replace fallback audio analysis with feature-based classification.
- Persist emotion history and add export options.

## License

Provided as-is for educational and research purposes.