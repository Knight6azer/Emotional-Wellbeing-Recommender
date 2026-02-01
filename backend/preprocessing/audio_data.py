import librosa
import numpy as np

def extract_mfcc(audio_data, sample_rate, max_len=100):
    """
    Extracts MFCC features from audio data.
    
    Args:
        audio_data: The audio signal array.
        sample_rate: Sampling rate.
        max_len: Fixed length for padding/truncating time steps.
        
    Returns:
        np.array: MFCC features with shape (1, n_mfcc, max_len)
    """
    try:
        # Extract MFCCs
        # n_mfcc=40 is standard for emotion recognition
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        
        # Padding or truncation to ensure fixed input shape
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
            
        # Add batch dimension and channel dimension if needed for CNN
        # Shape: (1, 40, max_len, 1) usually or just (1, 40, max_len) for LSTM
        return np.expand_dims(mfccs, axis=0)
        
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None
