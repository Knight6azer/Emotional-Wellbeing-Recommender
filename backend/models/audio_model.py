from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D

def build_audio_model(input_shape=(40, 100), num_classes=7):
    """
    Builds a CNN + BiLSTM architecture for Speech Emotion Recognition.
    Uses CNN to extract spectral features and LSTM for temporal dependencies.
    """
    model = Sequential()

    # CNN Layers for feature extraction from MFCCs
    # Input shape: (n_mfcc, time_steps) -> we might need to transpose for 1D Conv if treating time as steps
    # Usually Conv1D expects (time_steps, features). 
    # Let's assume input is transposed to (100, 40) for Conv1D, OR we use Conv2D on the spectrogram image.
    # Standard approach: Conv1D on time axis.
    
    # We will assume input is (time_steps, n_mfcc)
    
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM Layers for sequence accumulation
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    
    # Classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example input: 100 time steps, 40 MFCC features
    model = build_audio_model(input_shape=(100, 40))
    model.summary()
