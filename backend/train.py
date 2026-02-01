import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from backend.models.vision_model import build_vision_model
from backend.models.audio_model import build_audio_model

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = (48, 48)

def train_vision_model(dataset_path):
    """
    Trains the vision model on the FRDS dataset.
    Assumes dataset_path has 'train' and 'test' subdirectories with class folders.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print("Setting up Data Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    model = build_vision_model(input_shape=IMG_SIZE + (1,), num_classes=7)
    
    callbacks = [
        ModelCheckpoint('backend/weights/vision_best.h5', monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    print("Starting Vision Model Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    print("Training Complete. Model saved.")

def train_audio_model(dataset_path):
    """
    Trains the audio model on the VRDS dataset.
    This is a skeleton function - usually involves loading all .wav files, 
    extracting features, and preparing X_train, y_train.
    """
    print("Audio training implementation requires detailed dataset structure parsing.")
    print("Please use the 'preprocessing/audio_data.py' to extract features first.")
    # Placeholder for user to implement specific loading logic based on directory structure

if __name__ == "__main__":
    # Example Usage
    # Ensure you have 'datasets/FRDS' folder created and populated
    print("1. Train Vision Model")
    print("2. Train Audio Model")
    choice = input("Select option: ")
    
    if choice == '1':
        train_vision_model('backend/datasets/FRDS')
    elif choice == '2':
        train_audio_model('backend/datasets/VRDS')
