import cv2
import numpy as np

def preprocess_face(frame, target_size=(48, 48)):
    """
    Preprocesses a face image for the emotion model.
    1. Grayscale conversion
    2. Resize to target_size
    3. Normalization (0-1)
    
    Args:
        frame: The input image (BGR or Gray).
        target_size: Tuple (width, height).
        
    Returns:
        np.array: Preprocessed image with shape (1, target_size[0], target_size[1], 1)
    """
    if frame is None:
        raise ValueError("Input frame is None")
        
    # Convert to grayscale if necessary
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize
    normalized = resized.astype("float32") / 255.0
    
    # Expand dims to match model input (batch_size, height, width, channels)
    # Shape: (1, 48, 48, 1)
    processed = np.expand_dims(normalized, axis=-1)
    processed = np.expand_dims(processed, axis=0)
    
    return processed

def augment_data(image):
    """
    Applies random augmentations for training.
    """
    # Placeholder for training augmentation logic
    pass
