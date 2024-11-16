import numpy as np
from PIL import Image
import os

def normalize_image(image):
    # Convert to numpy array
    image = np.array(image, dtype=np.float32)
    # Normalize to [0, 1] range
    image = image / 255.0
    return image

def preprocess_images(folder_path, target_size=(256, 256)):
    """
    Preprocess images in a folder by resizing and normalizing.

    Resize krna zaroori hai q ki without resizing shape error aata hai jab different size ke image arrays ko same array me append krte hai.
    
    Args:
        folder_path (str): Path to the folder containing images.
        target_size (tuple): Desired image size (width, height).
    
    Returns:
        numpy.ndarray: Array of processed images.
    """
    c = 0
    processed_images = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            # Open the image and convert to RGB
            image = Image.open(img_path).convert("RGB")
            # Resize to target size
            image = image.resize(target_size, Image.BICUBIC)
            # Normalize and append to the list
            processed_images.append(np.array(image, dtype=np.float32) / 255.0)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
        print(c)
        c+=1
    return np.array(processed_images)

# Preprocess HR and LR images
hr_images = preprocess_images('../data/train/DIV2K_HR')
lr_images = preprocess_images('../data/train/DIV2K_LR')
