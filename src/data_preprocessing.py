# from normalizarion import hr_images, lr_images
import keras
from keras import layers
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import os

IMG_SIZE = 256

# Create a Sequential model for resizing and rescaling
resize_and_rescale = keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

# Define the data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1), # Equivalent to rotation_range=10 degrees
    layers.RandomTranslation(0.1, 0.1), # width_shift_range and height_shift_range
    layers.RandomFlip("horizontal")
])

# Function to load and preprocess images
def preprocess_image(image_path):
    # Load image
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_tensor = tf.convert_to_tensor(img_array)  # Convert to tensor
    
    # Apply resize and rescale
    img_tensor = resize_and_rescale(img_tensor)
    
    # Optionally apply data augmentation
    img_tensor = data_augmentation(img_tensor)
    
    return img_tensor

def load_data(image_dir):
    processed_images = []
    c=1
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        try:
            img_tensor = preprocess_image(img_path)
            processed_images.append(img_tensor)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
        print(c)
        c+=1
    return tf.stack(processed_images)

# Load HR and LR datasets
hr_images = load_data("../data/train/DIV2K_HR")
lr_images = load_data("../data/train/DIV2K_LR")

print(f"HR dataset shape: {hr_images.shape}")
print(f"LR dataset shape: {lr_images.shape}")