from PIL import Image
import os

def generate_lr_images(hr_folder, lr_folder, scale=4):
    """
    Function to generate low-resolution images using bicubic downscaling from high-resolution images.
    
    Bicubic downscaling is a mathematical technique used in image processing to reduce the size of images while maintaining quality.
    Bicubic interpolation considers the values of 16 surrounding pixels (a 4x4 grid) to compute the new pixel values when resizing an image.
    
    hr_folder: Folder containing HR images
    lr_folder: Folder where LR images will be saved
    scale: Downscaling factor (e.g., 2, 4, 8)
    """

    # agar folder exist nhi krta to create kro
    if not os.path.exists(lr_folder): 
        os.makedirs(lr_folder)

    for img_name in os.listdir(hr_folder):
        # generate image path for each image 
        hr_img_path = os.path.join(hr_folder, img_name)

        # open HR image
        hr_img = Image.open(hr_img_path)

        # calculate new dimensions (downscale by a factor of scale)
        new_width = hr_img.width // scale
        new_height = hr_img.height // scale

        # downscale the image using bicubic interpolation
        lr_img = hr_img.resize((new_width, new_height), Image.BICUBIC)

        # save the LR image
        lr_img.save(os.path.join(lr_folder, img_name))

        print(f"{img_name}: ({new_width}, {new_height})")
    
    print(f"LR images saved to {lr_folder}")


# Set directories for HR and LR images
hr_train_folder = 'train/DIV2K_HR'  # High-resolution train images
lr_train_folder = 'train/DIV2K_LR'  # Folder for low-resolution train images
hr_val_folder = 'val/DIV2K_HR'      # High-resolution validation images
lr_val_folder = 'val/DIV2K_LR'      # Folder for low-resolution validation images

# Generate low-resolution images (scale=4)
generate_lr_images(hr_train_folder, lr_train_folder, scale=4)
generate_lr_images(hr_val_folder, lr_val_folder, scale=4)