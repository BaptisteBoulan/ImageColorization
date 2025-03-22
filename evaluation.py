import os
import shutil
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from torchvision.transforms import functional as F

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def resize_to_model_compatible_size(image, size_factor=8):
    """
    Resizes the image so its dimensions are divisible by size_factor.
    Args:
        image (PIL.Image): Input image.
        size_factor (int): Factor to which dimensions must be divisible.
    Returns:
        PIL.Image: Resized image.
    """
    width, height = image.size
    new_width = (width // size_factor) * size_factor
    new_height = (height // size_factor) * size_factor
    return image.resize((new_width, new_height), Image.BICUBIC)


def visualize_first_image(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()

    # Extract the necessary data
    fake_color = model.fake_color.detach()
    L = model.L

    # Convert the first image to RGB
    fake_img = lab_to_rgb(L[:1], fake_color[:1])[0]  # Extract the first image

    # Display the first image
    plt.figure(figsize=(5, 5))
    plt.imshow(fake_img)
    plt.axis("off")
    plt.title("Fake Colored Image")
    plt.show()

def colorize_images_in_folder(folder_path, model):
    """
    Processes a folder of black-and-white images (jpg) and saves the colorized images
    in a new folder named <original_folder_name>_colored.

    Args:
        folder_path (str): Path to the folder containing black-and-white images.
        model: A pre-trained model for colorization.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # Create the output folder
    colored_folder = folder_path.parent / f"{folder_path.name}_colored"
    colored_folder.mkdir(exist_ok=True)

    for image_file in folder_path.glob("*.jpg"):
        # Load and process the image
        image = Image.open(image_file).convert("L")  # Grayscale
        image = resize_to_model_compatible_size(image)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = np.expand_dims(image_array, axis=1)  # Add channel dimension
    
        # Convert to tensor
        tensor_images = torch.from_numpy(image_array)
        
        print(tensor_images.shape)
        
        # Prepare input data with dummy 'ab'
        data = {
            "L": tensor_images,  # Grayscale L channel
            "ab": torch.zeros((tensor_images.shape[0], 2, tensor_images.shape[2], tensor_images.shape[3]))  # Zero-filled ab
        }
    
        # Visualize the result
        visualize_first_image(model, data)
        break


    print(f"Colorized images saved to {colored_folder}")
