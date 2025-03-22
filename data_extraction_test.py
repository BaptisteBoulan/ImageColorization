import os
import numpy as np
from skimage.io import imread
from PIL import Image
from utils import rgba_to_lab, lab_to_rgb
import cv2
from shapes import shapeX, shapeY

# Define the path to the training set
base_folder_path = 'dataset/landscape_folder'
compressed_folder_path = 'dataset/compressed_images'
bnw_folder_path = 'dataset/black_and_white_images'

# Create output directories if they don't exist
os.makedirs(compressed_folder_path, exist_ok=True)
os.makedirs(bnw_folder_path, exist_ok=True)

# Initialize an empty list to store all image file paths
all_image_files = []

# Walk through all directories and subdirectories in the base folder
for root, _, files in os.walk(base_folder_path):
    # Filter and collect .jpg files
    image_files = [os.path.join(root, f) for f in files if f.endswith('.jpg')]
    all_image_files.extend(image_files)

print(f"Found {len(all_image_files)} images.")

# Trier les fichiers si l'ordre est important (optionnel)
all_image_files.sort()

# Lire et redimensionner les images
images = []
for image_path in all_image_files:
    image = Image.open(image_path)
    image = image.resize((shapeY, shapeX))  # Redimensionner l'image à (shapeY, shapeX)
    image = np.array(image)  # Convertir l'image redimensionnée en tableau NumPy
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # Supprimer le canal alpha si présent
            image = image[:, :, :3]
        images.append((image_path, image))

# Sauvegarder les images compressées et en noir et blanc
for image_path, image in images:
    # Sauvegarder l'image compressée
    compressed_image_path = os.path.join(compressed_folder_path, os.path.basename(image_path))
    compressed_image = Image.fromarray(image)
    compressed_image.save(compressed_image_path, format='JPEG', quality=85)  # Compression qualité 85

    # Convertir l'image en noir et blanc et sauvegarder
    bnw_image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2GRAY)
    bnw_image_path = os.path.join(bnw_folder_path, os.path.basename(image_path))
    cv2.imwrite(bnw_image_path, bnw_image)

print(f"Images compressées sauvegardées dans {compressed_folder_path}")
print(f"Images noir et blanc sauvegardées dans {bnw_folder_path}")
