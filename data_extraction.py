import os
import numpy as np
from PIL import Image
import cv2
from shapes import shapeX, shapeY

# Define the path to the training set
compressed_folder_path = 'dataset/compressed_images'

# Initialize an empty list to store all image file paths
all_compressed_image_files = []

# Walk through all directories and subdirectories in the base folder
for root, _, files in os.walk(compressed_folder_path):
    # Filter and collect .jpg files
    image_files = [os.path.join(root, f) for f in files if f.endswith('.jpg')]
    all_compressed_image_files.extend(image_files)

print(f"Found {len(all_compressed_image_files)} images.")

all_compressed_image_files =all_compressed_image_files[:600]

# Trier les fichiers si l'ordre est important (optionnel)
all_compressed_image_files.sort()

entrees = []
sorties_a = []
sorties_b = []
for image_path in all_compressed_image_files:
    image = Image.open(image_path)
    image = image.resize((shapeY, shapeX))  # Redimensionner l'image à (shapeY, shapeX)
    image = np.array(image)  # Convertir l'image redimensionnée en tableau NumPy
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # Supprimer le canal alpha si présent
            image = image[:, :, :3]
        # Convertir en espace de couleur Lab
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        # Extraire le canal 'a'
        l_channel = lab_image[:, :, 0]
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]
        entrees.append(l_channel)
        sorties_a.append(a_channel)
        sorties_b.append(b_channel)

test_image_l = entrees.pop()
test_image_bnw = cv2.cvtColor(cv2.merge((test_image_l, np.full_like(test_image_l, 128), np.full_like(test_image_l, 128))), cv2.COLOR_Lab2RGB)
test_image_bnw_entree = test_image_bnw[:,:,0]

test_image_a = sorties_a.pop()
test_image_b = sorties_b.pop()

entrees = np.array(entrees)
sorties_a = np.array(sorties_a)
sorties_b = np.array(sorties_b)
