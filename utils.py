import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgba_to_lab(rgba_image):
    # Ensure the image is in the correct format (uint8)
    if rgba_image.dtype != np.uint8:
        rgba_image = np.clip(rgba_image * 255, 0, 255).astype(np.uint8) if rgba_image.dtype == np.float32 else rgba_image.astype(np.uint8)
    
    # Extract the RGB channels (ignore the alpha channel)
    rgb_image = rgba_image[:, :, :3]
    
    # Convert the RGB image to LAB
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    
    return lab_image


def lab_to_rgb(lab_image):
    # Convert the LAB image back to RGB
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
    
    return rgb_image



def imshow(image, asuint8=True):
    if asuint8:
        plt.imshow(image.astype(np.uint8), interpolation='none')
    else :
        plt.imshow(image.astype(np.float32), interpolation='none')
    plt.show()