import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import lab_to_rgb, imshow
import cv2
import matplotlib.pyplot as plt

from shapes import shapeX, shapeY
from data_extraction import test_image_bnw, entrees, sorties_a, sorties_b, test_image_l, test_image_a, test_image_b, test_image_bnw_entree
import torch.nn.functional as F
from gan import Discriminator, gan_loss, train_discriminator_without_generator  # Importer le Discriminateur et la perte GAN

# Define the neural network class (correcting Conv2d and padding issues)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        layer_size = 64
        kernel_shape = 4

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=1, kernel_size=kernel_shape, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Add function to save models and discriminators
def save_models(model_a, model_b, discriminator_a, discriminator_b, 
                path_model_a="model_a.pth", path_model_b="model_b.pth",
                path_discriminator_a="discriminator_a.pth", path_discriminator_b="discriminator_b.pth"):
    # Save models
    torch.save(model_a.state_dict(), path_model_a)
    torch.save(model_b.state_dict(), path_model_b)
    # Save discriminators
    torch.save(discriminator_a.state_dict(), path_discriminator_a)
    torch.save(discriminator_b.state_dict(), path_discriminator_b)
    print(f"Models and discriminators saved: {path_model_a}, {path_model_b}, {path_discriminator_a}, {path_discriminator_b}")

# Add function to load models and discriminators
def load_models(model_a, model_b, discriminator_a, discriminator_b, 
                path_model_a="model_a.pth", path_model_b="model_b.pth",
                path_discriminator_a="discriminator_a.pth", path_discriminator_b="discriminator_b.pth"):
    # Load models
    model_a.load_state_dict(torch.load(path_model_a))
    model_b.load_state_dict(torch.load(path_model_b))
    # Load discriminators
    discriminator_a.load_state_dict(torch.load(path_discriminator_a))
    discriminator_b.load_state_dict(torch.load(path_discriminator_b))
    # Set to evaluation mode
    model_a.eval()
    model_b.eval()
    discriminator_a.eval()
    discriminator_b.eval()
    print(f"Models and discriminators loaded: {path_model_a}, {path_model_b}, {path_discriminator_a}, {path_discriminator_b}")

# Initialize models
model_a = SimpleModel()
model_b = SimpleModel()

discriminator_a = Discriminator(input_height=shapeX, input_width=shapeY)
discriminator_b = Discriminator(input_height=shapeX, input_width=shapeY)

adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss

optimizer_D_a = optim.Adam(discriminator_a.parameters(), lr=0.0002)
optimizer_D_b = optim.Adam(discriminator_b.parameters(), lr=0.0002)
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)

def init() :
    # Initialize models
    model_a = SimpleModel()
    model_b = SimpleModel()
    
    discriminator_a = Discriminator(input_height=shapeX, input_width=shapeY)
    discriminator_b = Discriminator(input_height=shapeX, input_width=shapeY)
    
    adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
    
    optimizer_D_a = optim.Adam(discriminator_a.parameters(), lr=0.0002)
    optimizer_D_b = optim.Adam(discriminator_b.parameters(), lr=0.0002)
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)

def train(n=1, lr_d=0.01, lr_g=0.01):
    train_a(n, lr_d, lr_g)
    train_b(n, lr_d, lr_g)
    
def train_a_gan(n=1, lr_d_a=0.01, lr_g_a=0.01):
    optimizer_D_a = optim.Adam(discriminator_a.parameters(), lr=lr_d_a)
    optimizer_a = optim.Adam(model_a.parameters(), lr=lr_g_a)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    # Convert your data to PyTorch tensors
    entrees_tensor = torch.tensor(entrees, dtype=torch.float32).unsqueeze(1)
    sorties_a_tensor = torch.tensor(sorties_a, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for batches
    train_data_a = TensorDataset(entrees_tensor, sorties_a_tensor)
    train_loader_a = DataLoader(train_data_a, batch_size=16, shuffle=True)
    
    model_a.train()
    
    for epoch in range(n):
        for i, (inputs, targets_a) in enumerate(train_loader_a):
            # Forward pass for model A
            fake_images_a = model_a(inputs)

            # Compute GAN Loss for model A
            d_loss_a, g_loss_a = gan_loss(
                discriminator_a, fake_images_a, targets_a, adversarial_loss, optimizer_D_a
            )

            # Backward pass and optimization for model A
            optimizer_a.zero_grad()
            g_loss_a.backward()
            optimizer_a.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{n}, Batch {i}, Loss D_A: {d_loss_a:.4f}, Loss G_A: {g_loss_a:.4f}")
    save_models(model_a, model_b, discriminator_a, discriminator_b)
    test_a()
    
def train_b_gan(n=1, lr_d_b=0.01, lr_g_b=0.01):
    optimizer_D_b = optim.Adam(discriminator_a.parameters(), lr=lr_d_b)
    optimizer_b = optim.Adam(model_a.parameters(), lr=lr_g_b)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    # Convert your data to PyTorch tensors
    entrees_tensor = torch.tensor(entrees, dtype=torch.float32).unsqueeze(1)
    sorties_b_tensor = torch.tensor(sorties_b, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for batches
    train_data_b = TensorDataset(entrees_tensor, sorties_b_tensor)
    train_loader_b = DataLoader(train_data_b, batch_size=16, shuffle=True)
    
    model_b.train()
    
    for epoch in range(n):
        for i, (inputs, targets_b) in enumerate(train_loader_b):
            # Forward pass for model B
            fake_images_b = model_b(inputs)

            # Compute GAN Loss for model B
            d_loss_b, g_loss_b = gan_loss(
                discriminator_b, fake_images_b, targets_b, adversarial_loss, optimizer_D_b
            )

            # Backward pass and optimization for model A
            optimizer_b.zero_grad()
            g_loss_b.backward()
            optimizer_b.step()
            
            if i % 1 == 0:
                print(f"Epoch {epoch+1}/{n}, Batch {i}, Loss D_B: {d_loss_b:.4f}, Loss G_B: {g_loss_b:.4f}")
    save_models(model_a, model_b, discriminator_a, discriminator_b)
    

def train_a(n=1, lr_g_a=0.001):
    # Initialisation des optimizers
    optimizer_a = optim.Adam(model_a.parameters(), lr=lr_g_a)
    
    # Fonction de perte classique (L1Loss)
    criterion = nn.L1Loss()  # L1 Loss est souvent utilisée pour des tâches de transformation d'image

    # Conversion des données en tenseurs PyTorch
    entrees_tensor = torch.tensor(entrees, dtype=torch.float32).unsqueeze(1)
    sorties_a_tensor = torch.tensor(sorties_a, dtype=torch.float32).unsqueeze(1)
    
    # Création du DataLoader pour les batches
    train_data_a = TensorDataset(entrees_tensor, sorties_a_tensor)
    train_loader_a = DataLoader(train_data_a, batch_size=16, shuffle=True)
    
    # Mise en mode entraînement pour le modèle A
    model_a.train()
    
    for epoch in range(n):
        for i, (inputs, targets_a) in enumerate(train_loader_a):
            # Passage avant du modèle A
            fake_images_a = model_a(inputs)

            # Calcul de la perte (L1 Loss entre la sortie du modèle et la cible)
            loss_a = criterion(fake_images_a, targets_a)

            # Rétropropagation et optimisation pour le modèle A
            optimizer_a.zero_grad()
            loss_a.backward()
            optimizer_a.step()

            # Affichage des informations de perte pour chaque batch
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{n}, Batch {i}, Loss A: {loss_a.item():.4f}")

    # Sauvegarde des modèles après l'entraînement
    save_models(model_a, model_b, discriminator_a, discriminator_b)
    test_a()

def train_b(n=1, lr_d_a=0.01, lr_g_a=0.01):
    # Initialisation des optimizers
    optimizer_D_a = optim.Adam(discriminator_a.parameters(), lr=lr_d_a)
    optimizer_a = optim.Adam(model_a.parameters(), lr=lr_g_a)
    
    # Fonction de perte classique (L1Loss)
    criterion = nn.L1Loss()  # L1 Loss est souvent utilisée pour des tâches de transformation d'image

    # Conversion des données en tenseurs PyTorch
    entrees_tensor = torch.tensor(entrees, dtype=torch.float32).unsqueeze(1)
    sorties_a_tensor = torch.tensor(sorties_a, dtype=torch.float32).unsqueeze(1)
    
    # Création du DataLoader pour les batches
    train_data_a = TensorDataset(entrees_tensor, sorties_a_tensor)
    train_loader_a = DataLoader(train_data_a, batch_size=16, shuffle=True)
    
    # Mise en mode entraînement pour le modèle A
    model_a.train()
    
    for epoch in range(n):
        for i, (inputs, targets_a) in enumerate(train_loader_a):
            # Passage avant du modèle A
            fake_images_a = model_a(inputs)

            # Calcul de la perte (L1 Loss entre la sortie du modèle et la cible)
            loss_a = criterion(fake_images_a, targets_a)

            # Rétropropagation et optimisation pour le modèle A
            optimizer_a.zero_grad()
            loss_a.backward()
            optimizer_a.step()

            # Affichage des informations de perte pour chaque batch
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{n}, Batch {i}, Loss A: {loss_a.item():.4f}")

    # Sauvegarde des modèles après l'entraînement
    save_models(model_a, model_b, discriminator_a, discriminator_b)
     
def train_d_a(n=1, lr_d_a = 1):
    optimizer_D_a = optim.Adam(discriminator_a.parameters(), lr=lr_d_a)
    for epoch in range (n):
        loss = train_discriminator_without_generator(discriminator_a, sorties_a, adversarial_loss, optimizer_D_a, shapeY, shapeX)
        print(f"Epoch {epoch+1}/{n}, Loss : {loss:.4f}")
        
def test():
    # After training, run the models for prediction
    model_a.eval()  # Set the model to evaluation mode
    model_b.eval()  # Set the model to evaluation mode
    
    # Ensure that the input is converted into a tensor and has the correct shape
    test_image_bnw_tensor = torch.tensor(np.array([test_image_bnw]), dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    
    # Predict using the models
    new_test_image_a = model_a(test_image_bnw_tensor).detach().numpy()[0]  # Get the prediction for model A
    new_test_image_b = model_b(test_image_bnw_tensor).detach().numpy()[0]  # Get the prediction for model B

    
    # Remove the channel dimension from model outputs
    new_test_image_a = new_test_image_a.squeeze()  # Remove extra channel dimension, shape should be (32, 32)
    new_test_image_b = new_test_image_b.squeeze()  # Remove extra channel dimension, shape should be (32, 32)
    
    # Now concatenate the images along the last axis (axis=-1)
    new_test_image_lab = np.concatenate((
        test_image_l,  # Add back the channel dimension for proper concatenation
        new_test_image_a[..., np.newaxis],  # Add back the channel dimension
        new_test_image_b[..., np.newaxis],  # Add back the channel dimension
    ), axis=-1)
    
    # Correct the lab
    new_test_image_lab[:,:,0] /= 2.55
    
    # Subtract 128 from the second and third channels
    new_test_image_lab[:,:,1] *= 255
    new_test_image_lab[:,:,2] *= 0
    
    # Subtract 128 from the second and third channels
    new_test_image_lab[:,:,1] -= 128
    new_test_image_lab[:,:,2] -= 0
    
    # Clamp the values between -128 and 128
    new_test_image_lab[:,:,1] = np.clip(new_test_image_lab[:,:,1], -128, 128)
    new_test_image_lab[:,:,2] = np.clip(new_test_image_lab[:,:,2], -128, 128)

    
    print(new_test_image_lab[:,:,1])
    print(new_test_image_lab[:,:,2])
    
    # Show the result as RGB
    imshow(lab_to_rgb(new_test_image_lab), False)

    
def test_a():
    # After training, run the models for prediction
    model_a.eval()  # Set the model to evaluation mode
    # Ensure that the input is converted into a tensor and has the correct shape
    test_image_bnw_tensor = torch.tensor(np.array([test_image_bnw_entree]), dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    # Predict using the models
    new_test_image_a = model_a(test_image_bnw_tensor).detach().numpy()[0]  # Get the prediction for model A

    
    # Remove the channel dimension from model outputs
    new_test_image_a = new_test_image_a.squeeze()  # Remove extra channel dimension, shape should be (32, 32)
    
    # Now concatenate the images along the last axis (axis=-1)
    new_test_image_lab = np.concatenate((
        test_image_l[..., np.newaxis],  # Add back the channel dimension for proper concatenation
        new_test_image_a[..., np.newaxis],  # Add back the channel dimension
        test_image_b[..., np.newaxis],  # Add back the channel dimension
    ), axis=-1)
    
    
    
    # Clamp the values between -128 and 128
    new_test_image_lab[:,:,1] = np.clip(new_test_image_lab[:,:,1], 0, 255)

    
    print(new_test_image_lab[:,:,1])
    
    image_rgb = cv2.cvtColor(new_test_image_lab, cv2.COLOR_LAB2RGB)
    # Show the result as RGB
    plt.imshow(image_rgb)
    plt.show()


    
load_models(model_a, model_b, discriminator_a, discriminator_b)
