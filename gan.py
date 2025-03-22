import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from shapes import shapeX, shapeY

class Discriminator(nn.Module):
    def __init__(self, input_height, input_width):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Flatten()  # Flatten to (batch_size, flattened_size)
        )
        
        # Calculate output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            flattened_size = self.model(dummy_input).shape[1]

        print(f"Discriminator flattened size: {flattened_size}")

        # Add Linear layer with correct input size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

def gan_loss(discriminator, fake_images, real_images, adversarial_loss, optimizer_D):
    # Réel
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, 1, dtype=torch.float32)
    fake_labels = torch.zeros(batch_size, 1, dtype=torch.float32)

    # Calculer la perte pour les données réelles
    real_outputs = discriminator(real_images)  # Données réelles
    d_loss_real = adversarial_loss(real_outputs, real_labels)

    # Calculer la perte pour les données générées
    fake_outputs = discriminator(fake_images.detach())  # Stop gradient pour G
    d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

    # Mettre à jour le discriminateur
    d_loss = d_loss_real + d_loss_fake
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Calculer la perte pour le générateur
    fake_outputs = discriminator(fake_images)  # Données générées (G trompe D)
    g_loss = adversarial_loss(fake_outputs, real_labels)
    

    return d_loss.item(), g_loss

def train_discriminator_without_generator(discriminator, real_images_array, adversarial_loss, optimizer_D, input_height, input_width):
    # Batch size
    batch_size = real_images_array.shape[0]
    
    # Tensor
    real_images = torch.tensor(real_images_array, dtype=torch.float32).unsqueeze(1)
    
    # Labels
    real_labels = torch.ones(batch_size, 1, dtype=torch.float32)
    fake_labels = torch.zeros(batch_size, 1, dtype=torch.float32)

    # --- Calcul de la perte pour les données réelles ---
    real_outputs = discriminator(real_images)  # Données réelles
    d_loss_real = adversarial_loss(real_outputs, real_labels)

    # --- Création d'images fausses à partir de bruit aléatoire ---
    noise = torch.randn(batch_size, 1, input_height, input_width)  # Bruit aléatoire
    fake_images = 255 * noise  # Données générées aléatoirement (par exemple via bruit)

    fake_outputs = discriminator(fake_images.detach())  # Stoppe le gradient pour le générateur
    d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

    # Calcul de la perte totale pour le discriminateur
    d_loss = d_loss_real + d_loss_fake
    
    # Mise à jour du discriminateur
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()
    
    real_correct = (real_outputs > 0.5).float() == real_labels
    fake_correct = (fake_outputs < 0.5).float() == fake_labels
    
    accuracy_real = real_correct.sum() / batch_size
    accuracy_fake = fake_correct.sum() / batch_size
    
    total_accuracy = (accuracy_real + accuracy_fake) / 2
    print(f"Real Accuracy: {accuracy_real:.2f}, Fake Accuracy: {accuracy_fake:.2f}, Total Accuracy: {total_accuracy:.2f}")
    
    plt.hist(real_outputs.detach().numpy(), bins=20, alpha=0.5, label="Real Outputs")
    plt.hist(fake_outputs.detach().numpy(), bins=20, alpha=0.5, label="Fake Outputs")
    plt.legend()
    plt.show()

    return d_loss.item()
