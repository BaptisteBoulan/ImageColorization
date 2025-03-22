import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
Path.ls = lambda x: list(x.iterdir())

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from classes import ColorizationDataset, UnetBlock, PatchDiscriminator, AverageMeter
from main_model_class import MainModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

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
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def visualize_one_by_one(model, data, batch_number, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(len(fake_imgs)):
        print(f"Saving image n°{batch_number * 16 + i}")
        ax = plt.subplot(3, 5, 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        if save:
            # Create folder if it doesn't exist
            save_dir = "dataset/coloured"
            os.makedirs(save_dir, exist_ok=True)
            
            # Define the filename and save the image as .jpg
            save_path = os.path.join(save_dir, f"image_{batch_number * 16 + i + 1}.jpg")
            plt.imsave(save_path, fake_imgs[i])
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def train_model(model, train_dl, epochs, display_every=50):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 1:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs
    torch.save(model, "first_gan_loss_model.pt")



def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = torch.load(f, map_location=device)
        print(f"Modèle chargé avec succès depuis {file_path}")
        return model
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite lors du chargement du modèle : {e}")

if __name__ == "__main__":
    paths = glob.glob("dataset/compressed_images/*.[Jj][Pp][Gg]")[:300]
    paths += (glob.glob("dataset/suit/*.[Jj][Pp][Gg]"))
    paths += (glob.glob("dataset/philadelphia/*.[Jj][Pp][Gg]"))
    paths += (glob.glob("dataset/pinou/*.[Jj][Pp][Gg]"))
    paths += (glob.glob("dataset/mib/*.[Jj][Pp][Gg]"))
    
    # Check if paths is not empty before sampling
    if len(paths) == 0:
        raise ValueError("No image files found in the specified directory.")
    
    np.random.seed(716) # Some random seed
    paths_subset = np.random.choice(paths, 1000, replace=False)  # Choosing 1000 images randomly
    
    rand_idxs = np.random.permutation(1000)
    train_idxs = rand_idxs[:800] # choosing the first 8000 as training set
    val_idxs = rand_idxs[800:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))

    _, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")
        

    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))
    
    # Show discriminator
    PatchDiscriminator(3)
    discriminator = PatchDiscriminator(3)
    dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
    out = discriminator(dummy_input)
    out.shape
    
    model = load_model("first_gan_loss_model.pt")
    
    
    # (model, train_dl, 5)
    
    test_paths = (glob.glob("dataset/to_be_coloured/*.[Jj][Pp][Gg]"))
    
    # Déterminez la taille totale de l'ensemble de validation
    test_size = len(test_paths)
    
    # Créez le DataLoader avec un batch_size de 16
    test_dl = make_dataloaders(paths=test_paths, split='val', batch_size=16)
    
    # Parcourez tous les batches dans le DataLoader
    for i,data_batch in enumerate(test_dl):
        # Appliquez la fonction de visualisation sur chaque batch
        visualize_one_by_one(model, data_batch, i)


