"""# image_loading.py

Dataset and data-loading utilities for the Carvana image masking task.

This file includes Jupyter-like cell markers ("# %%" and "# %% [markdown]") so
editors such as VS Code can execute or display the file as notebook cells.
"""

# %%
import torch 
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , random_split
from torchvision import transforms


# %%
transforms_img = transforms.Compose([
    transforms.Resize((572, 572)),        # original U-Net input size
    transforms.ToTensor(),                # converts to [0,1]
    transforms.Normalize((0.0,), (1.0,))  # keep as-is, just scale to [0,1]
])

# %%
class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir)
        
    def __len__(self):
        return len(self.img_names)  
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        # Open images as PIL Images. torchvision transforms expect PIL Images (or tensors),
        # so keep `image` as a PIL Image and let the transform handle conversion to Tensor.
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L").resize((388, 388))  # original U-Net mask size

        # Apply transforms to the image (if provided). Keep image as PIL so Resize/ToTensor work.
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback: convert to tensor if no transform provided
            image = transforms.ToTensor()(image)

        # Prepare mask as a binary tensor with a channel dimension (1, H, W).
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask
    
# %%
train, val = random_split(
    CarvanaDataset(img_dir="D:/UNET/carvana-image-masking-challenge/train", mask_dir="D:/UNET/carvana-image-masking-challenge/train_masks", transform=transforms_img),
    [0.8, 0.2]
)

train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=0)


# %%
for x in train_loader:
    imgs, masks = x
    print("imgs.shape:", imgs.shape)
    print("masks.shape:", masks.shape)
    break


# %%
