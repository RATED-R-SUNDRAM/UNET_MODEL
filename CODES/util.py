
import torch 
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , random_split
from torchvision import transforms

print("Import successful")

def get_dataloaders(img_dir,
                    mask_dir,
                    
                    CarvanaDataset,
                    transform,
                    batch_size: int = 64,
                    val_fraction: float = 0.2,
                    num_workers: int = 0,
                    seed: int = 42):
    """Create and return (train_loader, val_loader).

    This is the recommended way to get dataloaders from other scripts or notebooks.
    It avoids executing training code at import time and provides reproducible splits.
    """
    dataset = CarvanaDataset(img_dir=img_dir, mask_dir=mask_dir, transform=transform)
    total = len(dataset)
    n_val = int(total * val_fraction)
    n_train = total - n_val
    # use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
