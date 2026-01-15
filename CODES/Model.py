#%%
# %%
import torch 
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , random_split
from torchvision import transforms
import util
import torch.nn as nn
print("Import successful")
# %%
# quick sanity cell (index 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
#%%
transforms_img = transforms.Compose([
    transforms.Resize((128, 128)),        # original U-Net input size
    transforms.ToTensor(),                # converts to [0,1]
    transforms.Normalize((0.0,), (1.0,))  # keep as-is, just scale to [0,1]
])

#%%

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
        mask = Image.open(mask_path).convert("L").resize((128, 128))  # original U-Net mask size

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
    
#%%
# reload util and inspect
import importlib, inspect, util
importlib.reload(util)
print("module:", util.get_dataloaders.__module__)
print("signature:", inspect.signature(util.get_dataloaders))
print("source:\n", inspect.getsource(util.get_dataloaders))
    
# %%
train, val = util.get_dataloaders(
    img_dir="D:/UNET/carvana-image-masking-challenge/train",
    mask_dir="D:/UNET/carvana-image-masking-challenge/train_masks",
    CarvanaDataset=CarvanaDataset,
    transform=transforms_img
)
print("DataLoaders created successfully")
# %%
for x,y in train:
    print("x:", x.shape, "y:", y.shape)
    break



#%%

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #print("DecoderBlock in_channel:", in_channel, "out_channel:", out_channel)
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(2*out_channel, out_channel, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding="same"),
            nn.ReLU()
        )

    def forward(self, x, skip_con):
        #print("DecoderBlock forward x:", x.shape, "skip_con:", skip_con.shape)
        x1 = self.up(x)
        #print("x1:", x1.shape, "skip_con:", skip_con.shape)
        x = torch.cat([x1,skip_con], dim=1)
        #print("After concat x:", x.shape)
        x = self.conv(x)
        return x

#%%
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
   
        self.enc3 = EncoderBlock(128, 256)

        self.dec1 = DecoderBlock( 256, 128)
        self.dec2 = DecoderBlock( 128, 64)
        
        self.dec3 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1, down1 = self.enc1(x)
        #print("down1:", down1.shape, "skip1:", skip1.shape)
        skip2, down2 = self.enc2(down1)
        #print("down2:", down2.shape, "skip2:", skip2.shape)
        enc_final, bottleneck = self.enc3(down2)
        #print("enc_final:", enc_final.shape, "bottleneck:", bottleneck.shape)
        up1 = self.dec1(enc_final, skip2)
        up2 = self.dec2(up1, skip1)
        out = self.sigmoid(self.dec3(up2))
        return out

unet_model = model()
print("Model created successfully")

# %%
optim=torch.optim.Adam(unet_model.parameters(), lr=1e-4)
loss_fn=nn.BCEWithLogitsLoss()
# %%

unet_model = model().to(device)
print("Model created successfully")




# %%
optim=torch.optim.Adam(unet_model.parameters(), lr=1e-4)
loss_fn=nn.BCEWithLogitsLoss()
print("Optimizer and loss function created successfully")
#%%

for epoch in range(10):
    print("Epoch:", epoch , end=" ")
    loss_epoch=0
    for x,y in train:
        imgs, masks = x.to(device), y.to(device)
        preds = unet_model(imgs)
        loss = loss_fn(preds, masks)
        loss_epoch += loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()
    print("Loss:", loss_epoch/len(train))
 # %%
# taking a image and generating its mask and visualizing it

import matplotlib.pyplot as plt
unet_model.eval()
with torch.no_grad():
    img = Image.open(r"C:\Users\shiva\OneDrive\Documents\pic.jpg")
    plt.imshow(img.convert("L"), cmap='gray')
    plt.title("Input Image")
    transforms_img = transforms.Compose([
    transforms.Resize((128, 128)),        # original U-Net input size
    transforms.ToTensor(),                # converts to [0,1]
    transforms.Normalize((0.0,), (1.0,))  # keep as-is, just scale to [0,1]
])  
    img = transforms_img(img.convert("L"))
    print("img.shape:", img.shape)
    preds = unet_model(img.unsqueeze(0).to(device))
    preds = (preds.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    plt.figure()
    plt.imshow(preds, cmap='gray')
    plt.title("Generated Mask")
    plt.show()

# %% 

# PACKAGING MODEL FOR REUSABILITY
torch.save(unet_model, "../MODELS/unet_model.pth")
print("Model saved to unet_model.pth")

load_model = torch.load("../MODELS/unet_model.pth",weights_only=False)
print("Model loaded from unet_model.pth")
load_model.eval()
with torch.no_grad():
    img = Image.open(r"C:\Users\shiva\OneDrive\Documents\pic.jpg")
    plt.imshow(img.convert("L"), cmap='gray')
    plt.title("Input Image")
    transforms_img = transforms.Compose([
    transforms.Resize((128, 128)),        # original U-Net input size
    transforms.ToTensor(),                # converts to [0,1]
    transforms.Normalize((0.0,), (1.0,))  # keep as-is, just scale to [0,1]
])  
    img = transforms_img(img.convert("L"))
    print("img.shape:", img.shape)
    preds = load_model(img.unsqueeze(0).to(device))
    preds = (preds.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    plt.figure()
    plt.imshow(preds, cmap='gray')
    plt.title("Generated Mask from Loaded Model")
    plt.show()

# %%

print("Now using torchjit to save the model for production use.")

example_input = torch.randn(1, 1, 128, 128).to(device)

#tj_model = torch.jit.trace(unet_model.to(device), example_input)
#torch.jit.save(tj_model, "../MODELS/unet_model_torchjit.pth")
print("TorchJIT model saved to unet_model_torchjit.pth")
loaded_tj_model = torch.jit.load("../MODELS/unet_model_torchjit.pth", map_location=device)
print("TorchJIT model loaded from unet_model_torchjit.pth")
loaded_tj_model.eval()
with torch.no_grad():
    img = Image.open(r"D:\UNET\carvana-image-masking-challenge\train\0cdf5b5d0ce1_04.jpg")
    plt.imshow(img.convert("L"), cmap='gray')
    plt.title("Input Image")
    transforms_img = transforms.Compose([
    transforms.Resize((128, 128)),        # original U-Net input size
    transforms.ToTensor(),                # converts to [0,1]
    transforms.Normalize((0.0,), (1.0,))  # keep as-is, just scale to [0,1]
])  
    img = transforms_img(img.convert("L"))
    print("img.shape:", img.shape)
    preds = loaded_tj_model(img.unsqueeze(0).to(device))
    preds = (preds.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    plt.figure()
    plt.imshow(preds, cmap='gray')
    plt.title("Generated Mask from TorchJIT Loaded Model")
    plt.show()
# %%

# USING ONNX FORMAT
import onnx
onnx_model_path = "../MODELS/unet_model_onnx.onnx"
torch.onnx.export(unet_model.to(device),
                   example_input, 
                   onnx_model_path,
                     export_params=True,
                       opset_version=11, 
                       do_constant_folding=True, 
                       input_names = ['input'], 
                       output_names = ['output'])
print("ONNX model saved to unet_model_onnx.onnx")

onxx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onxx_model)
print("ONNX model loaded and checked successfully.")
import onnx
import onnxruntime as ort
ort_session = ort.InferenceSession(onnx_model_path)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
outputs = ort_session.run(None, {'input': to_numpy(example_input)})
print("ONNX model inference successful. Output shape:", outputs[0].shape)
# %%
