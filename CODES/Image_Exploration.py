import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(f"D:/UNET/carvana-image-masking-challenge/train/0cdf5b5d0ce1_01.jpg")
mask = Image.open(f"D:/UNET/carvana-image-masking-challenge/train_masks/0cdf5b5d0ce1_01_mask.gif")

img_np = np.array(img)
mask_np = np.array(mask)
mask_bin = (mask_np > 0).astype(np.uint8)
print(mask_bin)
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Image")

plt.subplot(1, 3, 2)
plt.imshow(mask_np, cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(mask_bin, cmap='gray')
plt.title("Mask")
plt.show()

#%%

# for i in range(1,10):
#     print("i is ", i)
#     j="0" + str(i)
#     img = Image.open(f"D:/UNET/carvana-image-masking-challenge/train/0cdf5b5d0ce1_{j}.jpg")
#     mask = Image.open(f"D:/UNET/carvana-image-masking-challenge/train_masks/0cdf5b5d0ce1_{j}_mask.gif")

#     img_np = np.array(img)
#     mask_np = np.array(mask)
#     plt.subplot(1, 2, 1)
#     plt.imshow(img_np)
#     plt.title("Image")

#     plt.subplot(1, 2, 2)
#     plt.imshow(mask_np, cmap='gray')
#     plt.title("Mask")
#     plt.show()

# print("done")