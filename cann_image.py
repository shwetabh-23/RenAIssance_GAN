import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from train_utils import load_image
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load images
image_path_1 = "/data/train_GAN/Paredes - Reglas generales/page_3.png"
image_path_2 = "/data/train_GAN/Mendo - Principe perfecto/page_4.png"

img1 = Image.open(image_path_1).convert("RGB")
img2 = Image.open(image_path_2).convert("RGB")

# Convert PIL images to NumPy arrays
img1_np = np.array(img1, dtype=np.uint8)
img2_np = np.array(img2, dtype=np.uint8)

# Convert to grayscale
img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

# Apply Canny edge detection
edges1 = cv2.Canny(img1_gray, 100, 200)
edges2 = cv2.Canny(img2_gray, 100, 200)

# Plot original images and edges
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img1_np)
axes[0, 0].set_title("Image 1")
axes[0, 0].axis("off")

axes[0, 1].imshow(img2_np)
axes[0, 1].set_title("Image 2")
axes[0, 1].axis("off")

axes[1, 0].imshow(edges1, cmap="gray")
axes[1, 0].set_title("Canny Edges 1")
axes[1, 0].axis("off")

axes[1, 1].imshow(edges2, cmap="gray")
axes[1, 1].set_title("Canny Edges 2")
axes[1, 1].axis("off")

plt.savefig("canny_edges.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.show()
