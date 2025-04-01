import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from data_creation_pipeline import filter_short_sentences, generate_sentence_permutations
from generate_historical_image import generate_text_image, add_background_texture_mod, modify_foreground_pixels
import random
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class GANImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.generated_dir = os.path.join(root_dir, "generated_image")
        self.original_dir = os.path.join(root_dir, "original_image")
        self.transform = transform

        # Get the list of image filenames (assumes matching names in both folders)
        self.image_filenames = sorted(os.listdir(self.generated_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        gen_path = os.path.join(self.generated_dir, img_name)
        orig_path = os.path.join(self.original_dir, img_name)
        
        gen_image = Image.open(gen_path).convert("RGB")
        orig_image = Image.open(orig_path).convert("RGB")
        
        if self.transform:
            gen_image = self.transform(gen_image)
            orig_image = self.transform(orig_image)
        
        return gen_image, orig_image

def get_random_patch(folder_path, patch_size=(224, 224)):
    """
    Selects a random image from the given folder, extracts a random patch of size (64, 64),
    and returns it as a NumPy array.
    
    Args:
        folder_path (str): Path to the folder containing images.
        patch_size (tuple): Size of the patch to extract (default: (64, 64)).
    
    Returns:
        np.ndarray: Extracted image patch as a NumPy array.
    """
    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_files:
        raise ValueError("No image files found in the specified folder.")
    
    # Choose a random image
    random_image_path = os.path.join(folder_path, random.choice(image_files))
    img = Image.open(random_image_path).convert('RGB')
    img_np = np.array(img)  # Convert image to NumPy array
    # Get random top-left coordinates for the patch
    h, w, _ = img_np.shape
    patch_h, patch_w = patch_size
    
    if h < patch_h or w < patch_w:
        raise ValueError("Image size is smaller than the patch size.")
    
    top = random.randint(0, h - patch_h)
    left = random.randint(0, w - patch_w)
    
    # Extract patch
    patch = img_np[top:top + patch_h, left:left + patch_w, :]
    
    return patch

class TextImageDataset(Dataset):
    def __init__(self, img_folder, texture_folder, transform=None):
        self.img_folder = img_folder
        self.texture_folder = texture_folder
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_folder))

    def __getitem__(self, idx):
        # Load pre-saved images and textures
        img_path = os.path.join(self.img_folder, f"{idx}.png")
        texture_path = os.path.join(self.texture_folder, f"{idx}.png")

        img = Image.open(img_path).convert("RGB")
        texture = Image.open(texture_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            texture = self.transform(texture)
        else:
            transform = transforms.ToTensor()  # Default transform
            img = transform(img)
            texture = transform(texture)

        return img, texture

import matplotlib.pyplot as plt
def plot_tensor_image(image_tensor):
    # Convert from CHW to HWC and move tensor to CPU if necessary
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()

    # Normalize if values are between 0 and 1
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype("uint8")

    # Plot the image
    plt.imshow(image_np)
    plt.axis("off")  # Hide axes
    plt.savefig("resizes_image.png")  # Save the image
    # plt.show()

if __name__ == '__main__' : 
    # Usage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_folder = '/data/train_GAN/processed_data/images'
    texture_folder = '/data/train_GAN/processed_data/textures'

    dataset = TextImageDataset(img_folder=image_folder, texture_folder= texture_folder, transform=transform)
    breakpoint()    
    img, texture = dataset.__getitem__(0)
    plot_tensor_image(img)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # Example: Get a batch
    # batch = next(iter(dataloader))  # Shape: [batch_size, 3, 224, 224]

    # Load dataset
    # image_path = '/data/train_GAN/generated_images'
    # dataset = GANImageDataset(image_path, transform=transform)

    # # Split dataset into train (70%), val (15%), test (15%)
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - val_size  # Ensure total sums up correctly

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # # Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    # Create dataset and dataloader
    
    dataset = GANImageDataset("/data/train_GAN/output_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Example usage
    for gen_img, orig_img in dataloader:
        print(gen_img.shape, orig_img.shape)  # Example output: torch.Size([16, 3, 256, 256])

