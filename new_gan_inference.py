import torch
import os
from torchvision.utils import save_image
from model import Generator
from torchvision import transforms
from dataset import ImageFolderDataset, TextImageDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import random
import numpy as np

# Set random seeds for reproducibility
seed = 42  # Choose any fixed seed value
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
np.random.seed(seed)  
random.seed(seed)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False 


def load_generator(checkpoint_path, device):
    """Load the trained generator model."""
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    return generator

def generate_images(generator, dataloader, output_dir, device):
    """Generate and save images using the trained generator with input conditions."""
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (generated_image, original_image) in enumerate(dataloader):
            generated_images = generator(original_image)  # Assuming generator takes an image as input
            
            for j, image in enumerate(generated_images):
                save_image(image, os.path.join(output_dir, f"generated_{i * len(input_image) + j + 1}.png"), normalize=True)

    print(f"Generated images and saved to {output_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/data/train_GAN/checkpoints/only_l1_and_perceptual_loss_patches_porcones_with_l1complete/generator_epoch_50.pth"  # Change if needed
    output_dir = "gan_generated_images_new_porcones_with_l1_epoch_50"
    os.makedirs(output_dir, exist_ok= True)
    batch_size = 32  # Number of images per batch
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    image_folder = '/data/train_GAN/processed_data/images'
    texture_folder = '/data/train_GAN/processed_data/textures'

    dataset = TextImageDataset(img_folder=image_folder, texture_folder= texture_folder, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Split dataset into train (70%), val (15%), test (15%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure total sums up correctly

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    generator = load_generator(checkpoint_path, device)

    for i in range(100) : 
        input_image, texture = dataset.__getitem__(i)
        input_image = input_image.to(device)
        generated_image = generator(input_image.unsqueeze(dim=0))
        save_image(generated_image.squeeze(dim=0), os.path.join(output_dir, f"generated_image{i}.jpg"), normalize=True)
