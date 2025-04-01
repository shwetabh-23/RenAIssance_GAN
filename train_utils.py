import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

# Loss function (Binary Cross Entropy with logits)
loss_object = nn.BCEWithLogitsLoss()

# Generator Loss Function
def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    """
    Computes the generator loss:
    - GAN loss using Binary Cross Entropy.
    - L1 loss (Mean Absolute Error) for image reconstruction.
    """

    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = torch.mean(torch.abs(target - gen_output))  # L1 loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# Discriminator Loss Function
def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Computes the discriminator loss:
    - Real loss for real images.
    - Generated loss for fake images.
    """
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# Define Optimizers
def get_optimizers(generator, discriminator, lr=2e-4, beta1=0.5):
    """
    Returns Adam optimizers for both the generator and discriminator.
    """
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    return generator_optimizer, discriminator_optimizer

def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG expects 224x224 input
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)

        # Extract features up to ReLU2_2 and ReLU3_3
        self.layer_relu2_2 = nn.Sequential(*list(vgg.children())[:7])   # Up to relu2_2
        self.layer_relu3_3 = nn.Sequential(*list(vgg.children())[:14])  # Up to relu3_3
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False  

    def forward(self, img1, img2):
        # Extract features
        features1_relu2_2 = self.layer_relu2_2(img1)
        features1_relu3_3 = self.layer_relu3_3(img1)
        features2_relu2_2 = self.layer_relu2_2(img2)
        features2_relu3_3 = self.layer_relu3_3(img2)

        # Compute perceptual loss as the sum of MSE losses from both layers
        loss = (
            torch.nn.functional.mse_loss(features1_relu2_2, features2_relu2_2) +
            torch.nn.functional.mse_loss(features1_relu3_3, features2_relu3_3)
        )
        return loss

# Function to compute perceptual loss between two images
def compute_perceptual_loss(img1, img2, device="cuda" if torch.cuda.is_available() else "cpu"):

    loss_fn = PerceptualLoss(device)
    return loss_fn(img1, img2)

class CannyEdgeLoss(nn.Module):
    def __init__(self, low_threshold=100, high_threshold=200, save_image=False, save_path="combined.png"):
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.save_image = save_image
        self.save_path = save_path  # Path to save the combined image

    def forward(self, img1, img2):
        """
        img1, img2: (B, C, H, W) images in range [0,1] or [0,255]
        """
        # Convert images to numpy arrays
        img1_np = (img1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        img2_np = (img2.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges1 = cv2.Canny(img1_gray, self.low_threshold, self.high_threshold)
        edges2 = cv2.Canny(img2_gray, self.low_threshold, self.high_threshold)

        # Convert grayscale to RGB for consistency in visualization
        edges1_rgb = cv2.cvtColor(edges1, cv2.COLOR_GRAY2RGB)
        edges2_rgb = cv2.cvtColor(edges2, cv2.COLOR_GRAY2RGB)

        # Stack images in a 2x2 grid
        top_row = np.hstack((img1_np, img2_np))  # Original images
        bottom_row = np.hstack((edges1_rgb, edges2_rgb))  # Edge-detected images
        combined_image = np.vstack((top_row, bottom_row))  # Final combined image

        # Save the combined image if enabled
        if self.save_image:
            Image.fromarray(combined_image).save(self.save_path)

        # Convert edges back to tensor
        edges1_tensor = torch.tensor(edges1, dtype=torch.float32, device=img1.device) / 255.0
        edges2_tensor = torch.tensor(edges2, dtype=torch.float32, device=img2.device) / 255.0

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(edges1_tensor, edges2_tensor)
        return loss

def compute_canny_edge_loss(batch_img1, batch_img2):
    """
    Computes the average Canny edge loss over a batch of images.
    batch_img1 and batch_img2 have shape: (batch_size, 3, 256, 256)
    """
    canny_loss_fn = CannyEdgeLoss()
    total_loss = torch.tensor(0.0, device=batch_img1.device, requires_grad=True) 
    batch_size = batch_img1.shape[0]

    for i in range(batch_size):
        img1 = batch_img1[i].unsqueeze(0)  # Add batch dimension (1, 3, 256, 256)
        img2 = batch_img2[i].unsqueeze(0)
        loss = canny_loss_fn(img1, img2)
        total_loss = total_loss + loss # Convert loss tensor to a scalar

    avg_loss = total_loss / batch_size  # Compute the average loss
    return (avg_loss)

def batch_ssim_loss(batch_img1, batch_img2):
    """
    Compute average SSIM loss for a batch of images.
    
    batch_img1, batch_img2: (B, C, H, W) PyTorch tensors in range [0,1] or [0,255].
    
    Returns:
        Average SSIM loss across the batch.
    """
    batch_size = batch_img1.shape[0]
    total_loss = 0.0

    for i in range(batch_size):
        img1_np = batch_img1[i].detach().cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        img2_np = batch_img2[i].detach().cpu().permute(1, 2, 0).numpy()

        min_dim = min(img1_np.shape[:2])  # Get smallest image dimension
        win_size = min(7, min_dim) if min_dim >= 7 else min_dim  # Ensure win_size <= min_dim

        loss = 1 - ssim(img1_np, img2_np, data_range=1.0, win_size=win_size, channel_axis=-1)
        total_loss += loss

    return total_loss / batch_size  # Average loss across batch

def compute_l1_loss(batch1, batch2):
    """
    Computes the L1 loss between two batches of images.
    
    Args:
        batch1 (torch.Tensor): Tensor of shape (batch_size, 3, 256, 256)
        batch2 (torch.Tensor): Tensor of shape (batch_size, 3, 256, 256)

    Returns:
        torch.Tensor: Scalar tensor representing the L1 loss.
    """
    assert batch1.shape == batch2.shape, "Input batches must have the same shape"
    return F.l1_loss(batch1, batch2)

if __name__ == '__main__' : 

    # img1 = torch.rand(1, 3, 256, 256).to("cuda")  # Dummy image
    # img2 = torch.rand(1, 3, 256, 256).to("cuda")  # Another dummy image
    loss_fn = CannyEdgeLoss(save_image= True, save_path= "combined.png")
    # loss = loss_fn(img1, img2)
    # print("Canny Edge Loss:", loss.item())
    
    img_path_1 = '/data/train_GAN/processed_data/images/0.png'
    img_1 = Image.open(img_path_1).convert("RGB")
    img_1 = transforms.ToTensor()(img_1).unsqueeze(0).to("cuda")  # Add batch dimension
    
    img_path_2 = '/data/train_GAN/processed_data/images/4.png'
    img_2 = Image.open(img_path_2).convert("RGB")
    img_2 = transforms.ToTensor()(img_2).unsqueeze(0).to("cuda")  # Add batch dimension
    ssim_loss_value = batch_ssim_loss(img_1, img_2)

    print(f"SSIM Loss: {ssim_loss_value}")
    # loss = loss_fn(img_1, img_2)
    # Compute perceptual loss
    # loss = compute_perceptual_loss(image_path_1, image_path_2)
    # print(f"Perceptual Loss: {loss}")