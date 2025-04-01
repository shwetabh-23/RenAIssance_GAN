import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
from train_utils import generator_loss, discriminator_loss, get_optimizers
from model import Generator, Discriminator
from dataset import ImageFolderDataset, GANImageDataset
from torch.utils.data import random_split
from train import train_step, evaluate
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
from tqdm import tqdm
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

# Load dataset
dataset = GANImageDataset("/data/train_GAN/output_images", transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Split dataset into train (70%), val (15%), test (15%)
train_size = int(0.1 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size  # Ensure total sums up correctly

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models (assuming they are already defined)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Initialize TensorBoard writer
log_dir = "runs/real_GAN_training/"
os.makedirs(log_dir, exist_ok=True)
summary_writer = SummaryWriter(log_dir)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_gen_loss, total_disc_loss = 0.0, 0.0

    # Training
    for step, (generated_image, original_image) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
        gen_loss, disc_loss = train_step(
            original_image, generated_image, generator, discriminator, 
            generator_loss, discriminator_loss, 
            generator_optimizer, discriminator_optimizer, 
            step, summary_writer
        )
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss

    avg_gen_loss = total_gen_loss / len(train_loader)
    avg_disc_loss = total_disc_loss / len(train_loader)

    # Logging to TensorBoard
    summary_writer.add_scalar("Loss/Train_Generator", avg_gen_loss, epoch)
    summary_writer.add_scalar("Loss/Train_Discriminator", avg_disc_loss, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

    # Validation
    total_val_gen_loss, total_val_disc_loss = 0.0, 0.0
    for step, (generated_image, original_image) in enumerate(tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):

        val_gen_loss, val_disc_loss = evaluate(original_image, generated_image, generator, discriminator, generator_loss, discriminator_loss)
        total_val_gen_loss += val_gen_loss
        total_val_disc_loss += val_disc_loss
    
    avg_val_gen_loss = total_val_gen_loss / len(val_loader)
    avg_val_disc_loss = total_val_disc_loss / len(val_loader)
    
    torch.save(generator.state_dict(), f"/data/train_GAN/checkpoints/generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"/data/train_GAN/checkpoints/discriminator_epoch_{epoch+1}.pth")
    
    # Logging validation losses
    summary_writer.add_scalar("Loss/Val_Generator", avg_val_gen_loss, epoch)
    summary_writer.add_scalar("Loss/Val_Discriminator", avg_val_disc_loss, epoch)
    
    print(f"Validation - Gen Loss: {avg_val_gen_loss:.4f}, Disc Loss: {avg_val_disc_loss:.4f}")

# Close TensorBoard writer
summary_writer.close()
