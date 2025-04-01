import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Downsampling (encoder) block
def downsample(in_channels, out_channels, apply_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)

# Upsampling (decoder) block
def upsample(in_channels, out_channels, apply_dropout=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    layers.append(nn.BatchNorm2d(out_channels))
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)

# U-Net Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_channels = 3  # Explicitly setting the input channels
        output_channels = 3  # Explicitly setting the output channels
        
        self.down_stack = nn.ModuleList([
            downsample(input_channels, 64, apply_batchnorm=False),
            downsample(64, 128),
            downsample(128, 256),
            downsample(256, 512),
            downsample(512, 512),
            downsample(512, 512),
            downsample(512, 512),
            downsample(512, 512),
        ])

        self.up_stack = nn.ModuleList([
            upsample(512, 512, apply_dropout=True),
            upsample(1024, 512, apply_dropout=True),
            upsample(1024, 512, apply_dropout=True),
            upsample(1024, 512),
            upsample(1024, 256),
            upsample(512, 128),
            upsample(256, 64)
        ])

        self.last = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skips = []

        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = skips[:-1][::-1]  # Reverse skips, excluding last

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)  # Skip connection

        return torch.tanh(self.last(x))

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # First layer (No BatchNorm)
        self.down1 = downsample(input_channels * 2, 64, apply_batchnorm=False)
        
        # Second layer
        self.down2 = downsample(64, 128)
        
        # Third layer
        self.down3 = downsample(128, 256)
        
        # Fourth layer (PatchGAN core)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)  # Adjusted padding to keep size consistent
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=False)  
        # Final layer (1x1 output map, single channel)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)  # Adjusted padding

    def forward(self, input_image, target_image):
        # Concatenate input and target along channels
        x = torch.cat((input_image, target_image), dim=1)
        
        # Apply layers one by one (No Cloning Needed)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv4(x)  
        x = self.bn4(x)    
        x = self.relu4(x)
        x = self.conv5(x)  

        return x


def test_generator(dummy_input):
    print("Testing Generator...")
    model = Generator()
    model = model.to('cuda')
    dummy_input = dummy_input.to('cuda')
    with torch.no_grad():
        output = model(dummy_input)
    print("Generator Output Shape:", output.shape)

def test_discriminator(dummy_input):
    print("Testing Discriminator...")
    model = Discriminator()
    model = model.to('cuda')
    dummy_input = dummy_input.to('cuda')
    with torch.no_grad():
        output = model(dummy_input, dummy_input)  # Pix2Pix requires both input & target
    print("Discriminator Output Shape:", output.shape)
    print("Discriminator Output:", output)
    
if __name__ == "__main__":
    dummy_input = torch.randn(16, 3, 512, 512)
    test_generator(dummy_input)
    test_discriminator(dummy_input)
    print("Model test successful!")
    breakpoint()