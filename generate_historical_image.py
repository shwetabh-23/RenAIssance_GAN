import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

def generate_text_image(word, font_path="arial.ttf", image_size=(1200, 400)):
    """Generates a large, sharp, high-contrast text image."""
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    try:
        font_path = os.path.expanduser("~/.fonts/Lato-LightItalic.ttf")  # Load from local directory
        font = ImageFont.truetype(font_path, 50)  # Make the font bigger
    except:
        print("Font not found, using default!")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)  # System fallback

    # Draw the text in the center
    # Get text bounding box
    bbox = draw.textbbox((0, 0), word, font=font)

    # Calculate max possible x, y positions to keep text inside image
    max_x = image_size[0] - (bbox[2] - bbox[0])
    max_y = image_size[1] - (bbox[3] - bbox[1])

    # Ensure x, y are non-negative
    x = random.randint(0, max(0, max_x))
    y = random.randint(0, max(0, max_y))

    # Draw text at random position
    draw.text((x, y), word, font=font, fill="black")

    save_path = "text_image.png"

    return (img)

def add_background_texture(text_img, texture_path):
    """Adds a historical texture background while retaining colors."""
    # Load texture in COLOR mode
    historical_texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    
    if historical_texture is None:
        raise FileNotFoundError(f"Texture image not found: {texture_path}")

    # Resize texture to match text image
    historical_texture = cv2.resize(historical_texture, (text_img.shape[1], text_img.shape[0]))

    # Convert text image to OpenCV format (grayscale)
    text_gray = cv2.cvtColor(text_img, cv2.COLOR_RGB2GRAY)

    # Convert text into a mask (black text on white)
    _, mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)  # Remove "_INV"


    # Apply the mask onto the texture (only text gets colored)
    text_colored = cv2.bitwise_and(historical_texture, historical_texture, mask=mask)

    # Merge with original texture
    result = cv2.addWeighted(historical_texture, 0.6, text_colored, 1.0, 0)
    # cv2.imwrite("image_with_texture.jpg", result)

    return result

def modify_foreground_pixels(img):
    """Replace foreground pixels with vertically neighboring pixels."""
    h, w, c = img.shape
    modified_img = img.copy()
    
    # Convert to grayscale to determine dark text pixels
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for x in range(w):
        for y in range(h - 1):  # Avoid last row
            if gray[y, x] < 200:  # Check intensity in grayscale
                modified_img[y, x] = img[y + 1, x]  # Replace with lower pixel
    # breakpoint()
    # cv2.imwrite("mod_foreground_pix.jpg", modified_img)
    return modified_img

def add_noise(img):
    """Add random noise to simulate document aging."""
    noise = np.random.normal(0, 30, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return cv2.GaussianBlur(noisy_img, (3, 3), 0)

def add_background_texture_mod(text_img, texture_path):
    """Adds a randomly cropped historical texture background."""
    # Load texture in COLOR mode
    historical_texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    
    if historical_texture is None:
        raise FileNotFoundError(f"Texture image not found: {texture_path}")

    # Get dimensions
    tex_h, tex_w, _ = historical_texture.shape
    img_h, img_w, _ = text_img.shape

    # Resize texture FIRST to ensure it's at least as large as text image
    scale_factor = max(img_h / tex_h, img_w / tex_w)  # Ensure full coverage
    new_size = (int(tex_w * scale_factor), int(tex_h * scale_factor))
    historical_texture = cv2.resize(historical_texture, new_size, interpolation=cv2.INTER_LINEAR)

    # Get updated size after resizing
    tex_h, tex_w, _ = historical_texture.shape
    # Random crop if texture is still larger than text image
    if tex_h > img_h and tex_w >= img_w:
        y = np.random.randint(0, tex_h - img_h + 1)
        x = np.random.randint(0, tex_w - img_w + 1)
        historical_texture = historical_texture[y:y + img_h, x:x + img_w]
    else:
        # If somehow still smaller, just resize directly
        historical_texture = cv2.resize(historical_texture, (img_w, img_h))

    # Convert text image to grayscale
    text_gray = cv2.cvtColor(text_img, cv2.COLOR_RGB2GRAY)

    # Create mask for text (black text on white)
    _, mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY)

    # Apply mask onto the texture
    text_colored = cv2.bitwise_and(historical_texture, historical_texture, mask=mask)

    # Blend with texture
    result = cv2.addWeighted(historical_texture, 0.5, text_colored, 1.0, 0)

    # cv2.imwrite("image_with_texture.jpg", result)
    return result

if __name__ == "__main__":
    # Example usage:
    word = "bhaibhaobhai"
    text_img = generate_text_image(word)
    text_with_texture = add_background_texture_mod(text_img, '/data/train_GAN/texture.jpg')
    modified_text = modify_foreground_pixels(text_with_texture)
