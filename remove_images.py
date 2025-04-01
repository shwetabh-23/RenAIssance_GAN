import os
from PIL import Image

def remove_corrupt_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify if the file is a valid image
        except Exception as e:
            print(f"Removing corrupt image: {file_path} - {e}")
            os.remove(file_path)

# Example usage
folder_path = "/data/train_GAN/processed_data/textures"
remove_corrupt_images(folder_path)