import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
from dataset import get_random_patch
from data_creation_pipeline import filter_short_sentences, generate_sentence_permutations
from generate_historical_image import generate_text_image, add_background_texture_mod, modify_foreground_pixels
from multiprocessing import Pool, cpu_count

# Paths
csv_file_path = "/data/train_GAN/cleaned_spanish_sentences.csv"
texture_folder_path = "/data/train_GAN/Paredes - Reglas generales"
output_img_folder = "processed_data/paredes_images"
output_texture_folder = "processed_data/paredes_textures"

# Ensure directories exist
os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_texture_folder, exist_ok=True)

# Load CSV and preprocess
df = filter_short_sentences(csv_file_path)
df['permutations'] = df['Sentence'].apply(generate_sentence_permutations)
sentences = df.explode('permutations')['permutations'].tolist()
# Function to process a single image-texture pair
def process_item(idx_text):
    idx, text = idx_text  # Unpack tuple (index, text)

    # Generate and save text image
    img = generate_text_image(text)
    img_path = os.path.join(output_img_folder, f"{idx}.png")
    img.save(img_path)

    # Generate and save texture
    texture = get_random_patch(texture_folder_path, )
    texture_img = Image.fromarray(texture.astype(np.uint8))
    texture_path = os.path.join(output_texture_folder, f"{idx}.png")
    texture_img.save(texture_path)

    return idx  # Return index for progress tracking

if __name__ == "__main__":
    num_workers = max(1, cpu_count() - 2)  # Use most CPUs, leaving some free

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_item, enumerate(sentences)), total=len(sentences)))

    print("Preprocessing complete. Images and textures saved.")
    # process_item((32, "Hola mundo"))  # Test the function with a dummy input