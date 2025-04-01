import pandas as pd
import os
import tqdm
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data_creation_pipeline import get_random_text_variation
from generate_historical_image import generate_text_image, add_background_texture_mod, modify_foreground_pixels
import pandas as pd
import os
import tqdm
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load CSV
csv_path = "/data/train_GAN/cleaned_sentences.csv"
df = pd.read_csv(csv_path)

# Output directory
output_dir = "/data/train_GAN/generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to process a single row
def process_sentence(idx, sentence):
    variations = get_random_text_variation(sentence)
    results = []
    
    for i, text in enumerate([sentence]):  
        text_img = generate_text_image(text)
        text_with_texture = add_background_texture_mod(text_img, "/data/train_GAN/texture.jpg")
        modified_text = modify_foreground_pixels(text_with_texture)

        # Save image
        filename = f"{idx}_variation_{i}.jpg"
        save_path = Path(output_dir) / filename
        cv2.imwrite(str(save_path), modified_text)
        results.append(save_path)

    return results

if __name__ == "__main__":
    max_workers = min(8, os.cpu_count())  # Limit workers to avoid overhead

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sentence, idx, row["sentence_str"]): idx for idx, row in df.iterrows()}

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Generating images"):
            future.result()  # Ensures exceptions are caught

