import pandas as pd
import os
import random
import tqdm
from pathlib import Path
from generate_historical_image import generate_text_image, add_background_texture_mod, modify_foreground_pixels
import cv2
import multiprocessing as mp
import numpy as np

def generate_sentence_permutations(sentence: str):
    """
    Generates 50 random permutations of a sentence if it has more than 10 words.
    Each permutation has a random length and a random subset of words.
    
    Args:
        sentence (str): The input sentence.
    
    Returns:
        List[str]: A list of up to 50 randomly shuffled variations of the sentence.
    """
    words = sentence.split()
    if len(words) <= 10:
        return [sentence]  # Return the original sentence if it has 10 or fewer words

    permutations = []
    for _ in range(50):
        num_words = random.randint(5, len(words))  # Random length (at least 5 words)
        sampled_words = random.sample(words, num_words)  # Random words from sentence
        random.shuffle(sampled_words)  # Shuffle selected words
        permutations.append(" ".join(sampled_words))

    return permutations

def filter_short_sentences(csv_path: str):
    """
    Loads a CSV file, removes rows where the 'Sentence' column has fewer than 10 words, and saves the cleaned file.

    Args:
        csv_path (str): Path to the input CSV file.
        output_csv (str): Path to save the cleaned CSV file.

    Returns:
        None
    """
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Ensure the 'Sentence' column exists
    if "Sentence" not in df.columns:
        raise ValueError("CSV file must contain a 'Sentence' column.")

    # Filter rows where the sentence has at least 10 words
    df_filtered = df[df["Sentence"].apply(lambda x: isinstance(x, str) and len(x.split()) >= 5)]

    return df_filtered

def get_random_text_variation(sentence):
    """Generate a random word, phrase, and full sentence variation from the given sentence."""
    words = sentence.split()

    if len(words) == 1:
        return [sentence, sentence, sentence]  # If only one word, use it for all variations

    # Random word
    word = random.choice(words)

    # Random phrase (2-4 words)
    phrase_length = random.randint(2, min(4, len(words)))  # Max phrase length = 4 or total words
    start_idx = random.randint(0, len(words) - phrase_length)
    phrase = " ".join(words[start_idx:start_idx + phrase_length])

    # Full sentence
    full_sentence = sentence

    return [word, phrase, full_sentence]

def process_chunk(df_chunk, output_dir, process_id):
    for idx, row in tqdm.tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Process {process_id}"):
        sentence = row["sentence_str"]
        variations = get_random_text_variation(sentence)

        base_path_orig_image = Path(output_dir) / "original_image"
        os.makedirs(base_path_orig_image, exist_ok= True)
        
        base_path_gen_image = Path(output_dir) / "generated_image"
        os.makedirs(base_path_gen_image, exist_ok= True)
        
        for i, text in enumerate(variations):
            # print(text)
            text_img = generate_text_image(text)
            text_with_texture = add_background_texture_mod(text_img, "/data/train_GAN/texture.jpg")
            modified_text = modify_foreground_pixels(text_with_texture)

            # Save image
            filename = f"{idx}_variation_{i}.jpg"
            
            save_path =  Path(base_path_orig_image) / filename
            cv2.imwrite(str(save_path), text_img)
            
            # Save image
            filename = f"{idx}_variation_{i}.jpg"
            
            save_path =  Path(base_path_gen_image) / filename
            cv2.imwrite(str(save_path), modified_text)
            
            # breakpoint()

def main():
    # Load your DataFrame
    df = pd.read_csv("/data/train_GAN/cleaned_sentences.csv")  # Update with your actual CSV file
    output_dir = "output_images"  # Update with your desired output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    num_processes = 8
    df_splits = np.array_split(df, num_processes)
    
    processes = []
    for i, df_chunk in enumerate(df_splits):
        p = mp.Process(target=process_chunk, args=(df_chunk, output_dir, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

if __name__ == '__main__' : 

    csv_file_path = '/data/train_GAN/cleaned_spanish_sentences.csv'
    df = filter_short_sentences(csv_file_path)
    df['permutations'] = df['Sentence'].apply(generate_sentence_permutations)
    breakpoint()
    text_img = generate_text_image(text)