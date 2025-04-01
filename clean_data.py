import pandas as pd
import re
import os
import pandas as pd
from docx import Document
from typing import List
# Load the CSV file

# file_path = "/data/train_GAN/cleaned_sentences.csv"  # Replace with the actual file path
# df = pd.read_csv(file_path)

# # Select only the column containing sentences (assuming it's 'sentence_str')
# df = df[['sentence_str']]

# Function to clean sentences using regex
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation and brackets
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def extract_sentences_from_docx(folder_path: str, output_csv: str):
    """
    Reads all .docx files in a folder, extracts sentences, and saves them in a CSV file.

    Args:
        folder_path (str): Path to the folder containing .docx files.
        output_csv (str): Path to save the output CSV file.

    Returns:
        None
    """
    sentences = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".docx"):
            doc_path = os.path.join(folder_path, file_name)
            doc = Document(doc_path)
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Ignore empty lines
                    sentences.append(text)

    # Convert to DataFrame and save
    df = pd.DataFrame(sentences, columns=["Sentence"])
    df.to_csv(output_csv, index=False, encoding="utf-8")

# Example usage:
# extract_sentences_from_docx("path/to/folder", "output.csv")

# # Apply the cleaning function to the sentences
# df['sentence_str'] = df['sentence_str'].apply(clean_text)

# # Display the cleaned dataframe
# print(df.head())

# # Save the cleaned data (optional)
# df.to_csv("cleaned_sentences.csv", index=False)

if __name__ == '__main__' : 

    # Example usage
    folder_path = "/data/train_GAN/transcripts"  # Replace with your folder path
    output_csv = "/data/train_GAN/cleaned_spanish_sentences.csv"  # Replace with your desired output CSV path
    extract_sentences_from_docx(folder_path, output_csv)

    # Load the CSV file
    df = pd.read_csv(output_csv)

    # Apply the cleaning function to the sentences
    df['Sentence'] = df['Sentence'].apply(clean_text)
    breakpoint()
    # Display the cleaned dataframe
    # Save the cleaned data (optional)
    df.to_csv(output_csv, index=False)