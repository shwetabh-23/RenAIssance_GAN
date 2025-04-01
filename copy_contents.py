import os
import shutil

# Function to copy files and directories, excluding specific files/folders
def copy_with_exclusions(src_dir, dest_dir, exclude_list):
    # Check if source and destination directories exist
    if not os.path.exists(src_dir):
        print(f"Source directory '{src_dir}' does not exist.")
        return
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Traverse through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Skip directories/files listed in the exclude_list
        dirs[:] = [d for d in dirs if d not in exclude_list]  # Modify dirs in-place to exclude specified dirs
        
        # Get the relative path of the current directory (relative to the source)
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dest_dir, relative_path)

        # Make sure the destination directory exists
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Copy files, excluding those listed in exclude_list
        for file in files:
            file_path = os.path.join(root, file)
            # Exclude specific files
            if file not in exclude_list:
                shutil.copy(file_path, dest_path)

    print(f"Files copied from '{src_dir}' to '{dest_dir}' excluding specified files/folders.")

# List of folders and files to exclude
exclude = [
    "output_images", "checkpoints", "generated_images", "old_checkpoints", "processed_data", 
    "datasets", "PORCONES.228.35 – 1636", "Paredes - Reglas generales", "gan_generated_images_new", 
    "gan_generated_images", "Buendia - Instruccion (1)", "buendia_half_images", 
    "Mendo - Principe perfecto", "Ezcaray - Vozes", "cleaned_spanish_sentences.csv", 
    "cleaned_sentences.csv"
]

# Define your source and destination directories
src_directory = "/data/train_GAN"
dest_directory = "/data/RenAIssance"

# Call the function to copy the content
copy_with_exclusions(src_directory, dest_directory, exclude)
