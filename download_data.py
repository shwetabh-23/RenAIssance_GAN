import kagglehub
import os
# Define the custom download path
custom_path = "/data/train_GAN/"
os.environ["KAGGLEHUB_CACHE"] = custom_path

# Download the dataset to the specified location
path = kagglehub.dataset_download("kouroshalizadeh/history-of-philosophy")

print("Dataset downloaded to:", path)
