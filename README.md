# RenAIssance_GAN

Renaissance Text Generator

This project aims to generate synthetic Renaissance-style printed text images using generative models (GANs) with realistic degradation effects such as ink bleed, smudging, and faded characters. The pipeline includes text preprocessing, dataset creation, model training, and inference with custom degradation simulation.

🚀 Features
	•	Generate Renaissance-style text with realistic printing artifacts
 
	•	Support for multiple degradation types
 
	•	Modular pipeline for training and inference
 
	•	Preprocessing and formatting of historical Spanish texts
 
	•	Edge and texture simulation utilities
 
	•	Support for training with L1 and custom loss functions
 

 🧾 Directory Structure
 .
├── data_creation_pipeline.py       # Master pipeline for generating data

├── create_data.py                  # Create images from Spanish text

├── clean_data.py                   # Clean and format text

├── cleaned_sentences.csv           # Processed sentences

├── generate_historical_image.py    # Image generator with style & degradation

├── training_pipeline.py            # Model training script

├── train.py                        # GAN training entry point

├── new_training_pipeline.py        # Alternative training configuration

├── gan_inference.py                # Inference script using trained GANs

├── new_gan_inference.py            # Variant of GAN inference

├── dataset.py                      # Custom dataset class for training

├── model.py                        # Model architecture

├── train_utils.py                  # Utilities for training (logging, losses)

├── text_detection.py               # Region detection in generated text images

├── cann_image.py                   # Edge detection (possibly for augmentation)

├── remove_images.py                # Clean up or manage generated images

├── download_data.py                # Download or sync required datasets

├── copy_contents.py                # Utility to duplicate dataset/text assets

├── temp.py                         # Temporary experimentation script

└── README.md

📂 Datasets

Processed historical Spanish texts and intermediate data can be found in:
	•	cleaned_sentences.csv
	•	cleaned_spanish_sentences.csv
	•	processed_data/

📊 Outputs

Generated images with degradation artifacts are saved in:
	•	gan_generated_images/
	•	gan_generated_images_new_*
	•	generated_images/

🧱 Models

Model architectures and checkpoints are stored in:
	•	model.py
	•	checkpoints/
	•	old_checkpoints/

📜 License

This project is licensed under the terms of the MIT License.
