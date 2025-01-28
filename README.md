# Card Scanner Model

This repository contains a comprehensive project for classifying and comparing Pokémon cards using **MobileNetV3** models. The project includes:
- **Data Augmentation**: Enhance training datasets with diverse augmentations.
- **Feature Extraction**: Precompute features for similarity-based image retrieval.
- **Model Training**: Train a custom classification model using MobileNetV3.
- **FastAPI Server**: Serve the models for real-time classification and comparison.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
   1. [Data Augmentation](#data-augmentation)
   2. [Feature Extraction](#feature-extraction)
   3. [Model Training](#model-training)
   4. [FastAPI Server](#fastapi-server)
4. [Installation](#installation)
   1. [Clone the Repository](#clone-the-repository)
   2. [Install Dependencies](#install-dependencies)
5. [Usage](#usage)
   1. [Data Augmentation](#data-augmentation-1)
   2. [Feature Extraction](#feature-extraction-1)
   3. [Train the Model](#train-the-model)
   4. [Start the Server](#start-the-server)
6. [API Endpoints](#api-endpoints)
   1. [Root Endpoint](#root-endpoint)
   2. [Image Upload](#image-upload)
7. [Example Outputs](#example-outputs)
   1. [Normal Model Results](#normal-model-results)
   2. [Trained Model Results](#trained-model-results)
   3. [Combined Results](#combined-results)

---

## Project Overview

The **Card Scanner Model** is a deep learning application focused on classifying and comparing Pokémon card images using **MobileNetV3** models. The project leverages modern techniques in computer vision, including data augmentation, feature extraction, and real-time similarity-based search.

### Key Features:
- **Data Augmentation**: The project enhances the training dataset by applying random transformations like resizing, rotation, flipping, and adding background images to generate a diverse set of training samples.
- **Feature Extraction**: It uses both a pretrained MobileNetV3 model and a custom-trained MobileNetV3 model to extract feature embeddings from Pokémon card images. These embeddings are used for efficient similarity search and retrieval.
- **FastAPI Server**: A FastAPI-powered backend serves the models and allows real-time querying of the Pokémon cards. The server supports image upload, feature extraction, and similarity search across two models (pretrained and custom-trained).

The goal is to allow users to upload a Pokémon card image and receive a ranked list of the most similar cards based on features extracted from both a pretrained MobileNetV3 model and a custom-trained one. The results are returned in a way that combines the strengths of both models for more accurate retrieval.

This repository is organized into different sections:
- **Data preparation**: Collection and augmentation of Pokémon card images.
- **Model training**: Training a custom MobileNetV3 model on the augmented dataset.
- **Feature extraction**: Extracting meaningful features from the cards to be used for similarity-based searches.
- **Server**: FastAPI server for uploading images and retrieving similar cards using Faiss.

This project is designed to be scalable and efficient, leveraging powerful tools like **PyTorch** for deep learning, **Faiss** for fast similarity searches, and **FastAPI** for building the backend server.

## Project Structure

The project is organized into the following directory structure:

pokemon-card-project/  
├── data/  
│   ├── input/                        # Raw Pokémon card images  
│   ├── augmented_dataset/            # Augmented images with class-wise directories  
│   ├── backgrounds/                  # Background images for augmentation  
├── models/  
│   ├── mobilenet_v3.pth              # Pretrained MobileNetV3 weights  
│   ├── checkpoints/                  # Checkpoints of trained models  
├── results/  
│   ├── features_mobilenetv3.npy      # Precomputed features for MobileNetV3  
│   ├── features_trained_model.npy    # Precomputed features for the trained model  
│   ├── image_paths_mobilenetv3.npy   # Image paths for the normal model features  
│   ├── image_paths_trained_model.npy # Image paths for the trained model features  
├── scripts/  
│   ├── data_augmentation.py          # Script for dataset augmentation  
│   ├── feature_extraction.py         # Script for feature extraction  
│   ├── train_classifier.py           # Script for training the classification model  
├── server/  
│   ├── main.py                       # Main server script  
│   ├── requirements.txt              # Dependencies for the server  
├── README.md                         # Main project documentation

## Features

### Data Augmentation

Data augmentation ensures diverse training examples by:
- Adding random backgrounds.
- Resizing, rotating, and flipping images.
- This helps improve the model's generalization by providing various transformations of the same image.

To augment the dataset, run:

```bash
python scripts/data_augmentation.py --input-dir data/input --output-dir data/augmented_dataset --background-dir data/backgrounds
```
### Feature Extraction

Feature extraction is a crucial step in creating embeddings for similarity-based retrieval. The following steps are involved:
- Extracts embeddings using a pretrained **MobileNetV3** model.
- Extracts embeddings using a custom-trained **MobileNetV3** model.
- Saves embeddings as `.npy` files, which are used for fast similarity search with **Faiss**.

To extract features, run:

```bash
python scripts/feature_extraction.py --output-dic results/
```
### Model Training

Model training involves training a **MobileNetV3** classifier on the augmented dataset. The training process consists of the following:

- Loading the augmented dataset and preprocessing the images.
- Training the classifier with a custom model (MobileNetV3) fine-tuned on the Pokémon card data.
- Saving the trained model checkpoints for future use.

To train the model, run:

```bash
python scripts/train_classifier.py --data-dir data/augmented_dataset --output-dir models/checkpoints
```

### FastAPI Server

The FastAPI server serves both the pretrained and custom-trained MobileNetV3 models for real-time inference. It uses **Faiss** for similarity-based image retrieval. The server exposes API endpoints for uploading Pokémon card images and retrieving similar images based on the model’s features.

- **FastAPI** is used to create the server.
- **Faiss** is used for efficient similarity search across image features.

To start the FastAPI server, navigate to the `server` directory and run:

```bash
uvicorn main:app --reload
```

## Installation

### Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/pokemon-card-project.git
cd pokemon-card-project
```

### Install Dependencies

For Python dependencies, navigate to the `server` folder and install the required packages:

```bash
pip install -r server/requirements.txt
```

### Server Requirements

The server requires the following dependencies:

- **FastAPI**: Web framework for the server.
- **Faiss**: Library for efficient similarity search.
- **PyTorch**: Deep learning framework for model inference.

## Usage
### Data Augmentation

To augment the dataset, run the following command:

```bash
python scripts/data_augmentation.py --input-dir data/input --output-dir data/augmented_dataset --background-dir data/backgrounds
```

### Feature Extraction

To extract features using the models, run the following command:

```bash
python scripts/feature_extraction.py --output-dic results/
```

### Train the Model

To train the custom classifier using the augmented dataset, run the following command:

```bash
python scripts/train_classifier.py --data-dir data/augmented_dataset --output-dir models/checkpoints
```

### Start the Server

To start the FastAPI server, navigate to the `server` folder and run:

```bash
uvicorn main:app --reload
```

## API Endpoints
### Root Endpoint
1. **Root Endpoint**  
   **URL:** `GET /`  
   **Description:** Returns a message confirming the server is running.  
   **Response:**
   ```json
   {
       "message": "Comparison between Normal and Trained MobileNetV3 is Running!"
   }
### Image Upload
2. **Image Upload**  
   **URL:** `POST /upload/`  
   **Description:** Accepts an image file, performs similarity search, and returns the results.  
   **Request:**  
   - Form field: `file` (the image file to upload).

   **Response:**
   ```json
   {
       "status": "success",
       "results": {
           "normal_model": [...],    # Top 5 results from the pretrained MobileNetV3
           "trained_model": [...],   # Top 5 results from the trained model
           "combined_model": [...]   # Top 5 combined results with weighted scores
       }
   }

## Example Outputs
### Normal Model Results
  1. **Normal Model Results**
     ```json
     [
         {"similarity": 0.95, "image_path": "path/to/image1.jpg"},
         {"similarity": 0.92, "image_path": "path/to/image2.jpg"},
         {"similarity": 0.91, "image_path": "path/to/image3.jpg"},
         {"similarity": 0.89, "image_path": "path/to/image4.jpg"},
         {"similarity": 0.88, "image_path": "path/to/image5.jpg"}
     ]
### Trained Model Results
  2. **Trained Model Results**
     ```json
      [
          {"similarity": 0.98, "image_path": "path/to/image6.jpg"},
          {"similarity": 0.90, "image_path": "path/to/image7.jpg"},
          {"similarity": 0.88, "image_path": "path/to/image8.jpg"},
          {"similarity": 0.85, "image_path": "path/to/image9.jpg"},
          {"similarity": 0.82, "image_path": "path/to/image10.jpg"}
      ]
### Combined Results
  3. **Combined Results**
     ```json
        [
            {
                "image_path": "path/to/image11.jpg",
                "combined_score": 0.96,
                "normal_similarity": 0.93,
                "trained_similarity": 0.98
            },
            {
                "image_path": "path/to/image12.jpg",
                "combined_score": 0.94,
                "normal_similarity": 0.91,
                "trained_similarity": 0.97
            },
            {
                "image_path": "path/to/image13.jpg",
                "combined_score": 0.92,
                "normal_similarity": 0.89,
                "trained_similarity": 0.95
            },
            {
                "image_path": "path/to/image14.jpg",
                "combined_score": 0.91,
                "normal_similarity": 0.88,
                "trained_similarity": 0.94
            },
            {
                "image_path": "path/to/image15.jpg",
                "combined_score": 0.89,
                "normal_similarity": 0.85,
                "trained_similarity": 0.93
            }
        ]

