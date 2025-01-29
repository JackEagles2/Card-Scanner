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
   1. [Model For Card Location](#model-for-locating-cards)
      1. [Dataset](#dataset)
      2. [Training the Model](#training-the-model)
      3. [Testing the Model](#testing-the-model)
      4. [Demo of Locating Cards](#demo-of-locating-cards)
   2. [Model For Card Classifying](#model-for-classifying-cards)
      1. [Data Augmentation](#data-augmentation)
      2. [Feature Extraction](#feature-extraction)
      3. [Model Training](#model-training)
   3. [FastAPI Server](#fastapi-server)
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
- **Model to Predict the Location of Cards**:
  - **Dataset Generation**: The project generates a dataset by placing Pokémon cards randomly on background images, simulating realistic scenarios for object detection. The dataset is split into training, validation, and test sets.
  - **Object Detection**: The project uses the **Faster R-CNN** model, which is specifically trained for detecting Pokémon cards in various backgrounds. The model learns to identify the location of each card in an image and outputs bounding boxes.
  - **Model Training**: The Faster R-CNN model is fine-tuned using the generated dataset. It learns to predict bounding boxes around the Pokémon cards while distinguishing them from the background. The model is saved as checkpoints during training, with the best model saved based on performance metrics like Mean IoU (Intersection over Union).
  - **Evaluation Metrics**: The performance of the trained model is evaluated using standard object detection metrics such as **Precision**, **Recall**, **F1-Score**, and **Mean IoU**. These metrics help in understanding how well the model detects the cards and reduces false positives and false negatives.
  - **Model Checkpoints**: The model checkpoints are saved based on the highest validation performance, specifically using the Mean IoU metric to determine the best model for testing and deployment.
  - **Testing**: After training, the model is tested on a separate test set to measure its ability to generalize to new, unseen data.

- **Model to Predict the Card**:
  - **Data Augmentation**: The project enhances the training dataset by applying random transformations like resizing, rotation, flipping, and adding background images to generate a diverse set of training samples.
  - **Feature Extraction**: It uses both a pretrained MobileNetV3 model and a custom-trained MobileNetV3 model to extract feature embeddings from Pokémon card images. These embeddings are used for efficient similarity search and retrieval.
  - **Model Training**: The project includes the ability to train a custom MobileNetV3 model fine-tuned on the augmented Pokémon card dataset. This model is optimized for classification and can be saved as checkpoints for future use.
- **FastAPI Server**: A FastAPI-powered backend serves the models and allows real-time querying of the Pokémon cards. The server supports image upload, feature extraction, and similarity search across two models (pretrained and custom-trained).


The goal is to allow users to upload a Pokémon card image and receive a ranked list of the most similar cards based on features extracted from both a pretrained MobileNetV3 model and a custom-trained one. The results are returned in a way that combines the strengths of both models for more accurate retrieval.

This repository is organized into different sections:
- **Model to Predict the Location of Cards**:
  - **Dataset Generation**: Collection and making of the images with Cards on.
  - **Model Training**: Training a custom RCNN model using the dataset
  - **Model Checkpoints**: Extraction of the best RCNN based on metrics

- **Model to Predict the Card**:
  - **Data preparation**: Collection and augmentation of Pokémon card images.
  - **Model training**: Training a custom MobileNetV3 model on the augmented dataset.
  - **Feature extraction**: Extracting meaningful features from the cards to be used for similarity-based searches.
- **Server**: FastAPI server for uploading images and retrieving similar cards using Faiss.

This project is designed to be scalable and efficient, leveraging powerful tools like **PyTorch** for deep learning, **Faiss** for fast similarity searches, and **FastAPI** for building the backend server.

## Project Structure

The project is organized into the following directory structure:

Card-Scanner/  
├── data/  
│   ├── input/                        # Raw Pokémon card images  
│   ├── augmented_dataset/            # Augmented images with class-wise directories  
│   │   ├── test/                     # Test Dataset  
│   │   ├── train/                    # Train Dataset  
│   │   ├── val/                      # Validation Dataset  
│   ├── dataset_location/            # Dataset for model to location cards  
│   │   ├── test/                     # Test Dataset  
│   │   ├── train/                    # Train Dataset  
│   │   ├── val/                      # Validation Dataset 
│   ├── backgrounds/                  # Background images for augmentation  
├── models_features/  
│   ├── mobilenet_v3.pth              # Pretrained MobileNetV3 weights  
│   ├── checkpoints/                  # Checkpoints of trained models  
├── models_card_locator/  
│   ├── best_model.pth                # Best Model for the Card Locator  
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

### Model for locating Cards
#### Dataset

To train a model for locating Pokémon cards within images, we first need to generate a dataset of images containing cards placed on various backgrounds. The images are generated with multiple cards on each background to create diverse training examples. The dataset contains:
- **Images**: These are the background images with cards placed randomly or with some overlap.
- **Labels**: These include bounding boxes for each card in the image, indicating the location of the card on the background.

The dataset is generated with bounding box annotations that define the location of each Pokémon card. The class label for the cards is set to `"card"`, as the model does not need to distinguish between different types of cards.

We also split the dataset into **train**, **validation**, and **test** sets. The dataset is divided as follows:
- **Train**: Used for training the model.
- **Validation**: Used to evaluate the model during training to ensure it generalizes well.
- **Test**: Used for final evaluation after training to assess the model's performance.

To generate the dataset and split it, you can use the following command:

```bash
python scripts/data_generation.py --dataset-size 1000
```

#### Training the Model

Once the dataset has been generated and split into train, validation, and test sets, the next step is to train the model to detect Pokémon cards in the images. For this task, we will use an object detection model, such as **Faster R-CNN**, **YOLO**, or **RetinaNet**. These models are well-suited for identifying the location of objects (in this case, Pokémon cards) in images by predicting bounding boxes around them.

The process for training the model involves the following steps:

1. **Dataset Loading**: 
   The script automatically loads the dataset from the `train`, `val`, and `test` directories. Each image has a corresponding label file that includes the bounding boxes for the Pokémon cards.

2. **Model Initialization**: 
   The model used is **Faster R-CNN** with a **ResNet-50** backbone and **FPN (Feature Pyramid Network)**. It is pretrained on COCO, and the final classifier is replaced to predict the class `card` (along with the background).

3. **Training**: 
   The model is trained using the **Adam** optimizer and **cross-entropy loss**. The training loop runs for a specified number of epochs, printing the training loss after each epoch.

4. **Validation**: 
   After each epoch, the model is validated using the validation set to monitor the training process.

5. **Testing (optional)**: 
   After training, the model can be tested on a separate test dataset. This provides an opportunity to evaluate the model’s performance (e.g., calculate mAP or IoU).


To train the model, you can run the following command:

```bash
python scripts/train_card_locator.py --dataset data/dataset
```
#### Testing the Model

After training the model on the Pokémon card detection dataset, the next step is to evaluate its performance on the test set. This can be done by running the model on the test images and calculating performance metrics like precision, recall, F1 score, and mAP (Mean Average Precision). The following steps outline the testing procedure.

##### Steps for Testing:

1. **Dataset Loading**:  
   The script automatically loads the test dataset from the `test` directory. Each image has a corresponding label file containing the bounding boxes for the Pokémon cards. These annotations will be used to evaluate the model’s performance.

2. **Model Initialization**:  
   The model used for testing is the same Faster R-CNN model that was trained. We load the model weights saved during training.

3. **Testing**:  
   The model is evaluated on the test dataset. For each image, predictions are made, and the model's predicted bounding boxes and labels are compared to the ground truth. The following metrics are calculated:
   - **Precision**: The proportion of true positive predictions out of all positive predictions.
   - **Recall**: The proportion of true positive predictions out of all actual positive instances.
   - **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
   - **Mean Average Precision (mAP)**: Measures the accuracy of bounding box predictions at various IoU (Intersection over Union) thresholds.

4. **Metrics Calculation**:  
   The metrics are computed using the `torchmetrics` library, which provides efficient and accurate ways to calculate evaluation metrics for object detection.

5. **Visualization**:  
   After calculating the metrics, the script generates bar charts for each metric (precision, recall, F1 score, and mAP) to visually represent the model's performance.


Once you have the trained model and dataset ready, you can run the testing script using the following command:

```bash
python scripts/test_card_locator.py --dataset data/dataset --model-weights /models_card_locator/best_model.pth
```

#### Demo of Locating Cards

You can watch the video demonstrating the model detecting Pokémon cards below:

![Model Detection Demo](path_to_video/demo_video.mp4)

#### Video Highlights:
- **Accurate Card Detection**: The model correctly identifies the location of the Pokémon cards.
- **Bounding Boxes**: Bounding boxes are drawn around each detected card, showing the model's predictions.
- **Multiple Cards**: The model successfully detects multiple cards in the same image, even when they overlap.

### Model for classifying Cards
#### Data Augmentation

Data augmentation ensures diverse training examples by:
- Adding random backgrounds.
- Resizing, rotating, and flipping images.
- This helps improve the model's generalization by providing various transformations of the same image.

To augment the dataset, run:

```bash
python scripts/data_augmentation.py --input-dir data/input --output-dir data/augmented_dataset --background-dir data/backgrounds --augmentations-per-image-train 5 --augmentations-per-image-val-test 1
```
#### Feature Extraction

Feature extraction is a crucial step in creating embeddings for similarity-based retrieval. The following steps are involved:
- Extracts embeddings using a pretrained **MobileNetV3** model.
- Extracts embeddings using a custom-trained **MobileNetV3** model.
- Saves embeddings as `.npy` files, which are used for fast similarity search with **Faiss**.

To extract features, run:

```bash
python scripts/feature_extraction.py --cards-dir data/input --output-folder results --output-name trained_model --model models/best_mobilenet_v3_model.pth
```
#### Model Training

Model training involves training a **MobileNetV3** classifier on the augmented dataset. The training process consists of the following:

- Loading the augmented dataset and preprocessing the images.
- Training the classifier with a custom model (MobileNetV3) fine-tuned on the Pokémon card data.
- Saving the trained model checkpoints for future use.

To train the model, run:

```bash
python scripts/train_classifier.py --data-dir data/augmented_dataset --output-dir models/checkpoints --model-best-name best_model.pth
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
git clone git@github.com:JackEagles2/Card-Scanner.git
cd Card-Scanner
```

### Install Dependencies

For Python dependencies, navigate to the `server` folder and install the required packages:

```bash
pip install -r server/requirements.txt
```

### Required Libraries:
The project requires the following dependencies:

- **Flask**: A lightweight web framework for building the server.
- **requests**: To handle HTTP requests.
- **Pillow**: For image processing.
- **numpy**: Library for numerical operations.
- **opencv-python**: For advanced image manipulation and processing.
- **scikit-learn**: For machine learning utilities and models.
- **pandas**: For data manipulation and analysis.

### Server Requirements
The server requires the following dependencies:

- **FastAPI**: Web framework for building the backend server.
- **Faiss**: Library for efficient similarity search and retrieval.
- **PyTorch**: Deep learning framework for model inference.


## Usage
### Data Augmentation

#### To augment the dataset, run the following command:

```bash
python scripts/data_augmentation.py \
  --input-dir <input-directory> \
  --output-dir <output-directory> \
  --background-dir <background-directory> \
  --augmentations-per-image-train <augmentations-per-image-train> \
  --augmentations-per-image-val-test <augmentations-per-image-val-test>
```
#### Arguments:

| Argument                           | Description                                                       | Default Value  |
|-------------------------------------|-------------------------------------------------------------------|----------------|
| `--input-dir`                       | Path to the input dataset containing images.                      | Required       |
| `--output-dir`                      | Path to save the augmented dataset.                               | Required       |
| `--background-dir`                  | Path to background images used for augmentation.                  | Required       |
| `--augmentations-per-image-train`   | Number of augmentations to apply per image for training dataset.   | 5              |
| `--augmentations-per-image-val-test`| Number of augmentations to apply per image for validation/test.    | 1              |

#### Example Command

```bash
python scripts/data_augmentation.py \
  --input-dir data/input \
  --output-dir data/augmented_dataset \
  --background-dir data/backgrounds \
  --augmentations-per-image-train 5 \
  --augmentations-per-image-val-test 1
```

### Feature Extraction

Feature extraction is a crucial step for creating embeddings that can be used in similarity-based retrieval tasks. This process involves extracting meaningful features from images using a pre-trained or custom-trained model. In this case, we use **MobileNetV3** to extract embeddings for Pokémon cards and save them as `.npy` files for fast similarity search.

To extract features from the dataset, run the following command:

```bash
python scripts/feature_extraction.py \
  --cards-dir <cards-directory> \
  --output-folder <output-folder> \
  --output-name <output-name> \
  --model <model-path>
```
#### Arguments:

| Argument           | Description                                                                       | Default Value         |
|--------------------|-----------------------------------------------------------------------------------|-----------------------|
| `--cards-dir`       | Path to the directory containing the input images for which features will be extracted. | Required              |
| `--output-folder`   | Directory to save the extracted features.                                          | Required              |
| `--output-name`     | Name of the output file (e.g., `trained_model`) where the embeddings will be saved. | Required              |
| `--model`           | Path to the pre-trained or custom-trained model used for feature extraction.        | Required              |

#### Feature Extraction Command

To extract features using a pre-trained model, run the following command:

```bash
python scripts/feature_extraction.py \
  --cards-dir data/input \
  --output-folder results \
  --output-name trained_model \
  --model models/best_mobilenet_v3_model.pth
```

### Model Training

Model training involves training a **MobileNetV3** classifier on the augmented dataset. The training process consists of the following:

- **Loading the augmented dataset** and preprocessing the images.
- **Training the classifier** with a custom model (MobileNetV3) fine-tuned on the Pokémon card data.
- **Saving the trained model checkpoints** for future use.

To train the model, run the following command:

```bash
python scripts/train_classifier.py \
  --data-dir <data-directory> \
  --output-dir <output-directory> \
  --model-best-name <model-best-name> \
  --batch-size <batch-size> \
  --epochs <epochs> \
  --learning-rate <learning-rate> \
  --weight-decay <weight-decay> \
  --checkpoint-interval <checkpoint-interval>
```

#### Arguments:

| Argument                | Description                                                                 | Default Value         |
|-------------------------|-----------------------------------------------------------------------------|-----------------------|
| `--data-dir`             | Path to the directory containing the augmented dataset for training.        | Required              |
| `--output-dir`           | Path to save the model checkpoints.                                          | Required              |
| `--model-best-name`      | Name of the best model checkpoint file to save (e.g., `best_model.pth`).    | `best_model.pth`     |
| `--batch-size`           | Batch size used during training.                                            | 32                    |
| `--epochs`               | Number of epochs for training.                                              | 20                    |
| `--learning-rate`        | Learning rate used for training.                                            | 0.0001                |
| `--weight-decay`         | Weight decay used for regularization.                                        | 0.0001                |
| `--checkpoint-interval`  | Interval to save checkpoints (in terms of epochs).                          | 5                     |

#### Train the Model Command

To train the model, run the following command:

```bash
python scripts/train_classifier.py \
  --data-dir data/augmented_dataset \
  --output-dir models/checkpoints \
  --model-best-name best_model.pth \
  --batch-size 32 \
  --epochs 20 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --checkpoint-interval 5
```

### Start the Server

To start the FastAPI server, follow these steps:

1. **Navigate to the `server` folder**:
   In your terminal, change the directory to where the server code is located. For example:
   ```bash
   cd /server
    ```
2. **Run the Server Using `uvicorn`:**

    After navigating to the `server` folder, start the FastAPI server with the following command:

    ```bash
    uvicorn main:app --reload
    ```

#### Additional Configuration (Optional)

You can specify the paths for the trained model, features, and image paths as arguments when starting the server by using the following command:

```bash
python main.py --trained-model-name <trained_model.pth> --trained-features-name <features_trained_model.npy> --trained-image-paths-name <image_paths_trained_model.npy>
```

##### Arguments Table

| Argument                     | Description                                             | Example                               |
|------------------------------|---------------------------------------------------------|---------------------------------------|
| `--trained-model-name`        | Path to the trained model file (Always looks in /models)| `best_trained_model.pth`             |
| `--trained-features-name`     | Path to the trained features file (Always looks in /results)| `features_trained_model.npy`         |
| `--trained-image-paths-name`  | Path to the trained image paths file (Always looks in /results)| `image_paths_trained_model.npy`      |

##### Server Command Example

To start the server with the specified paths for the trained model, features, and image paths, you can use the following command:

```bash
python main.py --trained-model-name best_trained_model.pth --trained-features-name features_trained_model.npy --trained-image-paths-name image_paths_trained_model.npy
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
           "combined_model": [...],   # Top 5 combined results with weighted scores
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

