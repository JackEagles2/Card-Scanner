import os
import faiss
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import argparse

# Set the device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_weights_path=None):
    # Load the model architecture (for example, a pretrained MobileNetV3)
    model = models.mobilenet_v3_large(pretrained=True)  # Load pretrained MobileNetV3
    
    # Modify the classifier to match the number of classes in your custom model
    num_classes = 20183  # This should be the number of classes in your dataset
    model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    
    # If a custom model path is provided, load the custom model weights
    if model_weights_path:
        print(f"Loading custom model weights from: {model_weights_path}")
        model_weights = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(model_weights, strict=False)  # strict=False allows ignoring missing/extra keys
    else:
        print("Loading pretrained MobileNetV3 weights.")

    return model

def extract_features(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.to(device)  # Ensure image is on the correct device
    
    with torch.no_grad():
        if hasattr(model, 'features'):
            # If the model has a 'features' attribute (like MobileNetV3)
            features = model.features(img)
        else:
            # Otherwise, we assume it's a standard model with the forward pass
            features = model(img)

        # Flatten the features
        features = features.view(features.size(0), -1)  # Flatten the feature tensor

    # Move the tensor to CPU and convert to numpy
    return features.cpu().numpy()  # Move to CPU before calling .numpy()

# Precompute features for all images in selected folders
def precompute_features(cards_dir, output_folder, output_name, model):
    image_paths = []
    all_features = []

    # List the folders you want to process
    folders_to_process = []
    for i, folder in enumerate(os.listdir(cards_dir)):
        folders_to_process.append(folder)
    
    # Process each folder with a progress bar
    with tqdm(total=len(folders_to_process), desc="Processing Folders", unit="folder") as pbar:
        for folder in folders_to_process:
            folder_path = os.path.join(cards_dir, folder)
            
            if os.path.isdir(folder_path):
                # Process images inside each folder
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_path = os.path.join(folder_path, file)
                        image_paths.append(image_path)
                        features = extract_features(image_path, model)
                        all_features.append(features)

            pbar.update(1)  # Update the progress bar after each folder is processed

    # Convert the list to a numpy array
    all_features = np.vstack(all_features)

    # Construct the output file paths
    feature_output_file = os.path.join(output_folder, f"{output_name}.npy")
    image_paths_output_file = os.path.join(output_folder, f"{output_name}_image_paths.npy")

    # Save features and image paths to disk
    np.save(feature_output_file, all_features)
    np.save(image_paths_output_file, image_paths)

    print(f"Features saved to {feature_output_file}")
    print(f"Image paths saved to {image_paths_output_file}")

# Main function to parse arguments and execute feature extraction
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract features from Pokémon cards using MobileNetV3")
    parser.add_argument('--cards-dir', type=str, required=True, help="Directory containing the Pokémon card images")
    parser.add_argument('--output-folder', type=str, required=True, help="Directory to save the extracted features")
    parser.add_argument('--output-name', type=str, required=True, help="Name of the output file (without extension)")
    parser.add_argument('--model', type=str, default=None, help="Path to custom model weights (default: pretrained MobileNetV3)")

    # Parse the arguments
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Set the model to evaluation mode
    model.eval().to(device)

    # Precompute features
    precompute_features(args.cards_dir, args.output_folder, args.output_name, model)

if __name__ == '__main__':
    main()
