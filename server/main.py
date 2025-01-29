import os
import argparse
import faiss
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict

# Initialize FastAPI
app = FastAPI()

# Define CORS policy (you can modify this to allow your frontend)
origins = [
    "http://localhost:3000",  # Allow requests from localhost:3000 (your frontend)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from the specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Set up device
device = torch.device("cpu")

# Define the base directory for the project (the root folder)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Preload the pretrained MobileNetV3 model
normal_model = models.mobilenet_v3_large(pretrained=True)
normal_model.eval().to(device)

# Load the trained MobileNetV3 model
def load_trained_model(model_name):
    trained_model = models.mobilenet_v3_large(pretrained=False)
    num_classes = 20183  # Adjust this based on your dataset
    trained_model.classifier[3] = torch.nn.Linear(trained_model.classifier[3].in_features, num_classes)
    model_path = os.path.join(base_dir, "models", model_name)
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model.eval().to(device)
    return trained_model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load precomputed features and image paths
def load_features_and_paths(features_name, image_paths_name):
    features_path = os.path.join(base_dir, "results", features_name)
    image_paths_path = os.path.join(base_dir, "results", image_paths_name)
    features = np.load(features_path)
    image_paths = np.load(image_paths_path)
    return features, image_paths

# Normalize features and build Faiss index
def build_faiss_index(features):
    faiss.normalize_L2(features)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index

# Extract features using a specific model
def extract_features(image_bytes: BytesIO, model):
    img = Image.open(image_bytes).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.features(img)
        feats = feats.view(feats.size(0), -1).cpu().numpy()
    faiss.normalize_L2(feats)
    return feats

# Global variables for loaded models and features
normal_features, normal_image_paths = None, None
trained_features, trained_image_paths = None, None
normal_index, trained_index = None, None
trained_model = None

# Load models and features before server starts
def load_all_models_and_features(trained_model_name, trained_features_name, trained_image_paths_name):
    global normal_features, normal_image_paths, trained_features, trained_image_paths, normal_index, trained_index, trained_model

    # Load precomputed features and image paths for normal and trained models
    normal_features, normal_image_paths = load_features_and_paths(trained_features_name, trained_image_paths_name)
    trained_features, trained_image_paths = load_features_and_paths(trained_features_name, trained_image_paths_name)

    # Build Faiss indices for both models
    normal_index = build_faiss_index(normal_features)
    trained_index = build_faiss_index(trained_features)

    # Load the trained model
    trained_model = load_trained_model(trained_model_name)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        content = await file.read()

        # Extract features using both models
        query_features_normal = extract_features(BytesIO(content), normal_model)
        query_features_trained = extract_features(BytesIO(content), trained_model)

        # Perform similarity search with Faiss for the normal model
        D_normal, I_normal = normal_index.search(query_features_normal, 200)
        normal_results = [{"similarity": float(D_normal[0][i]), "image_path": normal_image_paths[I_normal[0][i]]} for i in range(200)]

        # Perform similarity search with Faiss for the trained model
        D_trained, I_trained = trained_index.search(query_features_trained, 200)
        trained_results = [{"similarity": float(D_trained[0][i]), "image_path": trained_image_paths[I_trained[0][i]]} for i in range(200)]

        # Define weights for the normal and trained models
        normal_weight = 0.6  # Weight for the normal model similarity
        trained_weight = 0.4  # Weight for the trained model similarity

        # Merge results by image path (dictionary with image_path as key)
        results_dict = defaultdict(lambda: {"normal_similarity": None, "trained_similarity": None})

        # Process normal results
        for result in normal_results:
            image_path = result['image_path']
            results_dict[image_path]["normal_similarity"] = result['similarity']

        # Process trained results
        for result in trained_results:
            image_path = result['image_path']
            results_dict[image_path]["trained_similarity"] = result['similarity']

        # Combine the similarity scores and prepare the final result
        combined_results = []
        for image_path, similarities in results_dict.items():
            # Only include image paths that have results from both models (i.e., no None values)
            if similarities["normal_similarity"] is not None and similarities["trained_similarity"] is not None:
                # Calculate the combined score with weights
                combined_score = (similarities["normal_similarity"] * normal_weight) + (similarities["trained_similarity"] * trained_weight)

                # Append the combined result
                combined_results.append({
                    "image_path": image_path,
                    "combined_score": combined_score,
                    "normal_similarity": similarities["normal_similarity"],
                    "trained_similarity": similarities["trained_similarity"]
                })

        # Sort the results by the combined score and get the top 5
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_5_combined = combined_results[:5]

        # Return the results for both models and the combined results
        return {
            "status": "success",
            "results": {
                "normal_model": normal_results[:5],
                "trained_model": trained_results[:5],
                "combined_model": top_5_combined,  # Add the top 5 combined results
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Example route for testing
@app.get("/")
def home():
    return {"message": "Comparison between Normal and Trained MobileNetV3 is Running!"}

def run_app():
    # Parse arguments to load the proper model and features
    parser = argparse.ArgumentParser(description="Start the FastAPI server with specific models and features.")
    parser.add_argument("--trained-model-name", type=str, default="best_trained_model.pth", help="Path to the trained model file.")
    parser.add_argument("--trained-features-name", type=str, default="features_trained_model.npy", help="Path to the trained features file.")
    parser.add_argument("--trained-image-paths-name", type=str, default="image_paths_trained_model.npy", help="Path to the trained image paths file.")
    args = parser.parse_args()

    # Load models and features before starting the server
    load_all_models_and_features(args.trained_model_name, args.trained_features_name, args.trained_image_paths_name)

    # Run the FastAPI app with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_app()
