import os
import torch
import random
import argparse
from pathlib import Path
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tools.dataset import CardDataset, collate_fn
from tools.metrics import test

def train(model, train_loader, val_loader, optimizer, num_epochs, device, model_output_dir):
    """Train the model."""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss}")
        
        # Validation
        validate(model, val_loader, device)
        
        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(model_output_dir, f"model_epoch_{epoch+1}.pth"))
        
    print("Training complete.")

def validate(model, val_loader, device):
    """Validate the model on the validation set."""
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Total loss
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
        
        print(f"Validation Loss: {running_loss}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations for the data (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Dataset paths
    train_images_dir = Path(args.dataset) / "train" / "images"
    train_labels_dir = Path(args.dataset) / "train" / "labels"
    val_images_dir = Path(args.dataset) / "val" / "images"
    val_labels_dir = Path(args.dataset) / "val" / "labels"
    
    # Create datasets
    train_dataset = CardDataset(train_images_dir, train_labels_dir, transform)
    val_dataset = CardDataset(val_images_dir, val_labels_dir, transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize the model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Sequential(
        torch.nn.Linear(in_features, 2)  # 2 classes: "background" and "card"
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create model output directory if it doesn't exist
    model_output_dir = Path(args.model_output)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    train(model, train_loader, val_loader, optimizer, args.num_epochs, device, model_output_dir)
    
    # Test the model (optional)
    test_images_dir = Path(args.dataset) / "test" / "images"
    test_labels_dir = Path(args.dataset) / "test" / "labels"
    test_dataset = CardDataset(test_images_dir, test_labels_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset location")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model-output", type=str, default="/models_card_locator", help="Directory to save the trained model")
    args = parser.parse_args()
    
    main(args)
