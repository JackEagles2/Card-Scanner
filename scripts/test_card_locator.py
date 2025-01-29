import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.dataset import CardDataset, collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tools.metrics import test
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations for the data (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Dataset paths
    test_images_dir = Path(args.dataset) / "test" / "images"
    test_labels_dir = Path(args.dataset) / "test" / "labels"
    
    # Create datasets
    test_dataset = CardDataset(test_images_dir, test_labels_dir, transform)
    
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize the model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Sequential(
        torch.nn.Linear(in_features, 2)  # 2 classes: "background" and "card"
    )
    model.to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(args.model_weights))
    
    # Test the model
    test(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test object detection model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset location")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--model-weights", type=str, required=True, help="Path to the trained model weights")
    args = parser.parse_args()
    
    main(args)
