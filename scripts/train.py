import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from tools.dataset import CardDataset  # Import your custom dataset class
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a Pokémon card segmentation model.")
parser.add_argument("--train-images", type=str, required=True, help="Path to training images directory.")
parser.add_argument("--train-annotations", type=str, required=True, help="Path to training annotations JSON file.")
parser.add_argument("--val-images", type=str, required=True, help="Path to validation images directory.")
parser.add_argument("--val-annotations", type=str, required=True, help="Path to validation annotations JSON file.")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--output-dir", type=str, default="output", help="Directory to save model checkpoints.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
args = parser.parse_args()

# Validate the device argument
if args.device not in ["cuda", "cpu"]:
    raise ValueError(f"Invalid device: {args.device}. Expected 'cuda' or 'cpu'.")

# Check if CUDA is available
if args.device == "cuda" and not torch.cuda.is_available():
    print("CUDA is not available. Switching to CPU.")
    args.device = "cpu"

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize datasets
train_dataset = CardDataset(
    images_dir=args.train_images,
    annotations_path=args.train_annotations,
    transform=ToTensor()
)

val_dataset = CardDataset(
    images_dir=args.val_images,
    annotations_path=args.val_annotations,
    transform=ToTensor()
)

# Initialize data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))  # Custom collate function for detection tasks
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))
)

# Load a pre-trained ResNet50-FPN backbone
backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)

# Create the Mask R-CNN model with a custom number of classes
model = MaskRCNN(
    backbone,
    num_classes=2,  # 2 classes: background + card
    min_size=800,   # Adjust based on your input image size
    max_size=1333   # Adjust based on your input image size
)

# Move the model to the specified device
model.to(args.device)

# Define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce learning rate every 5 epochs

# Mixed precision training
scaler = GradScaler(device='cuda' if args.device == 'cuda' else 'cpu')

# Training loop
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        # Move images and targets to the specified device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
            output = model(images, targets)
            print(f"Output type: {type(output)}")  # Debugging: Check the type of output
            if isinstance(output, list):
                print(f"Output length: {len(output)}")  # Debugging: Check the length of the list
                print(f"First element type: {type(output[0])}")  # Debugging: Check the type of the first element

            # Ensure the output is a dictionary
            if isinstance(output, dict):
                losses = sum(loss for loss in output.values())
            else:
                raise ValueError(f"Unexpected output type: {type(output)}. Expected a dictionary of losses.")

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")

# Validation loop
def validate(model, data_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            # Move images and targets to the specified device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Debugging: Inspect the output of model(images, targets)
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                output = model(images, targets)
                print(f"Output type: {type(output)}")  # Debugging: Check the type of output
                if isinstance(output, list):
                    print(f"Output length: {len(output)}")  # Debugging: Check the length of the list
                    print(f"First element type: {type(output[0])}")  # Debugging: Check the type of the first element

                # Ensure the output is a dictionary
                if isinstance(output, dict):
                    losses = sum(loss for loss in output.values())
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}. Expected a dictionary of losses.")

            total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# Main training function
def train():
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, optimizer, train_loader, args.device, epoch + 1)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        #val_loss = validate(model, val_loader, args.device)

        # Save the best model
        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        # Step the learning rate scheduler
        scheduler.step()

    print("Training complete!")

# Run training
if __name__ == "__main__":
    train()