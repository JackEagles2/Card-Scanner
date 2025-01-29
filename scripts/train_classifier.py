import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
import argparse

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier using MobileNetV3')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the augmented dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save model checkpoints and plots')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-interval', type=float, default=0.1, help='Save model every X epochs')
    parser.add_argument('--model-best-name', type=str, required=True, help='Filename to save the best model as')
    
    return parser.parse_args()

# Configuration (using parsed arguments)
args = parse_args()

input_folder = args.data_dir  # Path to the transformed dataset
batch_size = args.batch_size  # Batch size for training
num_epochs = args.num_epochs  # Number of epochs
learning_rate = args.learning_rate  # Learning rate
save_interval = args.save_interval  # Save model every X epochs
best_model_name = args.model_best_name  # Name for saving the best model

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Load datasets using ImageFolder
train_dir = os.path.join(input_folder, 'train')
val_dir = os.path.join(input_folder, 'val')
test_dir = os.path.join(input_folder, 'test')

train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained MobileNetV3 model (Large version)
model = models.mobilenet_v3_large(pretrained=True)

# Modify the classifier to match the number of classes in the dataset
num_classes = len(train_dataset.classes)  # Number of unique classes in the dataset
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Metrics tracking
train_accuracies = []
val_accuracies = []
epoch_precisions = []
epoch_recalls = []
epoch_f1_scores = []

def calculate_metrics(predictions, labels):
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return precision, recall, accuracy

# Training Loop
best_val_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_preds = []
    epoch_labels = []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions and labels for metrics calculation
        epoch_preds.extend(predicted.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    precision, recall, train_f1 = calculate_metrics(epoch_preds, epoch_labels)

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    epoch_preds = []
    epoch_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for metrics calculation
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_precision, val_recall, val_f1 = calculate_metrics(epoch_preds, epoch_labels)

    # Save metrics for plotting
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    epoch_precisions.append(val_precision)
    epoch_recalls.append(val_recall)
    epoch_f1_scores.append(val_f1)

    # Print training and validation results
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Accuracy: {val_accuracy:.2f}%, Val Precision: {val_precision:.2f}, "
          f"Val Recall: {val_recall:.2f}, Val F1: {val_f1:.2f}")

    # Save the model and plots every X epochs
    if (epoch + 1) % int(num_epochs * save_interval) == 0:
        # Create a folder for this epoch inside the output directory
        epoch_folder = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_folder, exist_ok=True)

        # Save model checkpoint
        checkpoint_path = os.path.join(epoch_folder, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at epoch {epoch+1} to {checkpoint_path}")

        # Save Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch+2), train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(range(1, epoch+2), val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(epoch_folder, 'accuracy_plot.png'))

        # Save Precision, Recall, F1 Score Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch+2), epoch_precisions, label='Validation Precision', color='orange')
        plt.plot(range(1, epoch+2), epoch_recalls, label='Validation Recall', color='red')
        plt.plot(range(1, epoch+2), epoch_f1_scores, label='Validation F1 Score', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Validation Precision, Recall, F1 Score')
        plt.legend()
        plt.savefig(os.path.join(epoch_folder, 'metrics_plot.png'))

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_path = os.path.join(args.output_dir, best_model_name)
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved as {best_model_path}")

# Plot the final accuracy and metrics (after training completes)
epochs = list(range(1, num_epochs+1))

# Final Accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy (Final)')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'final_accuracy_plot.png'))

# Final Precision, Recall, F1 Score plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_precisions, label='Validation Precision', color='orange')
plt.plot(epochs, epoch_recalls, label='Validation Recall', color='red')
plt.plot(epochs, epoch_f1_scores, label='Validation F1 Score', color='purple')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Validation Precision, Recall, F1 Score (Final)')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'final_metrics_plot.png'))

# Test the model
model.load_state_dict(torch.load(best_model_path))
model.eval()
correct = 0
total = 0
epoch_preds = []
epoch_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions and labels for metrics calculation
        epoch_preds.extend(predicted.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
test_precision, test_recall, test_f1 = calculate_metrics(epoch_preds, epoch_labels)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")
