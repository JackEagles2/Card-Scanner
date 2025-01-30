import torch
import torchvision
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tools.dataset import CardDataset
from torchvision.models.detection import mask_rcnn

def get_model(num_classes: int):
    # Load a pre-trained Mask R-CNN model from torchvision
    model = mask_rcnn.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    # Get the number of input channels for the box predictor and mask predictor
    in_channels = model.roi_heads.box_predictor.cls_score.in_features

    # Modify the box predictor for your custom dataset
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_channels, num_classes)

    # Get the correct number of input channels for the mask predictor
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Modify the mask predictor for your custom dataset
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels_mask, 256, num_classes)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model


def load_model(model, model_path):
    # Load the model's state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def draw_boxes(image, boxes, labels, scores):
    """
    Draw bounding boxes on the image without using a confidence threshold.
    Args:
        image: PIL Image
        boxes: Bounding box coordinates (list of [x1, y1, x2, y2])
        labels: List of labels for each box
        scores: List of scores for each box
    """
    draw = ImageDraw.Draw(image)
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score > 0.21:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=5)  # Change color and increase line width
            draw.text((x1, y1), f"Class {labels[i]}: {score:.2f}", fill="red")
        
    return image

def inference(model, test_loader, device):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]

            # Perform inference
            prediction = model(images)

            # Store the predictions and ground truths for evaluation
            for i, image in enumerate(images):
                predictions.append(prediction[i])
                ground_truths.append(targets[i])


                # Get the predicted boxes, labels, and scores
                boxes = prediction[i]['boxes'].cpu().numpy()
                labels = prediction[i]['labels'].cpu().numpy()
                scores = prediction[i]['scores'].cpu().numpy()

                # Convert the image tensor to a PIL image for drawing
                image_pil = F.to_pil_image(images[i].cpu())

                # Draw the boxes on the image
                image_with_boxes = draw_boxes(image_pil, boxes, labels, scores)

                # Show the image with boxes drawn
                image_with_boxes.show()  # This will display the image

def collate_fn(batch):
    # This ensures that the data is packed into a tuple of images and targets
    return tuple(zip(*batch))

def main():
    # Define the paths
    model_path = 'output/best_model.pth'
    test_images_dir = Path('data/dataset/test/images')
    test_labels_dir = Path('data/dataset/test/annotations/instances_test.json')  # Use an empty string or a dummy path for labels_dir

    # Get the model
    num_classes = 2  # Background + card
    model = get_model(num_classes)

    # Load the trained model
    model = load_model(model, model_path)

    # Prepare the test dataset and dataloader
    transform = None  # No transform needed for inference
    test_dataset = CardDataset(test_images_dir,test_labels_dir, transform=transform)  # labels_dir can be empty for inference
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run inference and display predictions
    inference(model, test_loader, device)


if __name__ == "__main__":
    main()
