import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

class CardDataset(Dataset):
    """Custom dataset for loading Pokémon card images and annotations with segmentation masks."""

    def __init__(self, images_dir, annotations_path, transform=None):
        """
        Args:
            images_dir (str): Path to the folder containing image files (PNG).
            annotations_path (str): Path to the JSON file containing annotations.
            transform (callable, optional): Optional transform to be applied to both the image and annotations.
        """
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        # Load the annotations JSON file
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        # Create a mapping from image IDs to image file names
        self.image_id_to_file = {img["id"]: img["file_name"] for img in self.annotations["images"]}

        # Get all image IDs
        self.image_ids = list(self.image_id_to_file.keys())

        print(f"Found {len(self.image_ids)} images in {self.images_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """Load image, bounding boxes, labels, and masks."""
        image_id = self.image_ids[idx]
        image_file = self.image_id_to_file[image_id]
        image_path = self.images_dir / image_file
        image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB

        # Filter annotations for the current image
        image_annotations = [ann for ann in self.annotations["annotations"] if ann["image_id"] == image_id]

        boxes = []
        labels = []
        masks = []

        # Load bounding boxes, labels, and masks from annotations
        for annotation in image_annotations:
            bbox = annotation.get('bbox')
            segmentation = annotation.get('segmentation', [])

            # Assign label (1 for 'card', 0 for background)
            label = annotation.get('category_id', 0)

            # Handling bounding box
            if bbox and len(bbox) == 4:
                x1, y1, width, height = bbox
                boxes.append([x1, y1, x1 + width, y1 + height])  # Convert to [x_min, y_min, x_max, y_max]
                labels.append(label)

            # Handling segmentation (polygon)
            for seg in segmentation:
                if isinstance(seg, list):
                    # Convert the polygon segmentation to a binary mask
                    mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
                    polygon = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 1)  # Fill the polygon in the mask
                    masks.append(mask)
                else:
                    masks.append(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))

        # Convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # Convert masks to a single numpy array before converting to tensor
        masks = np.stack(masks, axis=0) if masks else np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
        masks = torch.tensor(masks, dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        # Apply transformation (to both image and masks)
        if self.transform:
            image = self.transform(image)

        return image, target