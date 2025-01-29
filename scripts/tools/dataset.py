import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CardDataset(Dataset):
    """Custom dataset for loading Pokémon card images and annotations."""
    
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_paths = list(Path(images_dir).glob("*.png"))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label_path = Path(self.labels_dir) / f"{image_path.stem}.json"
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        
        boxes = []
        labels = []
        
        for bbox in annotations["bboxes"]:
            x1, y1, x2, y2 = bbox
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # Class 1 (card) for all instances
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
