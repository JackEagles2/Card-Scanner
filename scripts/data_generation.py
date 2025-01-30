import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
import numpy as np
import cv2

def get_random_card(card_paths, bg_width, bg_height):
    """Select a random Pokémon card and resize it to fit within the background, maintaining aspect ratio."""
    card_path = random.choice(card_paths)
    card = Image.open(card_path).convert("RGBA")
    
    width, height = card.size
    aspect_ratio = width / height
    max_width = int(bg_width * 0.3)
    max_height = int(bg_height * 0.3)
    
    if width > height:
        new_width = min(width, max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(height, max_height)
        new_width = int(new_height * aspect_ratio)
    
    card = card.resize((new_width, new_height), Image.Resampling.LANCZOS)
    card = card.rotate(random.randint(-15, 15), expand=True)
    
    return card

def check_overlap(card_bbox, placed_cards):
    """Check if a card's bounding box overlaps with any previously placed cards."""
    x1, y1, x2, y2 = card_bbox
    for card in placed_cards:
        px1, py1, px2, py2 = card["bbox"]
        if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):
            return True
    return False

def place_cards_on_background(background, card_paths, min_distance=30):
    """Place random Pokémon cards on the background, ensuring some spacing and occasional overlap."""
    bg = background.copy().convert("RGBA")
    bg_width, bg_height = bg.size
    
    placed_cards = []
    segmentation_masks = []
    for _ in range(3):
        card = get_random_card(card_paths, bg_width, bg_height)
        card_width, card_height = card.size
        
        attempts = 0
        while attempts < 50:
            x = random.randint(0, max(0, bg_width - card_width))
            y = random.randint(0, max(0, bg_height - card_height))
            
            card_bbox = [x, y, x + card_width, y + card_height]  # Store as a list
            
            if check_overlap(card_bbox, placed_cards):
                attempts += 1
                continue
            
            bg.paste(card, (x, y), card)
            
            mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
            card_array = np.array(card.convert("L"))
            mask[y:y+card_height, x:x+card_width] = card_array[:card_height, :card_width]
            
            placed_cards.append({
                "class": "card",
                "bbox": card_bbox  # Store as a list
            })
            
            segmentation_masks.append(mask)
            break
        else:
            print(f"Warning: Could not place card after {attempts} attempts.")
    
    # Extract bboxes as a list of lists
    bboxes = [card["bbox"] for card in placed_cards]
    
    return bg, bboxes, segmentation_masks

def generate_dataset(output_dir, card_dir, bg_dir, dataset_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Generate a dataset with Pokémon cards placed on backgrounds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset into train, val, and test directories
    images_dir_train = output_dir / "train" / "images"
    labels_dir_train = output_dir / "train" / "annotations"
    images_dir_val = output_dir / "val" / "images"
    labels_dir_val = output_dir / "val" / "annotations"
    images_dir_test = output_dir / "test" / "images"
    labels_dir_test = output_dir / "test" / "annotations"
    
    # Create subdirectories
    images_dir_train.mkdir(parents=True, exist_ok=True)
    labels_dir_train.mkdir(parents=True, exist_ok=True)
    images_dir_val.mkdir(parents=True, exist_ok=True)
    labels_dir_val.mkdir(parents=True, exist_ok=True)
    images_dir_test.mkdir(parents=True, exist_ok=True)
    labels_dir_test.mkdir(parents=True, exist_ok=True)
    
    # Get the next available index to append to
    next_index = 0
    
    # Use glob to include all subdirectories with '**'
    card_paths = list(Path(card_dir).rglob("*.png")) + list(Path(card_dir).rglob("*.jpg")) + list(Path(card_dir).rglob("*.webp"))
    bg_paths = list(Path(bg_dir).glob("*.png")) + list(Path(bg_dir).glob("*.jpg"))
    
    if not card_paths:
        raise ValueError(f"No card images found in the directory: {card_dir}")
    if not bg_paths:
        raise ValueError(f"No background images found in the directory: {bg_dir}")
    
    # Calculate split sizes
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    # Initialize COCO annotation dictionaries for each split
    coco_train = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "card", "supercategory": "none"}]}
    coco_val = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "card", "supercategory": "none"}]}
    coco_test = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "card", "supercategory": "none"}]}
    
    annotation_id = 1  # Unique ID for each annotation
    
    for i in tqdm(range(dataset_size), desc="Generating Dataset", unit="image"):
        background = Image.open(random.choice(bg_paths))
        image, bboxes, masks = place_cards_on_background(background, card_paths)
        
        # Determine the split
        if i < train_size:
            split = "train"
            images_dir = images_dir_train
            coco_data = coco_train
        elif i < train_size + val_size:
            split = "val"
            images_dir = images_dir_val
            coco_data = coco_val
        else:
            split = "test"
            images_dir = images_dir_test
            coco_data = coco_test
        
        image_path = images_dir / f"{next_index:06d}.png"
        
        # Save the generated image
        image.save(image_path)
        
        # Add image info to COCO annotations
        coco_data["images"].append({
            "id": next_index,
            "file_name": f"{next_index:06d}.png",
            "height": image.height,
            "width": image.width
        })
        
        # Add bounding boxes and segmentation masks for each card placed
        for bbox, mask in zip(bboxes, masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = [contour.flatten().tolist() for contour in contours]
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": next_index,
                "category_id": 1,  # 'card' category
                "segmentation": segmentation,
                "area": cv2.contourArea(contours[0]),
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Convert to [x_min, y_min, width, height]
                "iscrowd": 0
            })
            annotation_id += 1
        
        next_index += 1
    
    # Save COCO annotations for each split
    with open(labels_dir_train / "instances_train.json", "w") as f:
        json.dump(coco_train, f, indent=4)
    with open(labels_dir_val / "instances_val.json", "w") as f:
        json.dump(coco_val, f, indent=4)
    with open(labels_dir_test / "instances_test.json", "w") as f:
        json.dump(coco_test, f, indent=4)
    
    print(f"Generated {dataset_size} images and annotations in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/dataset", help="Output dataset directory")
    parser.add_argument("--cards", type=str, default="data/pokemon_cards", help="Directory containing Pokémon card images")
    parser.add_argument("--backgrounds", type=str, default="data/backgrounds", help="Directory containing background images")
    parser.add_argument("--dataset-size", type=int, required=True, help="Total number of images to generate")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of dataset to be used for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of dataset to be used for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of dataset to be used for testing")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.cards, args.backgrounds, args.dataset_size, args.train_ratio, args.val_ratio, args.test_ratio)