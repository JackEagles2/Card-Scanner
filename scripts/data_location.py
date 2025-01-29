import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm  # Import tqdm for progress bar

def get_random_card(card_paths, bg_width, bg_height):
    """Select a random Pokémon card and resize it to fit within the background, maintaining aspect ratio."""
    card_path = random.choice(card_paths)
    card = Image.open(card_path).convert("RGBA")
    
    # Get the current width and height of the card
    width, height = card.size
    
    # Calculate the aspect ratio of the card
    aspect_ratio = width / height
    
    # Set the maximum size of the card relative to the background (e.g., 30% of background width or height)
    max_width = int(bg_width * 0.3)  # Card width should not exceed 30% of the background width
    max_height = int(bg_height * 0.3)  # Card height should not exceed 30% of the background height
    
    # Resize the card to fit within the max_width and max_height, maintaining aspect ratio
    if width > height:
        # Scale based on width
        new_width = max(width, max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        # Scale based on height
        new_height = max(height, max_height)
        new_width = int(new_height * aspect_ratio)
    
    # Resize the card to the new dimensions
    card = card.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Apply random rotation (keeping the expanded canvas if necessary)
    card = card.rotate(random.randint(-15, 15), expand=True)
    
    return card


def check_overlap(card_bbox, placed_cards):
    """Check if a card's bounding box overlaps with any previously placed cards."""
    x1, y1, x2, y2 = card_bbox
    for card in placed_cards:
        px1, py1, px2, py2 = card["bbox"]
        # Check if the card overlaps with an already placed card
        if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):  # Bounding box overlap check
            return True  # Overlap found
    return False


def place_cards_on_background(background, card_paths, min_distance=30):
    """Place random Pokémon cards on the background, ensuring some spacing and occasional overlap."""
    bg = background.copy().convert("RGBA")
    bg_width, bg_height = bg.size
    
    placed_cards = []
    for _ in range(3):
        card = get_random_card(card_paths, bg_width, bg_height)
        card_width, card_height = card.size
        
        # Try to place the card at a valid location, allowing some overlap
        attempts = 0
        while attempts < 50:  # Try up to 50 times to place a card
            x = random.randint(0, max(0, bg_width - card_width))  # Random x placement with boundary check
            y = random.randint(0, max(0, bg_height - card_height))  # Random y placement with boundary check
            
            card_bbox = [x, y, x + card_width, y + card_height]
            
            # Check for overlap with previously placed cards
            if check_overlap(card_bbox, placed_cards):
                attempts += 1
                continue  # Try again if overlap occurs
            
            # If no overlap, place the card
            bg.paste(card, (x, y), card)
            placed_cards.append({
                "class": "card",  # The class is now always "card"
                "bbox": card_bbox
            })
            break
        else:
            print(f"Warning: Could not place card after {attempts} attempts.")
    
    return bg, placed_cards

def get_next_index(images_dir):
    """Get the next available index based on the existing files in the images directory."""
    existing_files = list(images_dir.glob("*.png"))
    if not existing_files:
        return 0
    # Find the highest existing index (assuming filenames are numeric like 000001.png, 000002.png, etc.)
    existing_indices = [int(f.stem) for f in existing_files]
    return max(existing_indices) + 1

def generate_dataset(output_dir, card_dir, bg_dir, dataset_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Generate a dataset with Pokémon cards placed on backgrounds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset into train, val, and test directories
    images_dir_train = output_dir / "train" / "images"
    labels_dir_train = output_dir / "train" / "labels"
    images_dir_val = output_dir / "val" / "images"
    labels_dir_val = output_dir / "val" / "labels"
    images_dir_test = output_dir / "test" / "images"
    labels_dir_test = output_dir / "test" / "labels"
    
    # Create subdirectories
    images_dir_train.mkdir(parents=True, exist_ok=True)
    labels_dir_train.mkdir(parents=True, exist_ok=True)
    images_dir_val.mkdir(parents=True, exist_ok=True)
    labels_dir_val.mkdir(parents=True, exist_ok=True)
    images_dir_test.mkdir(parents=True, exist_ok=True)
    labels_dir_test.mkdir(parents=True, exist_ok=True)
    
    # Get the next available index to append to
    next_index = get_next_index(images_dir_train)
    
    # Use glob to include all subdirectories with '**'
    card_paths = list(Path(card_dir).rglob("*.png")) + list(Path(card_dir).rglob("*.jpg")) + list(Path(card_dir).rglob("*.webp"))
    bg_paths = list(Path(bg_dir).glob("*.png")) + list(Path(bg_dir).glob("*.jpg"))
    
    # Check if card_paths and bg_paths are empty
    if not card_paths:
        raise ValueError(f"No card images found in the directory: {card_dir}")
    if not bg_paths:
        raise ValueError(f"No background images found in the directory: {bg_dir}")
    
    # Calculate split sizes
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    # Add tqdm progress bar here
    for i in tqdm(range(dataset_size), desc="Generating Dataset", unit="image"):
        background = Image.open(random.choice(bg_paths))
        image, bboxes = place_cards_on_background(background, card_paths)
        
        # Determine the split
        if i < train_size:
            split = "train"
            images_dir = images_dir_train
            labels_dir = labels_dir_train
        elif i < train_size + val_size:
            split = "val"
            images_dir = images_dir_val
            labels_dir = labels_dir_val
        else:
            split = "test"
            images_dir = images_dir_test
            labels_dir = labels_dir_test
        
        image_path = images_dir / f"{next_index:06d}.png"
        label_path = labels_dir / f"{next_index:06d}.json"
        
        image.save(image_path)
        
        # Only save the bounding boxes and class info (class is always "card")
        with open(label_path, "w") as f:
            json.dump({"bboxes": bboxes}, f, indent=4)
        
        # Increment the index for the next image/label
        next_index += 1
    
    print(f"Generated {dataset_size} images and labels in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/dataset_location", help="Output dataset directory")
    parser.add_argument("--cards", type=str, default="data/pokemon_cards", help="Directory containing Pokémon card images")
    parser.add_argument("--backgrounds", type=str, default="data/backgrounds", help="Directory containing background images")
    parser.add_argument("--dataset-size", type=int, required=True, help="Total number of images to generate")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of dataset to be used for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of dataset to be used for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of dataset to be used for testing")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.cards, args.backgrounds, args.dataset_size, args.train_ratio, args.val_ratio, args.test_ratio)
