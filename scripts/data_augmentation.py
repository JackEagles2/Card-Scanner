import os
import random
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
import concurrent.futures
import argparse

# Configuration
min_card_coverage = 0.85  # Card must occupy at least 85% of the background 

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply data augmentation to images.')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the input image dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the augmented dataset')
    parser.add_argument('--background-dir', type=str, required=True, help='Path to background images for augmentation')
    parser.add_argument('--augmentations-per-image-train', type=int, default=5, help='Number of augmentations per image for the training set')
    parser.add_argument('--augmentations-per-image-val-test', type=int, default=1, help='Number of augmentations per image for validation/test sets')

    return parser.parse_args()

def add_random_background(object_image, background_folder, background_files):
    """Place the object image on a random background, allowing the card to go off the background slightly (up to 20%)."""
    background_file = random.choice(background_files)
    background_path = os.path.join(background_folder, background_file)

    # Check if the background file exists and is a valid image
    if not os.path.exists(background_path):
        raise ValueError(f"Background file not found: {background_path}")

    background = Image.open(background_path).convert("RGBA")

    # Get dimensions of the card and background
    bg_width, bg_height = background.size
    obj_width, obj_height = object_image.size

    # Randomly determine the coverage percentage (between 85% and 100%)
    coverage = random.uniform(min_card_coverage, 1.0)

    # Calculate the required background size so the card covers the desired percentage
    required_bg_width = int(obj_width / coverage)
    required_bg_height = int(obj_height / coverage)

    # Resize the background to meet the requirement
    background = background.resize((required_bg_width, required_bg_height), Image.Resampling.LANCZOS)

    # Calculate the max offset allowed (20% of the card's width and height)
    max_x_offset = int(obj_width * 0.2)
    max_y_offset = int(obj_height * 0.2)

    # Randomly determine the position of the card, with the possibility of it going off the background by up to 20%
    x_pos = random.randint(-max_x_offset, max_x_offset)  # Allow the card to go off the left/right
    y_pos = random.randint(-max_y_offset, max_y_offset)  # Allow the card to go off the top/bottom

    # Paste the card onto the resized background, accounting for transparency
    background.paste(object_image, (x_pos, y_pos), object_image if object_image.mode == 'RGBA' else None)
    return background

def manual_rotate(image, degrees):
    """Manually rotate the image and maintain transparency."""
    rotated_image = image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))  # Transparent background
    return rotated_image

def save_bg(image, target_folder, base_name, i, background_folder, background_files):
    """Save the image with a random background."""
    rotated_img = manual_rotate(image, random.randint(-10, 10))
    augmented_with_nobg = add_random_background(rotated_img, background_folder, background_files)
    
    # Ensure the directory exists before saving
    os.makedirs(target_folder, exist_ok=True)

    # Save image
    image_path = os.path.join(target_folder, f"{base_name}_aug_{i+1}_bg.webp")
    augmented_with_nobg.save(image_path)

def save_degrade(image, target_folder, base_name, i, background_folder, background_files):
    """
    Degrade the image by resizing, adding blur, noise, and adjusting brightness/contrast.
    Then, apply a random background and save the result, ensuring the card covers at least 85% of the background.
    """
    # Rotate the image randomly within a range of -10 to 10 degrees
    rotated_img = manual_rotate(image, random.randint(-10, 10))
    
    # Get the dimensions of the rotated image
    width, height = rotated_img.size
    
    # Step 1: Resize down and then back up to introduce pixelation
    temp_size = (int(width * 0.5), int(height * 0.5))  # Downscale to 50%
    degraded_image = rotated_img.resize(temp_size, Image.Resampling.NEAREST)  # Use NEAREST for pixelation
    degraded_image = degraded_image.resize((width, height), Image.Resampling.NEAREST)  # Upscale back
    
    # Step 2: Add slight blurring
    degraded_image = degraded_image.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Step 3: Adjust brightness and contrast randomly
    enhancer = ImageEnhance.Brightness(degraded_image)
    degraded_image = enhancer.enhance(random.uniform(0.8, 1.2))  # Random brightness adjustment
    enhancer = ImageEnhance.Contrast(degraded_image)
    degraded_image = enhancer.enhance(random.uniform(0.8, 1.2))  # Random contrast adjustment
    
    # Step 4: Add random noise to simulate compression artifacts
    noisy_image = degraded_image.copy()
    pixels = noisy_image.load()
    for y in range(noisy_image.size[1]):
        for x in range(noisy_image.size[0]):
            if random.random() < 0.1:  # 10% chance to add noise
                r, g, b, a = pixels[x, y]
                noise = random.randint(-50, 50)
                pixels[x, y] = (
                    min(max(r + noise, 0), 255),
                    min(max(g + noise, 0), 255),
                    min(max(b + noise, 0), 255),
                    a
                )
    
    # Step 5: Apply a random background using the add_random_background function
    augmented_with_bg = add_random_background(noisy_image, background_folder, background_files)
    
    # Ensure the directory exists before saving
    os.makedirs(target_folder, exist_ok=True)

    # Save image
    image_path = os.path.join(target_folder, f"{base_name}_aug_{i+1}_degrade.webp")
    augmented_with_bg.save(image_path)

def augment_and_save_image(image_path, target_folder, augmentations_per_image, background_folder, background_files):
    """Apply augmentations (with and without backgrounds) and save augmented images."""
    try:
        img = Image.open(image_path).convert("RGBA")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(augmentations_per_image):
            save_bg(img, target_folder, base_name, i, background_folder, background_files)
            save_degrade(img, target_folder, base_name, i, background_folder, background_files)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def create_standard_dataset_with_class_folders(input_folder, output_folder, augmentations_per_image_train, augmentations_per_image_val_test, background_folder, background_files):
    """Transform dataset to a standard train/val/test layout with class-specific folders."""
    os.makedirs(output_folder, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

    # Collect all tasks to track in one progress bar
    tasks = []
    for set_name in os.listdir(input_folder):
        set_folder = os.path.join(input_folder, set_name)
        if not os.path.isdir(set_folder):
            continue

        card_files = [f for f in os.listdir(set_folder) if f.endswith(".webp")]
        for card_file in card_files:
            class_name = f"{set_name}_{os.path.splitext(card_file)[0]}"
            image_path = os.path.join(set_folder, card_file)

            for split in splits:
                class_split_folder = os.path.join(output_folder, split, class_name)
                os.makedirs(class_split_folder, exist_ok=True)

                if split != "train":
                    augmentations_per_image = augmentations_per_image_val_test
                else:
                    augmentations_per_image = augmentations_per_image_train

                # Add tasks for concurrent processing, now including background_folder and background_files
                tasks.append((image_path, class_split_folder, augmentations_per_image, background_folder, background_files))

    # Create a single progress bar for the entire process
    with tqdm(total=len(tasks), desc="Processing images", unit="image") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use map to ensure sequential execution (and progress bar updates only after completion)
            for _ in executor.map(lambda task: augment_and_save_image(*task), tasks):
                pbar.update(1)

    print(f"Dataset transformed and saved to {output_folder}.")

# Main function to process the images and apply augmentations
def main():
    # Parse command line arguments
    args = parse_args()

    input_folder = args.input_dir
    output_folder = args.output_dir
    background_folder = args.background_dir
    augmentations_per_image_train = args.augmentations_per_image_train
    augmentations_per_image_val_test = args.augmentations_per_image_val_test

    # Pre-load background images to avoid reading from disk each time
    background_files = os.listdir(background_folder)
    if not background_files:
        raise ValueError(f"Background folder '{background_folder}' is empty.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create the dataset with class-specific folders
    create_standard_dataset_with_class_folders(input_folder, output_folder, augmentations_per_image_train, augmentations_per_image_val_test, background_folder, background_files)

# Run the script
if __name__ == "__main__":
    main()
