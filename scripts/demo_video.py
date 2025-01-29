import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model
model = torch.load('/models_card_locator/best_model.pth')
model.eval()

# Prepare the image
image_path = 'path_to_test_image.jpg'  # Example image path
image = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run the model
with torch.no_grad():
    prediction = model(image_tensor)

# Visualize results
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Draw bounding boxes
for element in range(len(prediction[0]['boxes'])):
    box = prediction[0]['boxes'][element].cpu().numpy()
    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'))

plt.show()
