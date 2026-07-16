import os
import torch
from torchvision.io import read_image, write_jpeg, ImageReadMode
import torchvision.transforms.functional as F

# --- CONFIGURATION ---
# Path to your HUGE images
input_folder = r"C:\Users\ASUS\Desktop\yolo_train\yolo_dataset\val\images"

# Path where the compressed images will be saved
output_folder = r"C:\Users\ASUS\Desktop\yolo_train\compress"


target_size = 1280


jpeg_quality = 90
# ---------------------

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processing on device: {device}")

files = [
    f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
total_files = len(files)

print(f"Found {total_files} images. Starting compression...")

for i, filename in enumerate(files):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(
        output_folder, filename
    )  # Keep same filename for labels match

    try:
        # 1. Read Image (Loads to CPU first)
        # We use RGB mode to avoid alpha channel issues with JPEGs
        img_tensor = read_image(input_path, mode=ImageReadMode.RGB)

        # 2. Move to GPU
        img_tensor = img_tensor.to(device)

        # 3. Resize on GPU
        # 'antialias=True' is crucial for downsizing large images to avoid artifacts
        img_resized = F.resize(img_tensor, target_size, antialias=True)

        # 4. Save to Disk
        # We must move back to CPU to save files
        # Note: write_jpeg creates the file directly from the tensor
        write_jpeg(img_resized.cpu(), output_path, quality=jpeg_quality)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_files} images...")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nDone! Compressed images are in:", output_folder)
print("You can now replace your original 'images' folder with this new one.")
