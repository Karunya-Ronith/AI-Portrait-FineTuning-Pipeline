from PIL import Image
import os

# Change to the Leonardo DiCaprio directory
os.chdir("Leonardo DiCaprio")

# Get all jpg files
jpg_files = [f for f in os.listdir(".") if f.endswith(".jpg")]
print(f"Total images: {len(jpg_files)}")

# Check first image properties
if jpg_files:
    img = Image.open(jpg_files[0])
    print(f"First image size: {img.size}")
    print(f"First image mode: {img.mode}")
    
    # Check a few more images for size consistency
    sizes = []
    for i in range(min(5, len(jpg_files))):
        img = Image.open(jpg_files[i])
        sizes.append(img.size)
    
    print(f"Sample image sizes: {sizes}") 