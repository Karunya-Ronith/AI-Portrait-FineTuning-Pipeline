import os
import json
from PIL import Image
from tqdm import tqdm
import argparse

def resize_and_save_image(input_path, output_path, target_size=(512, 512)):
    """Resize image to target size while maintaining aspect ratio with padding"""
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider, fit to width
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        else:
            # Image is taller, fit to height
            new_height = target_size[1]
            new_width = int(target_size[1] * img_ratio)
        
        # Resize image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calculate position to center the image
        x = (target_size[0] - new_width) // 2
        y = (target_size[1] - new_height) // 2
        
        # Paste the resized image
        new_img.paste(img_resized, (x, y))
        
        # Save the image
        new_img.save(output_path, 'JPEG', quality=95)

def prepare_dataset(input_dir, output_dir, caption="leonardo dicaprio"):
    """Prepare dataset for LoRA training"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Get all jpg files
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    jpg_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(jpg_files)} images in {input_dir}")
    
    # Prepare metadata
    metadata = []
    
    # Process each image
    for i, filename in enumerate(tqdm(jpg_files, desc="Processing images")):
        input_path = os.path.join(input_dir, filename)
        
        # Create new filename
        new_filename = f"{i+1:03d}.jpg"
        output_path = os.path.join(images_dir, new_filename)
        
        # Resize and save image
        resize_and_save_image(input_path, output_path)
        
        # Add to metadata
        metadata.append({
            "file_name": new_filename,
            "text": caption
        })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset prepared successfully!")
    print(f"Images saved to: {images_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total images processed: {len(jpg_files)}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Leonardo DiCaprio dataset for LoRA training")
    parser.add_argument("--input_dir", default="Leonardo DiCaprio", help="Input directory containing images")
    parser.add_argument("--output_dir", default="dataset", help="Output directory for processed dataset")
    parser.add_argument("--caption", default="leonardo dicaprio", help="Caption for all images")
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(args.input_dir, args.output_dir, args.caption) 