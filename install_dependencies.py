import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("Installing dependencies for Leonardo DiCaprio LoRA training...")
    print("=" * 60)
    
    # List of required packages
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "datasets>=2.12.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.0",
        "xformers>=0.0.20"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 60)
    print(f"Installation complete: {success_count}/{total_count} packages installed successfully")
    
    if success_count == total_count:
        print("✓ All dependencies installed successfully!")
        print("You can now proceed with data preparation and training.")
    else:
        print("⚠ Some packages failed to install. Please check the errors above.")
        print("You may need to install them manually or check your Python environment.")

if __name__ == "__main__":
    main() 