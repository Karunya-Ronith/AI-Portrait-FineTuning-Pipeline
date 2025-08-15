import torch
import os
import sys

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    try:
        import diffusers
        print(f"✓ diffusers {diffusers.__version__}")
    except ImportError:
        print("✗ diffusers not found")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        print("✗ transformers not found")
        return False
    
    try:
        import accelerate
        print(f"✓ accelerate {accelerate.__version__}")
    except ImportError:
        print("✗ accelerate not found")
        return False
    
    try:
        import datasets
        print(f"✓ datasets {datasets.__version__}")
    except ImportError:
        print("✗ datasets not found")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError:
        print("✗ Pillow not found")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm not found")
        return False
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("✗ CUDA not available")
        return False

def test_dataset():
    """Test if dataset is properly prepared"""
    print("\nTesting dataset...")
    
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory '{dataset_dir}' not found")
        return False
    
    images_dir = os.path.join(dataset_dir, "images")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    
    if not os.path.exists(images_dir):
        print(f"✗ Images directory '{images_dir}' not found")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"✗ Metadata file '{metadata_path}' not found")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"✓ Found {len(image_files)} images in dataset")
    
    # Check metadata
    with open(metadata_path, 'r') as f:
        metadata_lines = f.readlines()
    print(f"✓ Found {len(metadata_lines)} metadata entries")
    
    return True

def main():
    print("=" * 50)
    print("Leonardo DiCaprio LoRA Training - System Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test GPU
    gpu_ok = test_gpu()
    
    # Test dataset
    dataset_ok = test_dataset()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    if deps_ok and gpu_ok and dataset_ok:
        print("✓ All tests passed! You're ready to train.")
        print("\nNext steps:")
        print("1. Run: python train_lora_simple.py")
        print("2. Wait ~45 minutes for training to complete")
        print("3. Generate images with the trained model")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        
        if not deps_ok:
            print("\nTo install dependencies, run:")
            print("python install_dependencies.py")
        
        if not gpu_ok:
            print("\nGPU is required for training. Make sure CUDA is installed.")
        
        if not dataset_ok:
            print("\nTo prepare dataset, run:")
            print("python data_preparation.py")

if __name__ == "__main__":
    main() 