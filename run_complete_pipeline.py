import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("⚠ Dataset not found. Running data preparation...")
        return run_command("python data_preparation.py", "Step 1: Data Preparation")
    
    print("✓ Dataset found")
    return True

def main():
    print("🎭 Leonardo DiCaprio LoRA Training - Complete Pipeline")
    print("=" * 60)
    
    print("\nThis script will guide you through the complete pipeline:")
    print("1. Data preparation (if needed)")
    print("2. LoRA training")
    print("3. Image generation demo")
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues above.")
        return
    
    # Step 2: Training
    print("\n" + "="*60)
    print("🎯 Step 2: LoRA Training")
    print("="*60)
    print("This will take ~45 minutes on your RTX 3050.")
    print("Training parameters:")
    print("- LoRA Rank: 16")
    print("- Learning Rate: 1e-4")
    print("- Training Steps: 100")
    print("- Batch Size: 1")
    print("- Mixed Precision: FP16")
    
    response = input("\nStart training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training skipped. You can run it manually with: python train_lora_final.py")
        return
    
    if not run_command("python train_lora_final.py", "Step 2: LoRA Training"):
        print("❌ Training failed. Please check the error messages above.")
        return
    
    # Step 3: Generation Demo
    print("\n" + "="*60)
    print("🎨 Step 3: Image Generation Demo")
    print("="*60)
    print("Training completed! Now let's generate some images.")
    
    # Check if model exists
    if not os.path.exists("models/pipeline"):
        print("❌ Trained model not found. Please check if training completed successfully.")
        return
    
    print("Generating sample images...")
    
    # Generate a few example images
    examples = [
        "leonardo dicaprio eating",
        "leonardo dicaprio smiling",
        "leonardo dicaprio in a suit"
    ]
    
    for prompt in examples:
        command = f'python generate_images.py --prompt "{prompt}" --num_images 1'
        if not run_command(command, f"Generating: {prompt}"):
            print(f"⚠ Failed to generate: {prompt}")
    
    print("\n🎉 Pipeline completed!")
    print("\n📁 Your generated images are in the 'outputs/' directory")
    print("\n💡 Try more prompts:")
    print("   python generate_images.py --prompt 'leonardo dicaprio portrait'")
    print("   python generate_images.py --prompt 'leonardo dicaprio on a red carpet'")
    print("   python generate_images.py --prompt 'leonardo dicaprio with sunglasses'")

if __name__ == "__main__":
    main() 