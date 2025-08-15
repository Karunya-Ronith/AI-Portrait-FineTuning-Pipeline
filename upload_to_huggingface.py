#!/usr/bin/env python3
"""
Upload Leonardo DiCaprio LoRA model to Hugging Face
"""

import os
from huggingface_hub import HfApi, login

def upload_model():
    """Upload the model to Hugging Face"""
    print("🚀 Uploading Leonardo DiCaprio LoRA Model to Hugging Face")
    print("=" * 60)
    
    # Get the current directory
    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, "leonardo_dicaprio_lora")
    
    if not os.path.exists(model_folder):
        print(f"❌ Model folder not found: {model_folder}")
        print("Please make sure you're in the Project directory and the model is prepared.")
        return
    
    print(f"📁 Model folder: {model_folder}")
    
    # List files to be uploaded
    print("\n📋 Files to be uploaded:")
    for root, dirs, files in os.walk(model_folder):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, model_folder)
            print(f"   - {relative_path}")
    
    # Login to Hugging Face (you'll need to enter your token)
    print("\n🔐 Logging in to Hugging Face...")
    try:
        login()
        print("✓ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("\n💡 To get your token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' permissions")
        print("3. Copy the token and run this script again")
        return
    
    # Upload the model
    print("\n📤 Uploading model files...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=model_folder,
            repo_id="KarunyaRonith29/leonardo_dicaprio_lora",
            repo_type="model",
        )
        print("✓ Upload successful!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return
    
    print("\n🎉 Model uploaded successfully!")
    print(f"🌐 Your model is now available at:")
    print(f"   https://huggingface.co/KarunyaRonith29/leonardo_dicaprio_lora")
    
    print("\n📝 Next steps:")
    print("1. Visit your model page")
    print("2. Edit the README.md to replace 'YOUR_USERNAME' with 'KarunyaRonith29'")
    print("3. Test the model to ensure it works")
    print("4. Share with the community!")

if __name__ == "__main__":
    upload_model() 