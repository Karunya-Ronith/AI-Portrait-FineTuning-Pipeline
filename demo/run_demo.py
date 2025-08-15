#!/usr/bin/env python3
"""
Launcher script for Leonardo DiCaprio AI Portrait Generator Demo
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'diffusers', 'transformers', 
        'accelerate', 'peft', 'PIL', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  GPU not available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not available")
        return False

def main():
    """Main launcher function"""
    print("🎭 Leonardo DiCaprio AI Portrait Generator - Demo Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check GPU
    check_gpu()
    
    print("\n🚀 Starting Streamlit web interface...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("\n" + "=" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for using Leonardo DiCaprio AI!")
    except Exception as e:
        print(f"\n❌ Error starting demo: {e}")

if __name__ == "__main__":
    main() 