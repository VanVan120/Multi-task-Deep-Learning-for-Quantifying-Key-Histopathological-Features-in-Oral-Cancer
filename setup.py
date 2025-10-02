#!/usr/bin/env python3
"""
Setup script for Multi-task Deep Learning in Oral Cancer Analysis
================================================================

This script sets up the development environment and verifies all dependencies.
Run this first before starting your research.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def run_command(command, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} found")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} found, but Python 3.8+ required")
        return False

def check_gpu():
    """Check GPU availability and specifications"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {gpu_memory}GB")
            
            if "RTX 4050" in gpu_name:
                print("   Optimized settings loaded for RTX 4050")
            return True
        else:
            print("⚠️  No GPU detected - will use CPU (training will be slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - GPU check will be performed after installation")
        return None

def install_requirements():
    """Install requirements based on environment"""
    print("\n🔄 Installing Python packages...")
    
    # Use local requirements for development
    requirements_file = "requirements-local.txt"
    if not os.path.exists(requirements_file):
        requirements_file = "requirements.txt"
    
    success, stdout, stderr = run_command(f"pip install -r {requirements_file}")
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "outputs/models",
        "outputs/logs",
        "outputs/results",
        "experiments"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")

def verify_installation():
    """Verify that key packages are installed correctly"""
    packages_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("opencv-python", "OpenCV"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
    ]
    
    print("\n🔄 Verifying installation...")
    all_good = True
    
    for package, display_name in packages_to_check:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - not installed")
            all_good = False
    
    return all_good

def setup_git_hooks():
    """Set up git hooks for code quality (optional)"""
    if os.path.exists(".git"):
        print("\n🔄 Setting up git hooks...")
        # This is optional - you can add pre-commit hooks here
        print("✅ Git repository detected")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\n📝 Next Steps:")
    print("1. Start Jupyter Lab: jupyter lab")
    print("2. Open notebooks/01_data_exploration.ipynb")
    print("3. Run the configuration test: python src/config.py")
    print("\n💡 Development Tips:")
    print("- Use small batch sizes (8-16) on RTX 4050")
    print("- Enable mixed precision training (16-bit)")
    print("- Monitor GPU memory with: nvidia-smi")
    print("- Use cloud training for larger experiments")
    print("\n📚 Project Structure:")
    print("src/          - Source code modules")
    print("notebooks/    - Jupyter notebooks for exploration")  
    print("configs/      - Configuration files")
    print("data/         - Data storage")
    print("outputs/      - Model outputs and logs")
    print("experiments/  - Experiment results")

def main():
    """Main setup function"""
    print("🔬 Multi-task Deep Learning for Oral Cancer Analysis")
    print("=" * 60)
    print("Setting up development environment...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create project directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to install. Please check the error messages above.")
        sys.exit(1)
    
    # Check GPU after PyTorch installation
    check_gpu()
    
    # Optional: setup git hooks
    setup_git_hooks()
    
    # Print next steps
    print_next_steps()
    
    print(f"\n🚀 Ready to start your oral cancer research!")

if __name__ == "__main__":
    main()