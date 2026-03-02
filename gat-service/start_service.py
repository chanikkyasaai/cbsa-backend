#!/usr/bin/env python3
"""
GAT Service Startup Script
Installs dependencies and starts the service
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, check=True, shell=True):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            raise
        return e

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling dependencies...")
    
    # Try to install with pip
    try:
        # Upgrade pip first
        run_command(f"{sys.executable} -m pip install --upgrade pip")
        
        # Install requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            run_command(f"{sys.executable} -m pip install -r {requirements_file}")
        else:
            print("requirements.txt not found, installing minimal dependencies...")
            
            # Install minimal required packages
            packages = [
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.24.0",
                "pydantic>=2.5.0",
                "numpy>=1.24.0"
            ]
            
            for package in packages:
                run_command(f"{sys.executable} -m pip install {package}")
        
        print("Dependencies installed successfully!")
        
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False
    
    return True

def check_pytorch():
    """Check if PyTorch is available"""
    try:
        import torch
        import torch_geometric
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch Geometric available: True")
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        return True
    except ImportError:
        print("PyTorch not available - service will run in simulation mode")
        print("To install PyTorch: pip install torch torch-geometric")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    service_dir = Path(__file__).parent
    
    # Create logs directory
    logs_dir = service_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create models directory
    models_dir = service_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("Directories created successfully!")

def check_service_health():
    """Check if service is running properly"""
    import time
    import requests
    
    print("\nWaiting for service to start...")
    time.sleep(3)
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is running and healthy!")
            health_data = response.json()
            print(f"Model status: {health_data.get('model_status', 'unknown')}")
            return True
        else:
            print(f"❌ Service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to service: {e}")
        return False

def start_service():
    """Start the GAT service"""
    print("\n" + "="*50)
    print("Starting GAT Behavioral Authentication Service")
    print("="*50)
    
    service_dir = Path(__file__).parent
    main_file = service_dir / "main.py"
    
    if not main_file.exists():
        print(f"ERROR: {main_file} not found!")
        sys.exit(1)
    
    print(f"Starting service from: {main_file}")
    print("Service will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop the service")
    print("-" * 50)
    
    try:
        # Change to service directory
        os.chdir(service_dir)
        
        # Start the service
        run_command(
            f"{sys.executable} -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload",
            check=False
        )
        
    except KeyboardInterrupt:
        print("\nShutting down service...")
    except Exception as e:
        print(f"Error starting service: {e}")

def main():
    """Main startup function"""
    print("GAT Service Startup Script")
    print("="*50)
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Check PyTorch availability
    pytorch_available = check_pytorch()
    
    if not pytorch_available:
        print("\nNote: Running in simulation mode without PyTorch")
        print("For full functionality, install PyTorch:")
        print("pip install torch torch-geometric")
    
    # Start the service
    start_service()

if __name__ == "__main__":
    main()