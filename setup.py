#!/usr/bin/env python3
"""
Setup script for the Automated Book Publication Workflow
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install Playwright browsers
    if not run_command("playwright install chromium", "Installing Playwright browsers"):
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/screenshots",
        "data/content",
        "logs",
        "temp",
        "models",
        "backups",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_environment():
    """Set up environment file"""
    env_example = "env.example"
    env_file = ".env"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys and configuration")
        else:
            print("❌ env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True


def setup_logging():
    """Set up logging configuration"""
    log_config = """
# Loguru configuration
[loguru]
level = INFO
format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
rotation = 10 MB
retention = 7 days
compression = gz
"""
    
    with open("loguru_config.ini", "w") as f:
        f.write(log_config)
    
    print("✅ Created logging configuration")


def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    if run_command("python -m pytest tests/test_basic.py -v", "Running tests"):
        print("✅ All tests passed")
        return True
    else:
        print("⚠️  Some tests failed, but setup can continue")
        return True


def create_startup_script():
    """Create startup script"""
    startup_script = """#!/bin/bash
# Startup script for Book Publication Workflow

echo "🚀 Starting Automated Book Publication Workflow..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the API server
python src/main.py --mode api

echo "✅ Server started successfully!"
echo "📖 API documentation: http://localhost:8000/docs"
echo "🔧 Press Ctrl+C to stop the server"
"""

    with open("start.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable on Unix systems
    os.chmod("start.sh", 0o755)
    print("✅ Created startup script: start.sh")


def create_windows_batch():
    """Create Windows batch file"""
    batch_script = """@echo off
echo 🚀 Starting Automated Book Publication Workflow...

REM Activate virtual environment if it exists
if exist ".venv\\Scripts\\activate.bat" (
    call .venv\\Scripts\\activate.bat
)

REM Start the API server
python src\\main.py --mode api

echo ✅ Server started successfully!
echo 📖 API documentation: http://localhost:8000/docs
echo 🔧 Press Ctrl+C to stop the server
pause
"""

    with open("start.bat", "w") as f:
        f.write(batch_script)
    
    print("✅ Created Windows batch file: start.bat")


def main():
    """Main setup function"""
    print("🚀 Automated Book Publication Workflow Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Set up environment
    print("\n⚙️  Setting up environment...")
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Set up logging
    print("\n📝 Setting up logging...")
    setup_logging()
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()
    
    # Create startup scripts
    print("\n📜 Creating startup scripts...")
    create_startup_script()
    create_windows_batch()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run the system:")
    print("   - Linux/Mac: ./start.sh")
    print("   - Windows: start.bat")
    print("   - Or: python src/main.py --mode api")
    print("3. Access API documentation: http://localhost:8000/docs")
    print("\n🔧 Configuration options:")
    print("- Edit config/settings.py for advanced configuration")
    print("- Modify .env file for environment-specific settings")
    print("\n📚 Documentation:")
    print("- README.md for detailed usage instructions")
    print("- API docs at http://localhost:8000/docs when running")


if __name__ == "__main__":
    main() 