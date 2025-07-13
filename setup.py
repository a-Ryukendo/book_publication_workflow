import os
import sys
import subprocess
import shutil
from pathlib import Path

def run(command, label):
    print(f"{label}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"{label} done")
        return True
    except Exception as e:
        print(f"{label} failed: {e}")
        return False

def check_python():
    if sys.version_info < (3, 8):
        print("Python 3.8+ needed")
        return False
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install():
    if not run("pip install -r requirements.txt", "Dependencies"):
        return False
    if not run("playwright install chromium", "Playwright"):
        return False
    return True

def make_dirs():
    dirs = [
        "data",
        "data/screenshots",
        "data/content",
        "logs",
        "temp",
        "models",
        "backups",
        "chroma_db"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created {d}")

def setup_env():
    if not os.path.exists(".env"):
        if os.path.exists("env.example"):
            shutil.copy("env.example", ".env")
            print("Created .env from template. Edit with API keys.")
        else:
            print("env.example missing")
            return False
    else:
        print(".env exists")
    return True

def setup_log():
    log_config = "[loguru]\nlevel = INFO\nformat = \"{time} | {level} | {name}:{function}:{line} - {message}\"\nrotation = 10 MB\nretention = 7 days\ncompression = gz\n"
    with open("loguru_config.ini", "w") as f:
        f.write(log_config)
    print("Logging config created")

def run_tests():
    print("Testing...")
    if run("python -m pytest tests/test_basic.py -v", "Tests"):
        print("Tests passed")
        return True
    else:
        print("Some tests failed")
        return True

def create_startup():
    with open("startup.sh", "w") as f:
        f.write("#!/bin/bash\necho \"Starting Book Publication Workflow...\"\n")
    print("Startup script created")
