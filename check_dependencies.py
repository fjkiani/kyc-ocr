#!/usr/bin/env python3
"""
Dependency Checker for Document Processing System

This script checks if all required dependencies are installed and properly configured.
It helps diagnose common setup issues before running the main application.
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {major}.{minor}")
        return False
    else:
        print(f"✅ Python version {major}.{minor} is compatible")
        return True

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets minimum version"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        else:
            version = "Unknown"
        
        if min_version and version != "Unknown":
            from packaging import version as version_parser
            if version_parser.parse(version) < version_parser.parse(min_version):
                print(f"❌ {package_name} version {version} is installed, but {min_version} or higher is required")
                return False
        
        print(f"✅ {package_name} is installed (version: {version})")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed and accessible"""
    print("Checking Tesseract OCR...")
    try:
        import pytesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract is installed (version: {version})")
            return True
        except Exception as e:
            print(f"❌ Tesseract is not properly configured: {str(e)}")
            if os.name == 'nt':  # Windows
                print("   Make sure Tesseract is installed and added to PATH")
                print("   Or set pytesseract.pytesseract.tesseract_cmd to the executable path")
            else:  # Unix/Linux/Mac
                print("   Make sure Tesseract is installed (e.g., apt-get install tesseract-ocr)")
            return False
    except ImportError:
        print("❌ pytesseract is not installed")
        return False

def check_api_key():
    """Check if Fireworks API key is configured"""
    print("Checking Fireworks API key...")
    
    # Check environment variable
    api_key = os.getenv('FIREWORKS_API_KEY')
    if api_key:
        print("✅ FIREWORKS_API_KEY environment variable is set")
        return True
    
    # Check .env file
    env_path = Path('.') / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('FIREWORKS_API_KEY='):
                    print("✅ FIREWORKS_API_KEY found in .env file")
                    return True
    
    print("❌ FIREWORKS_API_KEY not found in environment variables or .env file")
    print("   Please set the API key before running the application")
    return False

def check_streamlit():
    """Check if Streamlit is installed and working"""
    print("Checking Streamlit...")
    try:
        import streamlit
        print(f"✅ Streamlit is installed (version: {streamlit.__version__})")
        
        # Check if app.py exists
        if Path('app.py').exists():
            print("✅ app.py file found")
        else:
            print("❌ app.py file not found")
            return False
        
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        return False

def check_temp_directory():
    """Check if temp directory exists or can be created"""
    print("Checking temp directory...")
    temp_dir = Path("temp")
    
    if temp_dir.exists():
        if temp_dir.is_dir():
            print("✅ temp directory exists")
            # Check if writable
            try:
                test_file = temp_dir / "test_write.tmp"
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                print("✅ temp directory is writable")
                return True
            except Exception as e:
                print(f"❌ temp directory is not writable: {str(e)}")
                return False
        else:
            print("❌ temp exists but is not a directory")
            return False
    else:
        try:
            temp_dir.mkdir()
            print("✅ temp directory created successfully")
            return True
        except Exception as e:
            print(f"❌ Could not create temp directory: {str(e)}")
            return False

def main():
    """Run all dependency checks"""
    print("=" * 50)
    print("Document Processing System Dependency Checker")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_package("numpy", "1.19.0"),
        check_package("opencv-python", "4.5.0"),
        check_package("matplotlib", "3.3.0"),
        check_package("PIL", "7.0.0") or check_package("Pillow", "7.0.0"),
        check_package("requests", "2.25.0"),
        check_package("easyocr", "1.4.1"),
        check_tesseract(),
        check_streamlit(),
        check_api_key(),
        check_temp_directory()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ All dependency checks passed!")
        print("You can now run the application with: streamlit run app.py")
    else:
        print("❌ Some dependency checks failed")
        print("Please fix the issues above before running the application")
    print("=" * 50)

if __name__ == "__main__":
    main() 