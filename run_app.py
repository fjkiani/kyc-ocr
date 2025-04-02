#!/usr/bin/env python3
"""
Document Processing System Launcher

This script checks dependencies and launches the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Run dependency checker if available"""
    checker_path = Path("check_dependencies.py")
    if checker_path.exists():
        print("Running dependency checks...")
        try:
            result = subprocess.run([sys.executable, str(checker_path)], 
                                   capture_output=True, text=True)
            if result.returncode != 0:
                print("Dependency check failed. Please fix the issues before continuing.")
                print(result.stdout)
                return False
        except Exception as e:
            print(f"Error running dependency checker: {e}")
            print("Continuing without dependency check...")
    return True

def create_temp_directory():
    """Create temp directory if it doesn't exist"""
    temp_dir = Path("temp")
    if not temp_dir.exists():
        try:
            temp_dir.mkdir()
            print("Created temp directory for uploaded files")
        except Exception as e:
            print(f"Warning: Could not create temp directory: {e}")

def check_sample_images():
    """Check if sample images exist and download them if not"""
    test_dir = Path("test")
    if not test_dir.exists() or not list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png")):
        print("No sample images found. Downloading sample images...")
        downloader_path = Path("download_sample_images.py")
        if downloader_path.exists():
            try:
                subprocess.run([sys.executable, str(downloader_path)], check=True)
                print("Sample images downloaded successfully.")
            except Exception as e:
                print(f"Error downloading sample images: {e}")
                print("You can still use the app, but sample images will not be available.")
        else:
            print("Sample image downloader script not found.")
            print("You can still use the app, but sample images will not be available.")

def run_streamlit():
    """Launch the Streamlit application"""
    app_path = Path("app.py")
    if not app_path.exists():
        print(f"Error: {app_path} not found")
        return False
    
    print("\n" + "=" * 50)
    print("Launching Document Processing System")
    print("=" * 50 + "\n")
    
    try:
        # Run streamlit with the app
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        return True
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        return False

def main():
    """Main entry point"""
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Please install it with:")
        print("pip install streamlit")
        return
    
    # Check dependencies
    if not check_dependencies():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create temp directory
    create_temp_directory()
    
    # Check for sample images
    check_sample_images()
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main() 