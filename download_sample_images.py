#!/usr/bin/env python3
"""
Sample Image Downloader

This script downloads sample passport and driver's license images for testing the application.
The images are downloaded from public sources and saved to the test directory.
"""

import os
import requests
from pathlib import Path
import shutil
import zipfile
import io

# Sample image URLs (public domain or creative commons images)
SAMPLE_IMAGES = [
    {
        "url": "https://www.uscis.gov/sites/default/files/images/article-i9-central/M-274_Sample_US_Passport.jpg",
        "filename": "passport-sample-1.jpg",
        "type": "passport"
    },
    {
        "url": "https://www.uscis.gov/sites/default/files/images/article-i9-central/M-274_Sample_US_Passport_Card_Front.jpg",
        "filename": "passport-card-sample-1.jpg",
        "type": "passport"
    },
    {
        "url": "https://www.dhs.gov/sites/default/files/images/st/sandt_enhanced_drivers_license_630x420.jpg",
        "filename": "drivers-license-sample-1.jpg",
        "type": "drivers_license"
    },
    {
        "url": "https://www.uscis.gov/sites/default/files/images/article-i9-central/M-274_Sample_Enhanced_Drivers_License.jpg",
        "filename": "drivers-license-sample-2.jpg",
        "type": "drivers_license"
    }
]

def download_image(url, filename, test_dir):
    """Download an image from a URL and save it to the test directory"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_path = test_dir / filename
        with open(file_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {str(e)}")
        return False

def create_test_directory():
    """Create the test directory if it doesn't exist"""
    test_dir = Path("test")
    if not test_dir.exists():
        test_dir.mkdir()
        print("✅ Created test directory")
    return test_dir

def download_sample_images():
    """Download all sample images"""
    test_dir = create_test_directory()
    
    print("Downloading sample images...")
    success_count = 0
    
    for image in SAMPLE_IMAGES:
        if download_image(image["url"], image["filename"], test_dir):
            success_count += 1
    
    print(f"\nDownloaded {success_count} of {len(SAMPLE_IMAGES)} sample images")
    print(f"Images saved to: {test_dir.absolute()}")

def main():
    """Main entry point"""
    print("=" * 50)
    print("Sample Image Downloader")
    print("=" * 50)
    
    download_sample_images()
    
    print("\nDone!")

if __name__ == "__main__":
    main() 