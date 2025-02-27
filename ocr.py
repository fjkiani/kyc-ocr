import cv2
import os
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt

if os.name == 'nt':  # for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCR():
    
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = self.load_images_from_folder()
        self.text_reader = self._easyocr_model_load()

    def load_images_from_folder(self):
        images = []
        if not os.path.exists(self.image_folder):
            raise ValueError(f"Image folder '{self.image_folder}' does not exist")
        
        files = os.listdir(self.image_folder)
        if not files:
            raise ValueError(f"No files found in '{self.image_folder}'")
        
        for filename in files:
            file_path = os.path.join(self.image_folder, filename)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue
            
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(gray)
            except cv2.error as e:
                print(f"Error processing image {filename}: {str(e)}")
                continue
            
        if not images:
            raise ValueError("No valid images could be loaded from the folder")
        
        return images

    def _easyocr_model_load(self):
        # Initialize with English and an additional language if needed
        text_reader = easyocr.Reader(['en'])
        return text_reader

    def easyocr_model_works(self, visualization=True):
        """Primary OCR processing using EasyOCR"""
        for i in range(len(self.images)):
            print(f"\nProcessing Image {i+1}/{len(self.images)}:")
            results = self.text_reader.readtext(self.images[i])
            print(f"Found {len(results)} text regions:")
            
            # Sort results by confidence score
            results.sort(key=lambda x: x[2], reverse=True)
            
            for (bbox, text, prob) in results:
                confidence_indicator = "HIGH" if prob > 0.9 else "MEDIUM" if prob > 0.7 else "LOW"
                print(f"- Text: {text:<30} (Confidence: {prob:.2f} - {confidence_indicator})")
            
            if visualization:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.images[i])
                # Add bounding boxes and text
                for (bbox, text, prob) in results:
                    # Draw bounding box
                    (tl, tr, br, bl) = bbox
                    tl = tuple(map(int, tl))
                    br = tuple(map(int, br))
                    cv2.rectangle(self.images[i], tl, br, (0, 255, 0), 2)
                    # Add text and confidence
                    plt.text(tl[0], tl[1], f'{text} ({prob:.2f})',
                            bbox=dict(facecolor='white', alpha=0.7))
                
                plt.title(f"Image {i+1}/{len(self.images)} - OCR Results")
                plt.axis('off')
                plt.show()
                plt.close()

    def pytesseract_model_works(self, visualization=True):
        """Specialized processing for MRZ zones using Pytesseract"""
        for i, img in enumerate(self.images):
            print(f"\nProcessing Image {i+1}/{len(self.images)}:")
            
            # Convert to grayscale for better MRZ detection
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold to isolate MRZ region
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Configure Tesseract for MRZ
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            
            # Process MRZ region
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            print("MRZ Data Detected:")
            print(text.strip())
            
            if visualization:
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(thresh, cmap='gray')
                plt.title("Processed MRZ Region")
                plt.axis('off')
                
                plt.show()
                plt.close()