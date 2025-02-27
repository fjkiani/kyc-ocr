import cv2
import os
import keras_ocr
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt

if os.name == 'nt':  # for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCR():
    
    def __init__(self,image_folder):
        
        self.image_folder= image_folder
        self.images = self.load_images_from_folder()
        self.pipeline = self._keras_model_load()
        self.text_reader=self._easyocr_model_load()

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

    def _keras_model_load(self):
        pipeline = keras_ocr.pipeline.Pipeline()
        return pipeline

    def keras_ocr_works(self,visualization=True):
        print("\nProcessing images with Keras OCR...")
        images = [keras_ocr.tools.read(img) for img in self.load_images_from_folder()]
        prediction_groups = self.pipeline.recognize(images)
        
        for i, predictions in enumerate(prediction_groups):
            print(f"\nImage {i+1}/{len(images)}:")
            print(f"Found {len(predictions)} text regions:")
            for text, bbox in predictions:
                print(f"- Text: {text:<30}")
        
            if visualization:
                plt.figure(figsize=(10, 10))
                keras_ocr.tools.drawAnnotations(image=images[i], predictions=predictions, ax=plt.gca())
                plt.title(f"Image {i+1}/{len(images)}")
                plt.show()
                plt.close()

    def _easyocr_model_load(self):
        text_reader = easyocr.Reader(['tr','en']) #Initialzing the ocr
        return text_reader

    def easyocr_model_works(self,visualization=True):
        for i in range(len(self.images)):
            print(f"\nProcessing Image {i+1}/{len(self.images)}:")
            results = self.text_reader.readtext(self.images[i])
            print(f"Found {len(results)} text regions:")
            for (bbox, text, prob) in results:
                print(f"- Text: {text:<30} (Confidence: {prob:.2f})")
            if visualization:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.images[i])
                plt.title(f"Image {i+1}/{len(self.images)}")
                plt.show()
                plt.close()

    def pytesseract_model_works(self,visualization=True):
        for i, img in enumerate(self.images):
            print(f"\nProcessing Image {i+1}/{len(self.images)}:")
            text = pytesseract.image_to_string(img)
            print("Text detected:")
            print(text.strip())
            
            if visualization:
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.title(f"Image {i+1}/{len(self.images)}")
                plt.show()
                plt.close()