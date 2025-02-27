import cv2
import keras_ocr
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt

def test_imports():
    print("All imports successful!")
    
    # Test Tesseract
    print("\nTesseract version:", pytesseract.get_tesseract_version())
    
    # Test EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized successfully!")
    
    # Test Keras-OCR
    # pipeline = keras_ocr.pipeline.Pipeline()
    # print("Keras-OCR pipeline initialized successfully!")

if __name__ == "__main__":
    test_imports() 