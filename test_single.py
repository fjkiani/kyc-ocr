from ocr import OCR
import os
import matplotlib.pyplot as plt

def test_all_methods():
    image_dir = os.path.dirname("/Users/fahadkiani/Desktop/development/Comparison-of-OCR-main/test/passport-2.jpg")
    ocr = OCR(image_folder=image_dir)
    
    total_images = len(ocr.images)
    print(f"\nProcessing {total_images} images...")
    
    print("\n=== Primary OCR (EasyOCR) Results ===")
    try:
        print("Processing with EasyOCR for main text extraction...")
        ocr.easyocr_model_works(visualization=True)
        plt.close('all')  # Close any remaining plot windows
    except Exception as e:
        print(f"EasyOCR Error: {str(e)}")
    
    input("\nPress Enter to continue to MRZ Processing (Pytesseract)...")
    
    print("\n=== MRZ Processing (Pytesseract) Results ===")
    try:
        print("Processing MRZ zone with specialized settings...")
        ocr.pytesseract_model_works(visualization=True)
        plt.close('all')
    except Exception as e:
        print(f"Pytesseract Error: {str(e)}")
    
    print("\nProcessing complete! Check the results above for extracted text and confidence scores.")

if __name__ == "__main__":
    test_all_methods() 