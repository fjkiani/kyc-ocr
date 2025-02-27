from ocr import OCR
import os
import matplotlib.pyplot as plt

def test_all_methods():
    image_dir = os.path.dirname("/Users/fahadkiani/Desktop/development/Comparison-of-OCR-main/test/passport-2.jpg")
    ocr = OCR(image_folder=image_dir)
    
    total_images = len(ocr.images)
    print(f"\nProcessing {total_images} images...")
    
    print("\n=== EasyOCR Results ===")
    try:
        print("Close each image window to continue to the next image/method...")
        ocr.easyocr_model_works(visualization=True)
        plt.close('all')  # Close any remaining plot windows
    except Exception as e:
        print(f"EasyOCR Error: {str(e)}")
    
    input("\nPress Enter to continue to Keras OCR...")
    
    print("\n=== Keras OCR Results ===")
    try:
        ocr.keras_ocr_works(visualization=True)
        plt.close('all')
    except Exception as e:
        print(f"Keras OCR Error: {str(e)}")
    
    input("\nPress Enter to continue to Pytesseract...")
    
    print("\n=== Pytesseract Results ===")
    try:
        ocr.pytesseract_model_works(visualization=True)
        plt.close('all')
    except Exception as e:
        print(f"Pytesseract Error: {str(e)}")

if __name__ == "__main__":
    test_all_methods() 