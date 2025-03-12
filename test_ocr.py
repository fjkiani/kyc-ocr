import cv2
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_results(image_path, results):
    """
    Visualize the original image and OCR results from available engines
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure with subplots for each OCR engine plus original
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show EasyOCR results
    if results['easyocr']['status'] == 'success':
        plt.subplot(1, 3, 2)
        img_copy = image.copy()
        for detection in results['easyocr']['text']:
            bbox, text, conf = detection
            # Draw bounding box
            pts = np.array(bbox, np.int32)
            cv2.polylines(img_copy, [pts], True, (255, 0, 0), 2)
            # Put text above the box
            plt.text(bbox[0][0], bbox[0][1], f'{text} ({conf:.2f})', 
                    color='red', fontsize=8, backgroundcolor='white')
        plt.imshow(img_copy)
        plt.title(f'EasyOCR (Avg Confidence: {results["easyocr"]["confidence"]:.2f})')
        plt.axis('off')
    
    # Show Tesseract results
    if results['tesseract']['status'] == 'success':
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.title('Tesseract Results')
        # Add detected text as title
        plt.figtext(0.7, 0.3, results['tesseract']['text'][:100] + '...', 
                   wrap=True, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_ocr_engines(test_image_path):
    """
    Test available OCR engines with a sample image and compare results
    """
    results = {
        'easyocr': {},
        'tesseract': {}
    }
    
    print("Testing OCR engines...")
    
    # 1. Test EasyOCR
    try:
        reader = easyocr.Reader(['en'])
        easy_result = reader.readtext(test_image_path)
        results['easyocr'] = {
            'status': 'success',
            'text': easy_result,
            'confidence': sum(r[2] for r in easy_result) / len(easy_result) if easy_result else 0
        }
        print("✅ EasyOCR test successful")
    except Exception as e:
        results['easyocr'] = {'status': 'failed', 'error': str(e)}
        print("❌ EasyOCR test failed:", e)

    # 2. Test Tesseract
    try:
        tess_result = pytesseract.image_to_string(cv2.imread(test_image_path))
        results['tesseract'] = {
            'status': 'success',
            'text': tess_result
        }
        print("✅ Tesseract test successful")
    except Exception as e:
        results['tesseract'] = {'status': 'failed', 'error': str(e)}
        print("❌ Tesseract test failed:", e)

    # Visualize results
    visualize_results(test_image_path, results)
    return results

def test_imports():
    print("Testing imports and initializations...")
    
    # Test Tesseract
    print("\nTesseract version:", pytesseract.get_tesseract_version())
    
    # Test EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized successfully!")

def main():
    # First test imports
    test_imports()
    
    # Then test with actual images
    test_dir = Path("test")
    image_formats = ["*.jpg", "*.jpeg", "*.png"]
    
    # Test all images found
    for format in image_formats:
        for image_path in test_dir.glob(format):
            print(f"\nTesting with image: {image_path}")
            results = test_ocr_engines(str(image_path))
            
            # Print text results
            print("\nDetected Text:")
            print("\nEasyOCR:")
            if results['easyocr']['status'] == 'success':
                for detection in results['easyocr']['text']:
                    print(f"- {detection[1]} (Confidence: {detection[2]:.2f})")
            
            print("\nTesseract:")
            if results['tesseract']['status'] == 'success':
                print(results['tesseract']['text'])
            
            # Ask if user wants to continue to next image
            if input("\nProcess next image? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main() 