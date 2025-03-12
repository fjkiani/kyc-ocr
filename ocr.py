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
        """Initialize OCR with a base folder for images"""
        self.image_folder = image_folder
        self.text_reader = self._easyocr_model_load()

    def _easyocr_model_load(self):
        """Initialize EasyOCR with English language"""
        return easyocr.Reader(['en'])

    def process_image(self, image_path, engine='easyocr', visualization=True):
        """Process image with selected OCR engine and show visualization"""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if engine == 'easyocr':
            return self._process_easyocr(img, visualization)
        elif engine == 'tesseract':
            return self._process_tesseract(img, visualization)
        elif engine == 'keras':
            return self._process_keras(img, visualization)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

    def _process_easyocr(self, img, visualization=True):
        """Process with EasyOCR and show visualization"""
        results = self.text_reader.readtext(img)
        print(f"\nProcessing with EasyOCR:")
        print(f"Found {len(results)} text regions:")
        
        # Sort results by confidence score
        results.sort(key=lambda x: x[2], reverse=True)
        
        for (bbox, text, prob) in results:
            confidence_indicator = "HIGH" if prob > 0.9 else "MEDIUM" if prob > 0.7 else "LOW"
            print(f"- Text: {text:<30} (Confidence: {prob:.2f} - {confidence_indicator})")
        
        if visualization:
            plt.figure(figsize=(15, 5))
            
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Annotated Image
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            # Add bounding boxes and text
            for (bbox, text, prob) in results:
                # Draw bounding box
                (tl, tr, br, bl) = bbox
                tl = tuple(map(int, tl))
                br = tuple(map(int, br))
                cv2.rectangle(img, tl, br, (0, 255, 0), 2)
                # Add text and confidence
                plt.text(tl[0], tl[1], f'{text} ({prob:.2f})',
                        bbox=dict(facecolor='white', alpha=0.7))
            plt.title("EasyOCR Results")
            plt.axis('off')
            plt.show()
            plt.close()
        
        return results

    def _process_tesseract(self, img, visualization=True):
        """Process with Tesseract and show visualization"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Process with Tesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Get bounding boxes
        boxes = pytesseract.image_to_boxes(thresh, config=custom_config)
        
        print("\nProcessing with Tesseract:")
        print("Detected Text:")
        print(text.strip())
        
        if visualization:
            plt.figure(figsize=(15, 5))
            
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Processed Image with Boxes
            plt.subplot(1, 2, 2)
            plt.imshow(thresh, cmap='gray')
            
            # Draw boxes
            h = thresh.shape[0]
            for b in boxes.splitlines():
                b = b.split()
                if len(b) == 6:
                    char, x1, y1, x2, y2, _ = b
                    # Convert coordinates
                    x1, y1, x2, y2 = int(x1), h-int(y1), int(x2), h-int(y2)
                    # Draw rectangle
                    cv2.rectangle(thresh, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
            plt.title("Tesseract Results")
            plt.axis('off')
            plt.show()
            plt.close()
        
        return [(text.strip(), 1.0)]  # Return with confidence 1.0

    def _process_keras(self, img, visualization=True):
        """Process with Keras-OCR and show visualization"""
        try:
            import keras_ocr
            pipeline = keras_ocr.pipeline.Pipeline()
            
            # Process image
            predictions = pipeline.recognize([img])[0]
            
            print("\nProcessing with Keras-OCR:")
            print(f"Found {len(predictions)} text regions:")
            
            for text, box in predictions:
                print(f"- Text: {text}")
            
            if visualization:
                plt.figure(figsize=(15, 5))
                
                # Original Image
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis('off')
                
                # Annotated Image
                plt.subplot(1, 2, 2)
                plt.imshow(img)
                
                # Draw boxes and text
                for text, box in predictions:
                    # Draw polygon
                    points = np.array(box)
                    for i in range(4):
                        pt1 = tuple(points[i])
                        pt2 = tuple(points[(i + 1) % 4])
                        cv2.line(img, 
                                (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])),
                                (0, 255, 0), 2)
                    # Add text
                    plt.text(points[0][0], points[0][1], text,
                            bbox=dict(facecolor='white', alpha=0.7))
                
                plt.title("Keras-OCR Results")
                plt.axis('off')
                plt.show()
                plt.close()
            
            return [(text, 1.0) for text, _ in predictions]  # Return with confidence 1.0
            
        except ImportError:
            print("Keras-OCR not installed. Please install with: pip install keras-ocr")
            return []

    def easyocr_model_works(self, image_path, visualization=True):
        """Primary OCR processing using EasyOCR"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._process_easyocr(img, visualization)

    def pytesseract_model_works(self, image_path, visualization=True):
        """Specialized processing for MRZ zones using Pytesseract"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._process_tesseract(img, visualization)

    def keras_ocr_works(self, image_path, visualization=True):
        """Process image using Keras-OCR"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._process_keras(img, visualization)