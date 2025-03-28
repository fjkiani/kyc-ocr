import cv2
import os
import numpy as np
from ocr import OCR

class DocumentClassifier:
    def __init__(self):
        self.ocr = None
        self.document_types = {
            'passport': ['passport', 'united states of america', 'department of state'],
            'drivers_license': ['driver license', 'dl', 'operator license', 'commercial']
        }
        
    def classify(self, image_path):
        """Classify document type based on key text markers and layout"""
        # Initialize OCR with the image directory
        self.ocr = OCR(os.path.dirname(image_path))
        
        # Get OCR results using the full image path
        results = self._get_ocr_results(str(image_path))  # Ensure path is string
        
        # Check for MRZ pattern (passport indicator)
        has_mrz = self._detect_mrz_pattern(image_path)
        
        # Analyze text content
        doc_type = self._analyze_text_content(results)
        confidence = self._calculate_confidence(results, has_mrz)
        
        return {
            'document_type': doc_type,
            'confidence': confidence,
            'has_mrz': has_mrz,
            'detected_fields': results
        }
    
    def _get_ocr_results(self, image_path):
        """Get OCR results using EasyOCR"""
        # Use existing OCR implementation with full image path
        results = self.ocr.easyocr_model_works(str(image_path), visualization=False)
        # Convert results to our format
        formatted_results = []
        for text_obj in results:
            formatted_results.append({
                'text': text_obj[1],  # The text
                'confidence': text_obj[2],  # The confidence score
                'bbox': text_obj[0]  # The bounding box
            })
        return formatted_results
    
    def _detect_mrz_pattern(self, image_path):
        """Detect MRZ pattern in passport"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # MRZ is typically at the bottom of the passport
        height = gray.shape[0]
        mrz_region = gray[int(height*0.8):height, :]
        
        # Apply threshold to isolate MRZ
        _, thresh = cv2.threshold(mrz_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Look for consistent line patterns
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
                                          np.ones((1, 40), np.uint8))
        # Convert numpy.bool_ to Python bool
        return bool(np.sum(horizontal_lines) > 0)
    
    def _analyze_text_content(self, results):
        """Analyze OCR results to determine document type"""
        text_lower = ' '.join([result['text'].lower() for result in results])
        
        # Check for document type indicators
        for doc_type, indicators in self.document_types.items():
            if any(indicator in text_lower for indicator in indicators):
                return doc_type
        
        return 'unknown'
    
    def _calculate_confidence(self, results, has_mrz):
        """Calculate confidence score for classification"""
        confidence = 0.0
        
        if has_mrz:
            confidence += 0.5  # Strong indicator for passport
            
        # Add confidence based on detected keywords
        for result in results:
            text = result['text'].lower()
            if any(indicator in text for indicators 
                  in self.document_types.values() for indicator in indicators):
                confidence += result['confidence']
                
        return min(confidence, 1.0) 