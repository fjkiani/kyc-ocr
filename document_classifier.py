import cv2
import os
import numpy as np
from ocr import OCR
import re

class DocumentClassifier:
    def __init__(self):
        self.ocr = None
        self.document_types = {
            'passport': ['passport', 'united states of america', 'department of state'],
            'drivers_license': ['driver license', 'dl', 'operator license', 'commercial'],
            'resume': ['resume', 'cv', 'curriculum vitae', 'work experience', 'education', 'skills', 
                      'certifications', 'employment history', 'professional experience', 
                      'technical skills', 'contact info', 'references']
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
        doc_type, type_confidence = self._analyze_text_content(results)
        confidence = self._calculate_confidence(results, has_mrz, type_confidence)
        
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
        
        # Count matches for each document type
        type_matches = {doc_type: 0 for doc_type in self.document_types}
        type_confidence = {doc_type: 0.0 for doc_type in self.document_types}
        
        # Check for document type indicators
        for doc_type, indicators in self.document_types.items():
            matches = 0
            total_indicators = len(indicators)
            
            for indicator in indicators:
                if indicator in text_lower:
                    matches += 1
            
            # Calculate match percentage
            if total_indicators > 0:
                type_matches[doc_type] = matches
                type_confidence[doc_type] = matches / total_indicators
        
        # Find the document type with the most matches
        best_type = max(type_matches, key=type_matches.get)
        best_confidence = type_confidence[best_type]
        
        # If no good matches, check for resume-specific patterns
        if best_confidence < 0.3:
            # Check for resume-specific patterns like education/experience sections
            resume_patterns = [
                r'education|experience|skills|certifications',
                r'\d{4}\s*-\s*\d{4}|\d{2}/\d{4}',  # Date ranges like 2019-2021 or 05/2021
                r'university|college|school',
                r'bachelor|master|phd|bs|ba|ms|mba',
                r'technical skills|programming|languages',
                r'email|phone|contact'
            ]
            
            resume_matches = 0
            for pattern in resume_patterns:
                if re.search(pattern, text_lower):
                    resume_matches += 1
            
            if resume_matches >= 3:  # If at least 3 resume patterns match
                best_type = 'resume'
                best_confidence = 0.7  # Higher confidence based on pattern matching
        
        return best_type, best_confidence
    
    def _calculate_confidence(self, results, has_mrz, type_confidence):
        """Calculate confidence score for classification"""
        confidence = type_confidence
        
        if has_mrz:
            confidence = max(confidence, 0.8)  # Strong indicator for passport
            
        # Add confidence based on detected keywords
        keyword_confidence = 0.0
        for result in results:
            text = result['text'].lower()
            if any(indicator in text for indicators 
                  in self.document_types.values() for indicator in indicators):
                keyword_confidence += result['confidence'] * 0.1  # Scale down the impact
                
        confidence = min(confidence + keyword_confidence, 1.0)
        
        return confidence 