import cv2
import numpy as np
from PIL import Image
import pytesseract

class MRZProcessor:
    def __init__(self):
        self.mrz_patterns = {
            'passport': {
                'line_count': 2,
                'line_length': 44,
                'valid_chars': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
            }
        }
    
    def extract_mrz(self, image_path):
        """Extract and process MRZ from passport image"""
        # Read image
        image = cv2.imread(image_path)
        
        # Get MRZ region
        mrz_region = self._locate_mrz(image)
        if mrz_region is None:
            return None
            
        # Enhance MRZ
        enhanced_mrz = self._enhance_mrz(mrz_region)
        
        # Extract text
        mrz_text = self._extract_text(enhanced_mrz)
        
        # Parse MRZ data
        return self._parse_mrz(mrz_text)
    
    def _locate_mrz(self, image):
        """Locate MRZ zone in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        
        # Focus on bottom 20% of image where MRZ is typically located
        mrz_region = gray[int(height*0.8):height, :]
        
        return mrz_region
    
    def _enhance_mrz(self, mrz_region):
        """Enhance MRZ region for better text extraction"""
        # Apply adaptive threshold
        enhanced = cv2.adaptiveThreshold(
            mrz_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _extract_text(self, mrz_region):
        """Extract text from MRZ region"""
        # Configure Tesseract for MRZ
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        
        # Convert to PIL Image
        pil_image = Image.fromarray(mrz_region)
        
        # Extract text
        mrz_text = pytesseract.image_to_string(
            pil_image, 
            config=custom_config
        )
        
        return mrz_text.strip()
    
    def _parse_mrz(self, mrz_text):
        """Parse MRZ text into structured data"""
        lines = mrz_text.split('\n')
        if len(lines) < 2:
            return None
            
        try:
            # Parse first line (Document info)
            doc_type = lines[0][0:2]
            country = lines[0][2:5]
            surname = lines[0][5:44].split('<<')[0]
            
            # Parse second line (Personal info)
            passport_num = lines[1][0:9]
            nationality = lines[1][10:13]
            dob = lines[1][13:19]
            gender = lines[1][20]
            expiry = lines[1][21:27]
            
            return {
                'document_type': doc_type,
                'country': country,
                'surname': surname,
                'passport_number': passport_num,
                'nationality': nationality,
                'date_of_birth': dob,
                'gender': gender,
                'expiry_date': expiry,
                'mrz_raw': mrz_text
            }
        except Exception as e:
            print(f"Error parsing MRZ: {str(e)}")
            return None 