"""
Enhanced OCR Testing Module

This script provides comprehensive testing of the complete document processing pipeline.
It demonstrates the full integration of all components:
- DocumentProcessor: Main orchestrator
- DocumentClassifier: For document type detection
- OCR: For text extraction
- LLMProcessor: For field extraction and validation
- MRZProcessor: For specialized passport processing

Unlike test_llm.py (OCR+LLM) and test_llm_image.py (direct image-to-LLM),
this script tests the entire production pipeline with all validation steps.
"""

# Suppress warnings first
import warnings
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from document_processor import DocumentProcessor, NumpyEncoder
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import argparse
import tempfile
import shutil
import mimetypes
import traceback
import sys

# Import PDF handling libraries
try:
    import fitz  # PyMuPDF
    import magic
    from pdf2image import convert_from_path
    import cv2
    PDF_SUPPORT = True
except ImportError:
    print("Warning: PDF support libraries not fully installed. PDF processing will be limited.")
    PDF_SUPPORT = False

# Load environment variables from .env file
load_dotenv()

class EnhancedOCRTester:
    def __init__(self, api_key=None):
        """Initialize the enhanced OCR tester.
        
        Note: This class tests the complete document processing pipeline.
        It uses DocumentProcessor as the main orchestrator, which in turn
        coordinates all other components (OCR, LLM, MRZ processing).
        
        This represents the production-ready approach with all validation
        steps, unlike the simplified testing in test_llm.py and test_llm_image.py.
        """
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        self.processor = DocumentProcessor(self.api_key)
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Clean up temporary files on object destruction"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory: {str(e)}")
    
    def detect_file_type(self, file_path):
        """Detect the file type using python-magic and fallback to mimetypes"""
        try:
            if PDF_SUPPORT:
                # Try using python-magic first (more accurate)
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(file_path)
                print(f"Detected file type using python-magic: {file_type}")
                return file_type
            else:
                # Fallback to mimetypes
                file_type, _ = mimetypes.guess_type(file_path)
                print(f"Detected file type using mimetypes: {file_type}")
                return file_type
        except Exception as e:
            print(f"Error detecting file type: {str(e)}")
            # Last resort: use file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                return 'application/pdf'
            elif ext in ['.jpg', '.jpeg']:
                return 'image/jpeg'
            elif ext == '.png':
                return 'image/png'
            else:
                return None
    
    def convert_pdf_to_image(self, pdf_path, page_num=0):
        """Convert a PDF page to an image for processing"""
        if not PDF_SUPPORT:
            print("PDF support libraries not installed. Cannot convert PDF.")
            return None
            
        print(f"Converting PDF to image: {pdf_path}, page {page_num}")
        
        # Check if file exists and is accessible
        if not os.path.isfile(pdf_path):
            print(f"Error: PDF file not found or not accessible: {pdf_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Absolute path: {os.path.abspath(pdf_path)}")
            return None
            
        output_path = os.path.join(self.temp_dir, f"{os.path.basename(pdf_path).replace(':', '_')}_page_{page_num}.jpg")
        
        try:
            # Try using pdf2image first (better quality usually)
            try:
                print("Attempting conversion with pdf2image...")
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                if images:
                    images[0].save(output_path, 'JPEG')
                    print(f"Successfully converted PDF page to image with pdf2image: {output_path}")
                else:
                    print("No images extracted with pdf2image, trying PyMuPDF...")
                    raise Exception("No images extracted with pdf2image")
            except Exception as pdf2image_error:
                print(f"pdf2image conversion failed: {str(pdf2image_error)}")
                print("Falling back to PyMuPDF...")
                
                # Fallback to PyMuPDF
                doc = fitz.open(pdf_path)
                if page_num < len(doc):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
                    pix.save(output_path)
                    print(f"Successfully converted PDF page to image with PyMuPDF: {output_path}")
                else:
                    raise ValueError(f"Page number {page_num} out of range (document has {len(doc)} pages)")
            
            # Verify the image was created and can be opened
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Converted image file not found: {output_path}")
            
            # Try to open the image to verify it's valid
            img = cv2.imread(output_path)
            if img is None:
                raise ValueError(f"Failed to open converted image: {output_path}")
            
            height, width = img.shape[:2]
            print(f"Successfully verified converted image: {width}x{height} pixels")
            
            return output_path
        except Exception as e:
            print(f"Error converting PDF to image: {str(e)}")
            traceback.print_exc()
            return None
    
    def process_document(self, file_path):
        """Process a document file (image or PDF) through the complete pipeline"""
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        try:
            # Handle special characters in filenames
            safe_file_path = file_path
            if ':' in file_path:
                print(f"Warning: File path contains colon which may cause issues: {file_path}")
                # Create a copy with a safe name if needed
                if os.path.exists(file_path):
                    safe_name = os.path.basename(file_path).replace(':', '_')
                    safe_file_path = os.path.join(self.temp_dir, safe_name)
                    shutil.copy2(file_path, safe_file_path)
                    print(f"Created safe copy at: {safe_file_path}")
            
            # Detect file type
            file_type = self.detect_file_type(safe_file_path)
            
            # Handle different file types
            if file_type and 'pdf' in file_type.lower():
                if not PDF_SUPPORT:
                    raise ValueError("PDF support libraries not installed. Cannot process PDF.")
                    
                print(f"Detected PDF document: {safe_file_path}")
                
                # Get number of pages
                doc = fitz.open(safe_file_path)
                num_pages = len(doc)
                print(f"PDF has {num_pages} pages")
                
                # For testing purposes, process only the first page
                # In a real application, you might want to process all pages or let the user select
                page_num = 0
                
                # Convert PDF to image
                image_path = self.convert_pdf_to_image(safe_file_path, page_num)
                if not image_path:
                    raise ValueError("Failed to convert PDF to image")
                
                print(f"Processing PDF page {page_num} as image: {image_path}")
                return self.process_single_image(image_path, original_file=safe_file_path)
            
            elif file_type and ('image' in file_type.lower() or safe_file_path.lower().endswith(('.jpg', '.jpeg', '.png'))):
                print(f"Detected image document: {safe_file_path}")
                return self.process_single_image(safe_file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        except Exception as e:
            print(f"\n❌ Error processing document: {str(e)}")
            traceback.print_exc()
            return None
    
    def process_single_image(self, image_path, original_file=None):
        """Process a single image through the document processing pipeline"""
        try:
            print(f"\nProcessing image: {image_path}")
            
            # Process the document using DocumentProcessor
            result = self.processor.process_document(image_path)
            
            # Save results to file
            output_filename = f"result_{os.path.basename(image_path)}.json"
            with open(output_filename, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            print(f"\nDetailed results saved to: {output_filename}")
            
            # Print summary
            self._print_result_summary(os.path.basename(image_path), result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error processing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_test_suite(self, test_dir="test"):
        """Run comprehensive tests on different document types.
        
        Note: This method processes all images in the test directory,
        categorizes them by document type, and collects detailed metrics.
        
        The complete processing pipeline is:
        1. Image → DocumentProcessor
        2. DocumentProcessor → DocumentClassifier (determine document type)
        3. DocumentProcessor → OCR (extract text)
        4. DocumentProcessor → LLMProcessor (enhance and validate fields)
        5. For passports: DocumentProcessor → MRZProcessor (extract MRZ)
        6. DocumentProcessor → Cross-validation (MRZ vs. visual text)
        7. DocumentProcessor → Final structured output
        """
        results = {
            'passports': [],
            'drivers_licenses': [],
            'summary': {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'average_confidence': 0.0
            }
        }
        
        # Process all images in test directory
        test_path = Path(test_dir)
        for image_file in test_path.glob("*.jpg"):
            print(f"\nProcessing: {image_file.name}")
            try:
                # Process document
                # Note: This single call triggers the entire pipeline
                doc_result = self.processor.process_document(str(image_file))
                
                # Add to appropriate category
                if doc_result['document_type'] == 'passport':
                    results['passports'].append(self._analyze_result(image_file.name, doc_result))
                elif doc_result['document_type'] == 'drivers_license':
                    results['drivers_licenses'].append(self._analyze_result(image_file.name, doc_result))
                
                # Update summary
                results['summary']['total_processed'] += 1
                results['summary']['successful'] += 1
                results['summary']['average_confidence'] += doc_result.get('overall_confidence', 0)
                
                # Print progress
                self._print_result_summary(image_file.name, doc_result)
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
                results['summary']['failed'] += 1
        
        # Calculate final average confidence
        if results['summary']['successful'] > 0:
            results['summary']['average_confidence'] /= results['summary']['successful']
        
        return results
    
    def _analyze_result(self, filename, result):
        """Analyze individual document results.
        
        Note: This method extracts key metrics from the processing results,
        including document type, confidence scores, and validation status.
        
        For passports, it also includes MRZ validation metrics, demonstrating
        the additional processing layer compared to driver's licenses.
        """
        analysis = {
            'filename': filename,
            'document_type': result['document_type'],
            'confidence': result.get('confidence', 0),
            'fields_extracted': len(result.get('extracted_fields', {})),
            'validation_status': self._get_validation_status(result)
        }
        
        if 'mrz_data' in result:
            analysis['mrz_validation'] = result.get('cross_validation', {}).get('confidence', 0)
        
        return analysis
    
    def _get_validation_status(self, result):
        """Determine overall validation status.
        
        Note: This implements a tiered confidence scoring system:
        - HIGH: >0.8 confidence
        - MEDIUM: 0.6-0.8 confidence
        - LOW: <0.6 confidence
        
        This multi-level validation is more sophisticated than the
        basic pass/fail approach in simpler implementations.
        """
        if result.get('overall_confidence', 0) > 0.8:
            return 'HIGH'
        elif result.get('overall_confidence', 0) > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _print_result_summary(self, filename, result):
        """Print a summary of the processing results with clear separation between OCR and LLM.
        
        Note: This method provides a detailed visualization of each step in the pipeline:
        1. Initial OCR detection
        2. Document classification
        3. LLM Enhanced Results
        4. Validation notes
        5. Cross-validation (for passports)
        
        This multi-step output demonstrates the value added at each stage
        of the processing pipeline, from raw OCR to validated fields.
        """
        print(f"\n{'='*50}")
        print(f"Results for {filename}:")
        print(f"{'='*50}")
        
        # STEP 1: Raw OCR Results
        print("\n📷 INITIAL OCR DETECTION:")
        print("-" * 40)
        if 'detected_fields' in result:
            for text_obj in result['detected_fields']:
                confidence = text_obj.get('confidence', 0)
                text = text_obj.get('text', '')
                confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
                print(f"• {text:<30} (Confidence: {confidence:.2f} - {confidence_level})")
        
        # STEP 2: Document Classification
        print(f"\n🔍 DOCUMENT CLASSIFICATION:")
        print("-" * 40)
        print(f"Type: {result['document_type']}")
        print(f"Classification Confidence: {result.get('confidence', 0):.2f}")
        
        # STEP 3: LLM Enhanced Results
        print(f"\n🤖 LLM ENHANCED FIELDS:")
        print("-" * 40)
        if 'extracted_fields' in result:
            for field, data in result['extracted_fields'].items():
                print(f"\n• {field.replace('_', ' ').title()}:")
                print(f"  - Raw Value: {data['value']}")
                print(f"  - Standardized: {data['standardized_value']}")
                print(f"  - Confidence: {data['confidence']:.2f}")
                print(f"  - Status: {data['validation_status']}")
        
        # STEP 4: Validation Notes
        if 'validation_notes' in result:
            print(f"\n⚠️ VALIDATION NOTES:")
            print("-" * 40)
            for note in result['validation_notes']:
                print(f"• {note}")
        
        # STEP 5: Cross-Validation (for passports)
        if 'cross_validation' in result:
            print(f"\n✅ CROSS-VALIDATION RESULTS:")
            print("-" * 40)
            print(f"Confidence: {result['cross_validation']['confidence']:.2f}")
            if result['cross_validation']['matches']:
                print("Matched Fields:", ', '.join(result['cross_validation']['matches']))
            if result['cross_validation']['mismatches']:
                print("\nMismatched Fields:")
                for mismatch in result['cross_validation']['mismatches']:
                    print(f"• {mismatch['field']}:")
                    print(f"  - MRZ: {mismatch['mrz_value']}")
                    print(f"  - Visual: {mismatch['visual_value']}")
        
        print(f"\n{'='*50}\n")

def main():
    """Main entry point for the enhanced OCR test suite.
    
    Note: This function demonstrates the complete workflow:
    1. Initialize the DocumentProcessor with API credentials
    2. Process all test images through the full pipeline
    3. Collect and analyze results
    4. Save detailed metrics to a JSON file
    
    This represents the most comprehensive testing approach,
    evaluating all components working together in the production pipeline.
    """
    # Get API key from environment variable
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("Please set FIREWORKS_API_KEY environment variable")
    
    # Initialize tester
    tester = EnhancedOCRTester(api_key)
    
    # Run tests
    print("Starting Enhanced OCR Test Suite...")
    results = tester.run_test_suite()
    
    # Save results
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nTest Suite Summary:")
    print(f"Total Documents Processed: {results['summary']['total_processed']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Average Confidence: {results['summary']['average_confidence']:.2f}")
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()

# Note: This script demonstrates the complete system architecture:
# 
# Full Pipeline (this script):
# Image → DocumentProcessor → DocumentClassifier → OCR → LLMProcessor → MRZProcessor → Cross-validation → Output
#
# Compared to simplified testing approaches:
# 
# test_llm.py (OCR+LLM):
# Image → OCR → LLMProcessor → Output
#
# test_llm_image.py (Direct Vision):
# Image → DeepSeek V3 → Output
#
# Each approach has different trade-offs:
# - Full pipeline: Most comprehensive, highest accuracy, most validation layers
# - OCR+LLM: Good balance of accuracy and simplicity, no cross-validation
# - Direct Vision: Simplest approach, no OCR preprocessing, potentially lower accuracy on complex documents 