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

from document_processor import DocumentProcessor, NumpyEncoder
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import argparse

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
    
    def process_single_image(self, image_path):
        """Process a single image through the complete pipeline.
        
        This method provides detailed error handling and debugging for a single image,
        making it easier to identify where issues occur in the pipeline.
        """
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        
        try:
            # Step 1: Document Classification
            print("\n1. Starting document classification...")
            classification = self.processor.classifier.classify(image_path)
            print(f"   Document type: {classification['document_type']}")
            print(f"   Confidence: {float(classification['confidence']):.2f}")
            print(f"   Detected {len(classification['detected_fields'])} text regions")
            
            # Step 2: LLM Processing
            print("\n2. Starting LLM processing...")
            try:
                # Print the first few detected fields for debugging
                print("   Sample detected fields:")
                for i, field in enumerate(classification['detected_fields'][:3]):
                    print(f"   - Field {i+1}: {field['text']} (Confidence: {float(field['confidence']):.2f})")
                
                # Process with LLM
                llm_results = self.processor.llm_processor.process_document_fields(
                    classification['detected_fields'],
                    classification['document_type']
                )
                
                if llm_results:
                    print("   ✅ LLM processing successful")
                    print(f"   Extracted {len(llm_results['extracted_fields'])} fields")
                    print(f"   Overall confidence: {float(llm_results['overall_confidence']):.2f}")
                else:
                    print("   ❌ LLM processing returned no results")
            except Exception as e:
                print(f"   ❌ Error in LLM processing: {str(e)}")
                import traceback
                print(traceback.format_exc())
                llm_results = None
            
            # Step 3: MRZ Processing (for passports)
            mrz_data = None
            if classification['document_type'] == 'passport' and classification.get('has_mrz'):
                print("\n3. Starting MRZ processing...")
                try:
                    mrz_data = self.processor.mrz_processor.extract_mrz(image_path)
                    if mrz_data:
                        print("   ✅ MRZ processing successful")
                        print(f"   Extracted {len(mrz_data)} MRZ fields")
                    else:
                        print("   ⚠️ No MRZ data found")
                except Exception as e:
                    print(f"   ❌ Error in MRZ processing: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # Step 4: Combine results
            result = {
                'document_type': classification['document_type'],
                'confidence': classification['confidence'],
                'detected_fields': classification['detected_fields']
            }
            
            if llm_results:
                result['extracted_fields'] = llm_results['extracted_fields']
                result['validation_notes'] = llm_results['validation_notes']
                result['overall_confidence'] = llm_results['overall_confidence']
            
            # Step 5: Cross-validation (for passports with MRZ)
            if classification['document_type'] == 'passport' and mrz_data and llm_results:
                print("\n4. Performing cross-validation...")
                try:
                    cross_validation = self.processor._cross_validate_mrz(
                        mrz_data,
                        llm_results['extracted_fields']
                    )
                    result['mrz_data'] = mrz_data
                    result['cross_validation'] = cross_validation
                    print(f"   Cross-validation confidence: {float(cross_validation['confidence']):.2f}")
                    print(f"   Matched fields: {len(cross_validation['matches'])}")
                    print(f"   Mismatched fields: {len(cross_validation['mismatches'])}")
                except Exception as e:
                    print(f"   ❌ Error in cross-validation: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # Print detailed results
            self._print_result_summary(os.path.basename(image_path), result)
            
            # Save individual result to file
            output_file = f"result_{os.path.basename(image_path)}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            print(f"\nDetailed results saved to: {output_file}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error processing image: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
                # Process document one by one
                doc_result = self.process_single_image(str(image_file))
                
                if doc_result:
                    # Add to appropriate category
                    if doc_result['document_type'] == 'passport':
                        results['passports'].append(self._analyze_result(image_file.name, doc_result))
                    elif doc_result['document_type'] == 'drivers_license':
                        results['drivers_licenses'].append(self._analyze_result(image_file.name, doc_result))
                    
                    # Update summary
                    results['summary']['total_processed'] += 1
                    results['summary']['successful'] += 1
                    results['summary']['average_confidence'] += doc_result.get('overall_confidence', 0)
                else:
                    results['summary']['failed'] += 1
                    results['summary']['total_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                results['summary']['failed'] += 1
                results['summary']['total_processed'] += 1
        
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
            'confidence': float(result.get('confidence', 0)),  # Convert numpy types to Python types
            'fields_extracted': len(result.get('extracted_fields', {})),
            'validation_status': self._get_validation_status(result)
        }
        
        if 'mrz_data' in result:
            analysis['mrz_validation'] = float(result.get('cross_validation', {}).get('confidence', 0))
        
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
        3. LLM-enhanced field extraction
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
                confidence = float(text_obj.get('confidence', 0))  # Convert numpy types
                text = text_obj.get('text', '')
                confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
                print(f"• {text:<30} (Confidence: {confidence:.2f} - {confidence_level})")
        
        # STEP 2: Document Classification
        print(f"\n🔍 DOCUMENT CLASSIFICATION:")
        print("-" * 40)
        print(f"Type: {result['document_type']}")
        print(f"Classification Confidence: {float(result.get('confidence', 0)):.2f}")
        
        # STEP 3: LLM Enhanced Results
        print(f"\n🤖 LLM ENHANCED FIELDS:")
        print("-" * 40)
        if 'extracted_fields' in result:
            for field, data in result['extracted_fields'].items():
                print(f"\n• {field.replace('_', ' ').title()}:")
                print(f"  - Raw Value: {data['value']}")
                print(f"  - Standardized: {data['standardized_value']}")
                print(f"  - Confidence: {float(data['confidence']):.2f}")
                print(f"  - Status: {data['validation_status']}")
        else:
            print("No LLM-enhanced fields available")
        
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
            print(f"Confidence: {float(result['cross_validation']['confidence']):.2f}")
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
    """Main entry point for the enhanced OCR test suite."""
    parser = argparse.ArgumentParser(description='Test the enhanced OCR pipeline')
    parser.add_argument('--image', type=str, help='Process a single image')
    parser.add_argument('--dir', type=str, default='test', help='Directory containing test images')
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("Please set FIREWORKS_API_KEY environment variable")
    
    # Initialize tester
    tester = EnhancedOCRTester(api_key)
    
    if args.image:
        # Process a single image
        print(f"Processing single image: {args.image}")
        tester.process_single_image(args.image)
    else:
        # Run the full test suite
        print("Starting Enhanced OCR Test Suite...")
        results = tester.run_test_suite(args.dir)
        
        # Save results
        output_file = "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
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