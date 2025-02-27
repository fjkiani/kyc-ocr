from document_processor import DocumentProcessor
import os
import json
from pathlib import Path

class EnhancedOCRTester:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        self.processor = DocumentProcessor(self.api_key)
        
    def run_test_suite(self, test_dir="test"):
        """Run comprehensive tests on different document types"""
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
        """Analyze individual document results"""
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
        """Determine overall validation status"""
        if result.get('overall_confidence', 0) > 0.8:
            return 'HIGH'
        elif result.get('overall_confidence', 0) > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _print_result_summary(self, filename, result):
        """Print a summary of the processing results"""
        print(f"\nResults for {filename}:")
        print(f"Document Type: {result['document_type']}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        
        if 'extracted_fields' in result:
            print("\nExtracted Fields:")
            for field, data in result['extracted_fields'].items():
                print(f"- {field}: {data['value']} (Confidence: {data['confidence']:.2f})")
        
        if 'validation_notes' in result:
            print("\nValidation Notes:")
            for note in result['validation_notes']:
                print(f"- {note}")
        
        if 'cross_validation' in result:
            print("\nCross-Validation Results:")
            print(f"Confidence: {result['cross_validation']['confidence']:.2f}")
            if result['cross_validation']['matches']:
                print("Matched Fields:", ', '.join(result['cross_validation']['matches']))
            if result['cross_validation']['mismatches']:
                print("Mismatched Fields:", len(result['cross_validation']['mismatches']))

def main():
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