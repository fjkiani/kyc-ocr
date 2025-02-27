from document_processor import DocumentProcessor
import json
import os

def main():
    # Get API key from environment variable
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("Please set FIREWORKS_API_KEY environment variable")
    
    processor = DocumentProcessor(api_key)
    
    # Test with a passport
    passport_result = processor.process_document("test/passport-2.jpg")
    print("\nPassport Results:", json.dumps(passport_result, indent=2))
    
    # Test with a driver's license
    license_result = processor.process_document("test/license-1.jpg")
    print("\nLicense Results:", json.dumps(license_result, indent=2))

if __name__ == "__main__":
    main()


