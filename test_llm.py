"""
This script tests the integration between OCR engines and LLM processing.
It demonstrates the traditional OCR-first approach to document analysis.

Key differences from test_llm_image.py:
1. Uses OCR engines to extract text before LLM processing
2. Supports multiple OCR engines for comparison
3. Follows the original multi-step pipeline architecture
"""
# Note: This script is part of the original system architecture:
# Image → OCR → LLM → Output

from llm_processor import LLMProcessor
from ocr import OCR
import os
import json
import argparse
from dotenv import load_dotenv

def get_ocr_results(image_path, ocr_engine="easyocr"):
    """Get OCR results using the specified engine"""
    # Note: This function demonstrates the OCR-first approach
    # Unlike test_llm_image.py which sends images directly to the LLM,
    # this approach extracts text with OCR first, then sends text to LLM
    
    print(f"\nUsing OCR Engine: {ocr_engine}")
    ocr = OCR(os.path.dirname(image_path))
    
    # Process image with visualization
    results = ocr.process_image(image_path, engine=ocr_engine, visualization=True)
    
    if ocr_engine == "easyocr":
        return [(text.strip(), conf) for _, text, conf in results]
    else:
        return results  # Already in (text, confidence) format

    # Note: Each OCR engine returns results in a different format
    # - EasyOCR: [bbox, text, confidence]
    # - Tesseract and Keras: [text, confidence]
    # This function standardizes the output format to [text, confidence]

def test_llm_processing(test_image="test/passport-1.jpeg", ocr_engine="easyocr", doc_type="passport"):
    """Test LLM processing with different OCR engines"""
    # Note: This function demonstrates the traditional pipeline:
    # 1. OCR extracts text from image
    # 2. LLM processes and enhances the extracted text
    # 3. Results are formatted and displayed
    
    # Initialize processors
    llm = LLMProcessor()
    
    print(f"\nTesting with:")
    print(f"- Image: {test_image}")
    print(f"- OCR Engine: {ocr_engine}")
    print(f"- Document Type: {doc_type}")
    
    print("\n1. Getting OCR results...")
    try:
        ocr_results = get_ocr_results(test_image, ocr_engine)
        print(f"\nFound {len(ocr_results)} text regions")
        for text, confidence in ocr_results:
            print(f"- Text: {text:<30} (Confidence: {confidence:.2f})")
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        return
    
    # Note: This is where the OCR results are passed to the LLM
    # The LLM enhances the raw OCR text by:
    # - Identifying and extracting key fields
    # - Standardizing formats
    # - Validating data
    # - Providing confidence scores
    print("\n2. Processing with LLM...")
    llm_results = llm.process_document_fields(ocr_results, doc_type)
    
    print("\n3. Results:")
    if llm_results:
        print("\nExtracted Fields:")
        for field, data in llm_results['extracted_fields'].items():
            print(f"\n{field}:")
            print(f"  Value: {data['value']}")
            print(f"  Standardized: {data['standardized_value']}")
            print(f"  Confidence: {data['confidence']}")
            print(f"  Status: {data['validation_status']}")
        
        print("\nValidation Notes:")
        for note in llm_results['validation_notes']:
            print(f"- {note}")
        
        print(f"\nOverall Confidence: {llm_results['overall_confidence']}")
    else:
        print("Error processing document")

    # Note: The output format is the same as test_llm_image.py
    # This allows for direct comparison between the OCR-first approach
    # and the direct image-to-LLM approach

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test OCR with different engines')
    parser.add_argument('--engine', type=str, choices=['easyocr', 'tesseract', 'keras'],
                      default='easyocr', help='OCR engine to use')
    parser.add_argument('--image', type=str, default='test/passport-1.jpeg',
                      help='Path to the image file')
    parser.add_argument('--doc-type', type=str, choices=['passport', 'drivers_license'],
                      default='passport', help='Type of document')
    parser.add_argument('--all', action='store_true',
                      help='Test all OCR engines')
    
    args = parser.parse_args()
    
    if args.all:
        # Test with all OCR engines
        engines = ['easyocr', 'tesseract', 'keras']
        for engine in engines:
            print(f"\n{'='*80}")
            print(f"Testing with {engine.upper()}")
            print(f"{'='*80}")
            test_llm_processing(args.image, engine, args.doc_type)
    else:
        # Test with single engine
        test_llm_processing(args.image, args.engine, args.doc_type) 

# Note: This script is part of the original system architecture:
# Image → OCR → LLM → Output
#
# In contrast, test_llm_image.py uses a simplified approach:
# Image → LLM → Output
#
# Key differences:
# 1. This approach requires OCR preprocessing
# 2. It allows comparison between different OCR engines
# 3. It follows the traditional multi-step pipeline
# 4. It may have better performance on low-quality images where OCR preprocessing helps
# 5. It allows for specialized processing like MRZ detection

# Use Keras-OCR
# python test_llm.py --engine keras

# # Use EasyOCR
# python test_llm.py --engine easyocr

# # Use Tesseract
# python test_llm.py --engine tesseract