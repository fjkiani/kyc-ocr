from llm_processor import LLMProcessor
from ocr import OCR
import os
import json

def test_llm_processing():
    # Initialize processors
    llm = LLMProcessor()
    
    # Test with a sample document
    test_image = "test/License-1.png"
    
    print("1. Getting OCR results...")
    # Initialize OCR with the image directory
    ocr = OCR(os.path.dirname(test_image))
    
    # Get actual OCR results
    print("\nProcessing image with EasyOCR...")
    ocr_results = []
    results = ocr.text_reader.readtext(test_image)
    
    for (bbox, text, confidence) in results:
        ocr_results.append((text.strip(), confidence))
        print(f"- Text: {text:<30} (Confidence: {confidence:.2f})")
    
    print("\n2. Processing with LLM...")
    llm_results = llm.process_document_fields(ocr_results, "passport")
    
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

if __name__ == "__main__":
    test_llm_processing() 