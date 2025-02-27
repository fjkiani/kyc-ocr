from llm_processor import LLMProcessor
from ocr import OCR
import os
import json

def test_llm_processing():
    # Initialize processors
    llm = LLMProcessor()
    
    # Test with a sample document
    test_image = "test/passport-2.jpg"
    
    print("1. Getting OCR results...")
    # Create a list of tuples (text, confidence) to simulate OCR results
    ocr_results = [
        ("PASSPORT", 0.97),
        ("UNITED STATES OF AMERICA", 0.85),
        ("JOHN", 1.00),
        ("DOE", 1.00),
        ("Date of Birth: 15 Mar 1996", 1.00),
        ("Place of Birth: CALIFORNIA, U.S.A", 0.97),
        ("Issue Date: 14 Apr 2017", 0.72),
        ("Expiry: 2027", 0.90),
        ("Passport No: 963545637", 0.69),
        ("Department of State", 0.85),
        ("P<USAJOHN<<DOE<<<<<<kk<<<kk<kkkkkk<<kkkk<<<", 0.03),
        ("9635456374U5A9603150M2704140202113962<804330", 0.66)
    ]
    
    print("\nOCR Results:")
    for text, conf in ocr_results:
        print(f"- Text: {text:<30} (Confidence: {conf:.2f})")
    
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