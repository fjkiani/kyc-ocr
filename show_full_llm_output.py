"""
Full LLM Output Viewer

This script processes a document with OCR and LLM and shows the complete detailed output,
similar to test_llm.py but with more verbose logging of the API responses.
"""

from llm_processor import LLMProcessor
from ocr import OCR
import os
import json
import argparse
import time
from dotenv import load_dotenv
import requests
from pathlib import Path

def get_ocr_results(image_path, ocr_engine="easyocr"):
    """Get OCR results using the specified engine"""
    print(f"\nUsing OCR Engine: {ocr_engine}")
    ocr = OCR(os.path.dirname(image_path))
    
    # Process image with visualization
    results = ocr.process_image(image_path, engine=ocr_engine, visualization=True)
    
    if ocr_engine == "easyocr":
        extracted_results = [(text.strip(), conf) for _, text, conf in results]
    else:
        extracted_results = results  # Already in (text, confidence) format
    
    print(f"\nFound {len(extracted_results)} text regions")
    for text, confidence in extracted_results:
        print(f"- Text: {text:<30} (Confidence: {confidence:.2f})")
    
    return extracted_results

class VerboseLLMProcessor(LLMProcessor):
    """Extended LLM Processor that shows more detailed output"""
    
    def _call_llm(self, prompt):
        """Override to provide more verbose output"""
        # For debugging - print if the API key is configured
        if not self.api_key or len(self.api_key.strip()) < 10:
            print(f"WARNING: API key looks invalid: '{self.api_key[:5]}...'")
        else:
            print(f"Using API key: '{self.api_key[:5]}...'")
            
        payload = {
            "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
            "max_tokens": 4096,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert document analyzer specializing in ID verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        print("\nSending request to Fireworks AI API...")
        print(f"API URL: {self.url}")
        print(f"Using model: {payload['model']}")
        
        start_time = time.time()
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload
            )
            
            elapsed_time = time.time() - start_time
            print(f"API call completed in {elapsed_time:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                # Print the full raw response
                print("\n=== RAW API RESPONSE ===")
                print(json.dumps(result, indent=2))
                print("=======================\n")
                
                message = result["choices"][0]["message"]["content"]
                return message
            else:
                print(f"API Error ({response.status_code}): {response.text}")
                error_messages = {
                    401: "Authentication error - check your API key",
                    404: "Model not found - check that you're using a valid model ID",
                    429: "Rate limit exceeded or quota exceeded",
                    500: "Server error from the API provider",
                    503: "Service unavailable - the API service might be down"
                }
                status_message = error_messages.get(response.status_code, "Unknown error")
                print(f"Error details: {status_message}")
                
                raise Exception(f"API Error: {status_message} (Status code: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            raise

def process_document(image_path, doc_type="passport", ocr_engine="easyocr"):
    """Process document and show detailed output"""
    # Load environment variables from .env file
    load_dotenv()
    
    print("\n" + "="*80)
    print(f"PROCESSING DOCUMENT: {image_path}")
    print(f"Document Type: {doc_type}")
    print(f"OCR Engine: {ocr_engine}")
    print("="*80)
    
    # Step 1: Get OCR results
    print("\n1. EXTRACTING TEXT WITH OCR")
    ocr_results = get_ocr_results(image_path, ocr_engine)
    
    # Step 2: Process with LLM
    print("\n2. PROCESSING WITH LLM")
    llm = VerboseLLMProcessor()
    
    # Show the structured OCR results
    structured_results = llm._structure_ocr_results(ocr_results)
    print("\nStructured OCR Results:")
    print(json.dumps(structured_results, indent=2))
    
    # Show the prompt that will be sent to the LLM
    prompt = llm._create_field_extraction_prompt(structured_results, doc_type)
    print("\nLLM Prompt:")
    print("-"*80)
    print(prompt)
    print("-"*80)
    
    # Process with LLM
    llm_results = llm.process_document_fields(ocr_results, doc_type)
    
    # Step 3: Display Results
    print("\n3. RESULTS AFTER PARSING")
    print("\nParsed Extracted Fields:")
    if llm_results and 'extracted_fields' in llm_results:
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
        print("Error processing document or no fields extracted")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Show full LLM output for document processing')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image file')
    parser.add_argument('--doc-type', type=str, choices=['passport', 'drivers_license', 'bank_statement', 'resume'],
                      default='passport', help='Type of document')
    parser.add_argument('--engine', type=str, choices=['easyocr', 'tesseract', 'keras'],
                      default='easyocr', help='OCR engine to use')
    
    args = parser.parse_args()
    
    process_document(args.image, args.doc_type, args.engine)

if __name__ == "__main__":
    main() 