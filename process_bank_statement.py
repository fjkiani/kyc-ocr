#!/usr/bin/env python3
"""
Bank Statement Processor - Full Output

This script processes a bank statement and shows the complete LLM processing details.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from ocr import OCR
from llm_processor import LLMProcessor

# Load environment variables
load_dotenv()

def main():
    # Hard-coded bank statement sample path
    image_path = "test/bank_statement_sample.jpg"
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        print("Using fallback to passport sample")
        image_path = "test/passport-1.jpeg"
        if not os.path.exists(image_path):
            print(f"Error: Could not find fallback image either.")
            return
    
    print(f"Processing document: {image_path}")
    
    # 1. OCR Processing
    print("\n=== STEP 1: OCR PROCESSING ===")
    ocr = OCR(os.path.dirname(image_path))
    ocr_results = ocr.easyocr_model_works(image_path, visualization=True)
    
    print(f"\nExtracted {len(ocr_results)} text elements:")
    for i, result in enumerate(ocr_results):
        bbox, text, confidence = result
        print(f"{i+1}. '{text}' (Confidence: {confidence:.2f})")
    
    # 2. LLM Processing
    print("\n=== STEP 2: LLM PROCESSING ===")
    
    # Create LLM processor with verbose output
    class DetailedLLMProcessor(LLMProcessor):
        def _call_llm(self, prompt):
            print("\n--- LLM REQUEST DETAILS ---")
            if not self.api_key or len(self.api_key.strip()) < 10:
                print(f"WARNING: API key looks invalid: '{self.api_key[:5]}...'")
            else:
                print(f"Using API key: '{self.api_key[:5]}...'")
            
            # Print the prompt (first few lines)
            prompt_lines = prompt.split('\n')
            print(f"\nPrompt first 5 lines (total {len(prompt_lines)} lines):")
            for line in prompt_lines[:5]:
                print(f"> {line}")
            print("...")
            
            # Make the API call with more verbose output
            print("\nSending request to Fireworks AI API...")
            
            import time
            import requests
            
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
            
            start_time = time.time()
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload
            )
            
            elapsed_time = time.time() - start_time
            print(f"API call completed in {elapsed_time:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Print the full API response
                print("\n--- RAW API RESPONSE ---")
                print(json.dumps(result, indent=2))
                print("----------------------")
                
                message = result["choices"][0]["message"]["content"]
                
                # Print the content of the response
                print("\n--- LLM RESPONSE CONTENT ---")
                print(message)
                print("----------------------------")
                
                return message
            else:
                print(f"API Error ({response.status_code}): {response.text}")
                raise Exception(f"API Error (Status code: {response.status_code})")
    
    # Use the detailed processor
    llm = DetailedLLMProcessor()
    doc_type = "bank_statement"
    
    try:
        # Convert OCR results to format expected by LLM processor
        formatted_ocr_results = []
        for bbox, text, confidence in ocr_results:
            formatted_ocr_results.append((text, confidence))
        
        # Process with LLM
        llm_results = llm.process_document_fields(formatted_ocr_results, doc_type)
        
        # 3. Results
        print("\n=== STEP 3: FINAL RESULTS ===")
        
        if llm_results and 'extracted_fields' in llm_results:
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
            print("Error: No valid results returned from LLM")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 