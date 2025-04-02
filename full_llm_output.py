#!/usr/bin/env python3
"""
Full LLM Processing Output

This script runs the document processing pipeline and displays the complete
details of LLM processing, including the raw API responses.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import requests

# Import processing components
from ocr import OCR
from llm_processor import LLMProcessor

class VerboseLLMProcessor(LLMProcessor):
    """Extended LLM Processor with verbose output"""
    
    def _call_llm(self, prompt):
        """Override to provide more verbose output"""
        # For debugging - print if the API key is configured
        if not self.api_key or len(self.api_key.strip()) < 10:
            print(f"\n⚠️ WARNING: API key looks invalid: '{self.api_key[:5]}...'")
        else:
            print(f"\n🔑 Using API key: '{self.api_key[:5]}...'")
            
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
        
        print("\n📤 SENDING REQUEST TO FIREWORKS AI API")
        print(f"API URL: {self.url}")
        print(f"Using model: {payload['model']}")
        
        # Print a summary of the prompt (which can be very long)
        print(f"\nSystem prompt: {payload['messages'][0]['content']}")
        content_lines = payload['messages'][1]['content'].split('\n')
        print(f"User prompt: {content_lines[0]}... ({len(content_lines)} lines)")
        
        print("\n⏳ Sending request and waiting for response...")
        start_time = time.time()
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload
            )
            
            elapsed_time = time.time() - start_time
            print(f"\n✅ API call completed in {elapsed_time:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Print the raw response
                print("\n🔍 RAW API RESPONSE:")
                print("=" * 80)
                print(json.dumps(result, indent=2))
                print("=" * 80)
                
                # Extract and return the message content
                message = result["choices"][0]["message"]["content"]
                return message
            else:
                print(f"\n❌ API Error ({response.status_code}): {response.text}")
                error_messages = {
                    401: "Authentication error - check your API key",
                    404: "Model not found or unavailable - check that you're using a valid model ID",
                    429: "Rate limit exceeded or quota exceeded",
                    500: "Server error from the API provider",
                    503: "Service unavailable - the API service might be down"
                }
                status_message = error_messages.get(response.status_code, "Unknown error")
                print(f"Error details: {status_message}")
                
                raise Exception(f"API Error: {status_message} (Status code: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {str(e)}")
            raise

def process_document(image_path, doc_type="bank_statement"):
    """Process a document and show full details of LLM processing"""
    print("\n" + "=" * 80)
    print(f"PROCESSING DOCUMENT: {image_path}")
    print(f"Document Type: {doc_type}")
    print("=" * 80)
    
    # Initialize processors
    ocr_processor = OCR(os.path.dirname(image_path))
    llm_processor = VerboseLLMProcessor()
    
    # Step 1: Process with OCR to get text
    print("\n📷 STEP 1: EXTRACTING TEXT WITH OCR")
    ocr_results = ocr_processor.process_image(image_path, engine='easyocr', visualization=False)
    
    if not ocr_results or len(ocr_results) == 0:
        print("❌ OCR processing failed to extract any text")
        return
    
    print(f"\n✅ Extracted {len(ocr_results)} text elements")
    print("\nSample extracted text:")
    
    # Show a sample of the OCR results (first 20 items)
    for i, item in enumerate(ocr_results[:20]):
        if isinstance(item, tuple) and len(item) >= 2:
            text = item[1] if len(item) == 3 else item[0]
            confidence = item[2] if len(item) == 3 else item[1]
            confidence_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW"
            print(f"- Text: {text:<30} (Confidence: {confidence:.2f} - {confidence_label})")
    
    # If more than 20 items, show a message
    if len(ocr_results) > 20:
        print(f"... and {len(ocr_results) - 20} more items")
    
    # Step 2: Process with LLM
    print("\n🧠 STEP 2: PROCESSING WITH LLM")
    
    # Structure the OCR results
    structured_results = llm_processor._structure_ocr_results(ocr_results)
    
    # Show structured results
    print("\n📊 STRUCTURED OCR RESULTS:")
    print(json.dumps(structured_results, indent=2))
    
    # Create and show prompt
    prompt = llm_processor._create_field_extraction_prompt(structured_results, doc_type)
    print("\n📝 LLM PROMPT:")
    print("-" * 80)
    # Print first 10 lines and last 5 lines with "..." in between if the prompt is very long
    prompt_lines = prompt.split('\n')
    if len(prompt_lines) > 20:
        print('\n'.join(prompt_lines[:10]))
        print("\n... [prompt truncated for display] ...\n")
        print('\n'.join(prompt_lines[-5:]))
    else:
        print(prompt)
    print("-" * 80)
    
    # Process with LLM
    print("\n⏳ Calling LLM to process document...")
    try:
        llm_results = llm_processor.process_document_fields(ocr_results, doc_type)
    
        # Step 3: Display Results
        print("\n🔍 STEP 3: PROCESSING RESULTS")
        print("\n📋 Parsed Extracted Fields:")
        
        if llm_results and 'extracted_fields' in llm_results:
            # Print each field with its details
            for field, data in llm_results['extracted_fields'].items():
                print(f"\n{field}:")
                print(f"  Value: {data['value']}")
                print(f"  Standardized: {data['standardized_value']}")
                print(f"  Confidence: {data['confidence']}")
                print(f"  Status: {data['validation_status']}")
            
            # Print validation notes
            print("\nValidation Notes:")
            for note in llm_results['validation_notes']:
                print(f"- {note}")
            
            # Print overall confidence
            print(f"\nOverall Confidence: {llm_results['overall_confidence']}")
        else:
            print("❌ Error processing document or no fields extracted")
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n❌ Error during LLM processing: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Process a document with full LLM output')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image file')
    parser.add_argument('--doc-type', type=str, choices=['passport', 'drivers_license', 'bank_statement', 'resume'],
                      default='bank_statement', help='Type of document')
    
    args = parser.parse_args()
    
    # Check if the image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Process the document
    process_document(str(image_path), args.doc_type)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 