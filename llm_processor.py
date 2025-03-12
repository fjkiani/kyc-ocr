# Note: This module is a key component in the document processing pipeline:

# In the traditional approach (test_llm.py):
# OCR Results → LLMProcessor → Structured Fields

# In the DocumentProcessor class:
# OCR Results → LLMProcessor → Cross-validation with MRZ → Final Output

# The LLMProcessor provides several critical enhancements:
# 1. Field identification and extraction
# 2. Format standardization (dates, names, numbers)
# 3. Validation of field formats and values
# 4. Confidence scoring for each field
# 5. Overall document validation notes 


import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

class LLMProcessor:
    def __init__(self, api_key=None):
        # Load environment variables from .env file
        env_path = Path('.') / '.env'
        load_dotenv(env_path)
        
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables or .env file")
            
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def process_document_fields(self, ocr_results, document_type):
        """Use LLM to process and validate document fields"""
        # Structure OCR results
        structured_results = self._structure_ocr_results(ocr_results)
        
        # Create prompt based on document type and OCR results
        prompt = self._create_field_extraction_prompt(structured_results, document_type)
        
        # Get LLM response
        response = self._call_llm(prompt)
        return self._parse_llm_response(response)
    
    def _structure_ocr_results(self, ocr_results):
        """Structure OCR results into a more organized format
        
        This method handles different input formats:
        1. From DocumentClassifier: List of dicts with 'text' and 'confidence' keys
        2. From OCR class: List of tuples with (bbox, text, confidence) or (text, confidence)
        """
        structured = {
            "detected_fields": []
        }
        
        # Print debug info
        print(f"DEBUG: OCR results type: {type(ocr_results)}")
        if ocr_results and len(ocr_results) > 0:
            print(f"DEBUG: First OCR result type: {type(ocr_results[0])}")
        
        for item in ocr_results:
            # Handle different formats
            if isinstance(item, dict) and 'text' in item and 'confidence' in item:
                # Already in the right format (from DocumentClassifier)
                structured["detected_fields"].append({
                    "text": item['text'].strip(),
                    "confidence": float(item['confidence'])
                })
            elif isinstance(item, tuple) or isinstance(item, list):
                # From OCR class
                if len(item) == 3:  # (bbox, text, confidence)
                    structured["detected_fields"].append({
                        "text": item[1].strip(),
                        "confidence": float(item[2])
                    })
                elif len(item) == 2:  # (text, confidence)
                    structured["detected_fields"].append({
                        "text": item[0].strip(),
                        "confidence": float(item[1])
                    })
                else:
                    print(f"WARNING: Unexpected OCR result format: {item}")
            else:
                print(f"WARNING: Unexpected OCR result type: {type(item)}")
        
        print(f"DEBUG: Structured {len(structured['detected_fields'])} fields")
        return structured
    
    def _create_field_extraction_prompt(self, ocr_results, document_type):
        """Create a structured prompt for field extraction"""
        prompt = f"""As an expert document analyzer, analyze these OCR results from a {document_type}.
        The results include text and confidence scores for each detected field.

        Document Type: {document_type}
        OCR Results:
        {json.dumps(ocr_results, indent=2)}

        Instructions:
        1. Identify and extract key fields based on the document type
        2. For passports, look for:
           - Full Name (First, Middle, Last)
           - Date of Birth
           - Place of Birth
           - Passport Number
           - Issue Date
           - Expiry Date
           - Issuing Authority
           - MRZ Data (if present)
        
        3. For driver's licenses, look for:
           - Full Name
           - Address
           - Date of Birth
           - License Number
           - Issue/Expiry Dates
           - Class/Restrictions
        
        4. For each identified field:
           - Clean and standardize the value
           - Validate the format
           - Provide a confidence score
           - Note any issues or inconsistencies

        Return the data in this JSON format:
        {{
            "extracted_fields": {{
                "field_name": {{
                    "value": "raw_value",
                    "standardized_value": "formatted_value",
                    "confidence": 0.95,
                    "validation_status": "valid|invalid|uncertain"
                }}
            }},
            "validation_notes": ["list of any issues found"],
            "overall_confidence": 0.90
        }}"""
        return prompt

    def _call_llm(self, prompt):
        """Call Fireworks AI with specified configuration"""
        payload = {
            "model": "accounts/fireworks/models/deepseek-v3",
            "max_tokens": 16384,
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

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling Fireworks AI: {str(e)}")
            return None

    def _parse_llm_response(self, response):
        """Parse and validate LLM response"""
        if not response or 'choices' not in response:
            return None
            
        try:
            content = response['choices'][0]['message']['content']
            # Extract JSON from response (handle case where LLM might add extra text)
            json_str = content[content.find('{'):content.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return None 