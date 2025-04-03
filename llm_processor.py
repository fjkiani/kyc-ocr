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
        """
        
        if document_type == 'passport':
            prompt += """
        2. For passports, look for:
           - Full Name (First, Middle, Last)
           - Date of Birth
           - Place of Birth
           - Passport Number
           - Issue Date
           - Expiry Date
           - Issuing Authority
           - MRZ Data (if present)
            """
        elif document_type == 'drivers_license':
            prompt += """
        2. For driver's licenses, look for:
           - Full Name
           - Address
           - Date of Birth
           - License Number
           - Issue/Expiry Dates
           - Class/Restrictions
            """
        elif document_type == 'bank_statement':
            prompt += """
        2. For bank statements, look for:
           - Account Holder Name
           - Account Number
           - Bank Name
           - Statement Period or Date
           - Opening/Closing Balance
           - Transaction Details (if visible)
           - Contact Information
            """
        elif document_type == 'resume':
            prompt += """
        2. For resumes/CVs, look for:
           - Full Name
           - Contact Information (Email, Phone, LinkedIn, Website)
           - Education (Degrees, Institutions, Dates)
           - Work Experience (Job Titles, Companies, Dates, Responsibilities)
           - Skills (Technical Skills, Soft Skills)
           - Certifications/Licenses
           - Projects/Portfolio
           - Languages
           - Awards/Achievements
           
           For each work experience entry, try to extract:
           - Job Title
           - Company Name
           - Dates of Employment
           - Key Responsibilities/Achievements
           
           For each education entry, try to extract:
           - Degree/Certificate
           - Institution Name
           - Dates Attended
           - Major/Field of Study
            """
        else:
            prompt += """
        2. For general documents, look for:
           - Names of Individuals
           - Organizations/Companies
           - Dates
           - Addresses
           - Contact Information
           - Key Terms/Topics
           - Document Purpose/Type
            """
        
        prompt += """
        3. For each identified field:
           - Clean and standardize the value
           - Validate the format
           - Provide a confidence score (between 0.0 and 1.0)
           - Note any issues or inconsistencies

        4. IMPORTANT: You MUST return your response in the following JSON format:
        {
            "extracted_fields": {
                "field_name1": {
                    "value": "raw_value",
                    "standardized_value": "formatted_value",
                    "confidence": 0.95,
                    "validation_status": "valid"
                },
                "field_name2": {
                    "value": "raw_value2",
                    "standardized_value": "formatted_value2",
                    "confidence": 0.80,
                    "validation_status": "valid"
                }
            },
            "validation_notes": ["list", "of", "any", "issues", "found"],
            "overall_confidence": 0.90
        }

        CRITICAL: Your response MUST be valid JSON that can be parsed by json.loads().
        Do not include any text before or after the JSON.
        Do not include explanations, introductions, or conclusions.
        Start your response with '{' and end with '}'.
        Do not use markdown code blocks - just return the raw JSON.
        """
        return prompt

    def _call_llm(self, prompt):
        """Call Fireworks AI with specified configuration"""
        # For debugging - print if the API key is configured
        if not self.api_key or len(self.api_key.strip()) < 10:
            print(f"WARNING: API key looks invalid: '{self.api_key[:5]}...'")
        else:
            print(f"Using API key: '{self.api_key[:5]}...'")
            
        payload = {
            # Use a model that's known to be available on Fireworks AI
            "model": "accounts/fireworks/models/mixtral-8x7b-instruct",  # Popular model that should be available
            "max_tokens": 4096,  # Reduced token count for faster response
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
            print(f"Sending request to Fireworks AI API...")
            print(f"API URL: {self.url}")
            print(f"Using model: {payload['model']}")
            
            import time
            start_time = time.time()
            
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=30  # Add timeout to prevent indefinite hanging
            )
            
            elapsed = time.time() - start_time
            print(f"API call completed in {elapsed:.2f} seconds with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_message = f"API Error: HTTP {response.status_code}"
                if response.status_code == 404:
                    error_message += f" - Model '{payload['model']}' not found or not available"
                elif response.status_code == 401:
                    error_message += " - API key invalid or unauthorized"
                elif response.status_code == 429:
                    error_message += " - Rate limit exceeded"
                
                print(error_message)
                print(f"Response content: {response.text[:500]}")  # Print first 500 chars of error
                return None
            
            # Check response content type and try to parse as JSON
            content_type = response.headers.get('Content-Type', '')
            print(f"Response Content-Type: {content_type}")
            
            try:
                # Try to parse as JSON
                if 'application/json' in content_type:
                    result = response.json()
                    print(f"Successfully parsed response as JSON with keys: {list(result.keys())}")
                    return result
                else:
                    # If not JSON content type, try to parse anyway
                    print(f"Warning: Response Content-Type is not JSON: '{content_type}'")
                    try:
                        result = response.json()
                        print(f"Successfully parsed non-JSON-content-type response as JSON")
                        return result
                    except json.JSONDecodeError:
                        # If can't parse as JSON, use the text content
                        print(f"Response is not JSON. Treating as text. First 100 chars: {response.text[:100]}...")
                        
                        # Create a synthetic response in the expected format
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": response.text
                                    }
                                }
                            ]
                        }
            except json.JSONDecodeError as json_err:
                print(f"Warning: Could not parse response as JSON: {str(json_err)}")
                print(f"Response content type: {type(response.text)}")
                print(f"Response preview: {response.text[:100]}...")
                
                # Create a synthetic response in the expected format
                return {
                    "choices": [
                        {
                            "message": {
                                "content": response.text
                            }
                        }
                    ]
                }
                
        except requests.exceptions.Timeout:
            print("Error: API call timed out after 30 seconds")
            return None
        except requests.exceptions.ConnectionError:
            print("Error: Connection failed. Check your internet connection")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error calling Fireworks AI API: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error calling Fireworks AI: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_llm_response(self, response):
        """Parse and validate the LLM response
        
        Args:
            response (str): Raw response from the LLM
            
        Returns:
            dict: Structured response with extracted fields and metadata
        """
        if not response:
            print("Warning: Empty response from LLM")
            return {
                'status': 'error',
                'message': 'Empty response from LLM',
                'extracted_fields': {},
                'validation_notes': ['No data could be extracted'],
                'confidence': 0.0
            }
        
        try:
            # First try to parse the entire response as JSON
            try:
                result = json.loads(response)
                print("Successfully parsed response as JSON")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                print("Initial JSON parse failed, attempting to extract JSON from text")
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    print("Successfully extracted and parsed JSON from text")
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Validate required fields
            if 'extracted_fields' not in result:
                print("Warning: No extracted_fields in response")
                result['extracted_fields'] = {}
            
            # Ensure all fields have the required structure
            for field_name, field_data in result['extracted_fields'].items():
                if isinstance(field_data, str):
                    # Convert simple string values to full structure
                    result['extracted_fields'][field_name] = {
                        'value': field_data,
                        'standardized_value': field_data,
                        'confidence': 0.8,
                        'validation_status': 'valid'
                    }
                elif isinstance(field_data, dict):
                    # Ensure all required keys exist
                    field_data.setdefault('value', '')
                    field_data.setdefault('standardized_value', field_data['value'])
                    field_data.setdefault('confidence', 0.8)
                    field_data.setdefault('validation_status', 'valid')
            
            # Add validation notes if not present
            if 'validation_notes' not in result:
                result['validation_notes'] = []
            
            # Calculate overall confidence
            confidences = [
                field['confidence'] 
                for field in result['extracted_fields'].values() 
                if isinstance(field, dict) and 'confidence' in field
            ]
            result['confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Add processing metadata
            result['status'] = 'completed'
            result['message'] = 'Successfully processed document'
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Raw response: {response[:500]}...")  # Print first 500 chars for debugging
            return {
                'status': 'error',
                'message': f'Failed to parse LLM response: {str(e)}',
                'extracted_fields': {},
                'validation_notes': ['Error processing document'],
                'confidence': 0.0
            }
    
    def _create_fallback_response(self, content):
        """Create a fallback response when JSON parsing fails"""
        # Parse the content for potential field values based on doc type patterns
        field_values = {}
        
        # Look for common patterns in the text
        lines = content.split('\n')
        current_field = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for field name patterns
            if ':' in line and not line.startswith('-'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_name = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    
                    if field_name and value:
                        field_values[field_name] = {
                            "value": value,
                            "standardized_value": value,
                            "confidence": 0.6,
                            "validation_status": "extracted_from_text"
                        }
        
        # If we found some fields, use them
        if field_values:
            return {
                "extracted_fields": field_values,
                "validation_notes": ["Fields extracted from unstructured text response"],
                "overall_confidence": 0.5
            }
        
        # Otherwise, use the entire content
        return {
            "extracted_fields": {
                "raw_content": {
                    "value": content,
                    "standardized_value": content,
                    "confidence": 0.5,
                    "validation_status": "unstructured"
                }
            },
            "validation_notes": ["Failed to parse structured fields from response"],
            "overall_confidence": 0.4
        } 