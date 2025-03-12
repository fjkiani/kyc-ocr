"""
This script processes document images (passports and driver's licenses) using DeepSeek V3's vision capabilities.
It demonstrates direct image analysis without requiring OCR preprocessing.
"""

from llm_processor import LLMProcessor
import argparse
import json
import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

# Note: We're importing LLMProcessor but not using its full functionality
# This script demonstrates a standalone approach using DeepSeek V3 directly

def resize_image(image_path, max_size=(800, 800)):
    """Resize image while maintaining aspect ratio.
    
    Implementation notes:
    - Max size of 800x800 was chosen to balance quality and token limit
    - Using LANCZOS resampling for best quality
    - Converting to RGB to ensure compatibility
    - JPEG format with 85% quality reduces size while maintaining readability
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate new size maintaining aspect ratio
        ratio = min(max_size[0]/img.size[0], max_size[1]/img.size[1])
        new_size = tuple([int(x*ratio) for x in img.size])
        
        # Resize image
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to bytes with quality setting for size reduction
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        return img_byte_arr

    # Note: Image resizing is critical to stay within DeepSeek V3's context window limit (131,071 tokens)
    # Without resizing, large images would cause API errors due to token limits

def encode_image(image_path):
    """Convert image to base64 string.
    
    Implementation notes:
    - Resizes image first to reduce token count
    - Returns base64 string ready for API submission
    """
    img_bytes = resize_image(image_path)
    return base64.b64encode(img_bytes).decode('utf-8')

    # Note: Base64 encoding  is required for including images in API requests
    # The encoding increases the size by ~33%, which is why resizing is important

def get_system_prompt(doc_type):
    """Get the system prompt for document analysis.
    
    Implementation notes:
    - Structured to clearly specify expected fields
    - Includes format requirements for standardization
    - JSON template ensures consistent response format
    """
    json_format = '''
    {
        "extracted_fields": {
            "field_name": {
                "value": "raw_value",
                "standardized_value": "formatted_value",
                "confidence": 0.95,
                "validation_status": "valid|invalid|uncertain"
            }
        },
        "validation_notes": ["list of any issues found"],
        "overall_confidence": 0.90
    }
    '''
    
    return f"""You are an expert document analyzer. Analyze this {doc_type} image and extract all relevant information.
    For passports, extract:
    - Full Name (First, Middle, Last)
    - Date of Birth
    - Place of Birth
    - Passport Number
    - Issue Date
    - Expiry Date
    - Issuing Authority
    - MRZ Data (if visible)
    
    For driver's licenses, extract:
    - Full Name
    - Address
    - Date of Birth
    - License Number
    - Issue/Expiry Dates
    - Class/Restrictions
    
    For each field:
    - Provide the raw value as seen in the image
    - Provide a standardized version of the value
    - Indicate your confidence in the extraction (0.0-1.0)
    - Note any issues or uncertainties
    
    Return the data in this JSON format:
    {json_format}"""

    # Note: Prompt engineering is critical for getting structured results
    # The prompt design mirrors what DocumentProcessor would normally coordinate:
    # - Document type-specific field extraction
    # - Standardization requirements
    # - Confidence scoring
    # - Validation status

def process_image_with_llm(image_path, doc_type="passport"):
    """Process image directly with LLM vision capabilities.
    
    Implementation notes:
    - Uses DeepSeek V3's vision capabilities
    - Combines image and prompt in specific format required by the model
    - Handles API interaction and response parsing
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found in environment variables")

    # Encode image
    base64_image = encode_image(image_path)
    
    # Get system prompt
    system_prompt = get_system_prompt(doc_type)

    # Prepare API request
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use DeepSeek V3 for image processing with correct format
    # Note: DeepSeek V3 expects images as base64 strings wrapped in <image> tags
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3",
        "max_tokens": 4096,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"<image>{base64_image}</image>\nPlease analyze this {doc_type} image and extract all relevant information following the format specified."
            }
        ]
    }

    # Note: This format is specific to DeepSeek V3
    # We discovered through testing that:
    # 1. Images must be wrapped in <image> tags
    # 2. The content must be a simple string, not a complex object
    # 3. The image and text prompt must be combined in a single message

    # Make API request
    print(f"\nProcessing {doc_type} with DeepSeek V3...")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract JSON from response
        content = result['choices'][0]['message']['content']
        json_str = content[content.find('{'):content.rfind('}')+1]
        llm_results = json.loads(json_str)
        
        # Note: This JSON extraction is necessary because the LLM might include
        # explanatory text before or after the JSON structure

        # Print results
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
        
        # Save results
        output_file = f"llm_vision_results_{os.path.basename(image_path)}.json"
        with open(output_file, 'w') as f:
            json.dump(llm_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return llm_results
        
    except Exception as e:
        print(f"Error processing with LLM: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response details: {e.response.text}")
        return None

    # Note: This function replaces what would normally be a multi-step process:
    # 1. DocumentClassifier would determine document type
    # 2. OCR would extract text
    # 3. LLMProcessor would enhance and validate
    # 4. MRZProcessor would handle specialized passport data
    # 5. DocumentProcessor would coordinate and cross-validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process document image directly with LLM')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image file')
    parser.add_argument('--doc-type', type=str, choices=['passport', 'drivers_license'],
                      default='passport', help='Type of document')
    
    args = parser.parse_args()
    
    # Verify image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
    else:
        process_image_with_llm(args.image, args.doc_type)

    # Note: This script demonstrates a simplified workflow compared to the full system:
    # - Full system: Image → DocumentClassifier → OCR → LLM → Validation → Output
    # - This script: Image → Resize → DeepSeek V3 → Parse JSON → Output
    # 
    # Key improvements:
    # - Removed dependency on OCR for initial processing
    # - Direct image analysis through DeepSeek V3
    # - Simplified workflow with fewer components
    # - Still maintains structured output format 