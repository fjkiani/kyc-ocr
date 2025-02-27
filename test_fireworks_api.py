import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

def test_fireworks_connection():
    """Test basic connection to Fireworks AI API"""
    
    # Load environment variables from .env file
    env_path = Path('.') / '.env'
    load_dotenv(env_path)
    
    # Get API key from environment
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment variables or .env file")
        return False
        
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Simple test payload
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
                "role": "user",
                "content": "Hello, can you verify this connection is working?"
            }
        ]
    }
    
    try:
        print("Testing Fireworks AI API connection...")
        print(f"\nAPI Key (first 8 chars): {api_key[:8]}...")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        if response.status_code == 200:
            print("\nSuccess! API is working.")
            print("\nResponse content:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("\nError Response:")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"\nError testing API: {str(e)}")
        return False

if __name__ == "__main__":
    test_fireworks_connection() 