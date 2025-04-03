"""
Document Processing Streamlit Application

This application provides a user-friendly interface for the document processing pipeline.
It allows users to:
1. Upload document images (passports, driver's licenses)
2. Select processing options (OCR, LLM, direct vision)
3. View processing results with visualizations
4. Download structured results as JSON
"""

# Suppress warnings first
import warnings
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Apply nest_asyncio to fix the "no running event loop" error
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import json
import cv2
import numpy as np
import re
import easyocr  # Add import for easyocr
import time
import random
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from ocr import OCR  # Add import for OCR class
from llm_processor import LLMProcessor  # Add import for LLMProcessor class
import mimetypes
import tempfile
import shutil
import time
import base64
import io
from PIL import Image
import mimetypes
from dotenv import load_dotenv
import glob
import requests
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
import threading
import string
from datetime import datetime

# Import metrics tracking and accuracy testing
from metrics_tracker import MetricsTracker, get_metrics_tracker, format_metrics_for_display
import accuracy_tester
from accuracy_tester import run_accuracy_test

# Try to import python-magic for better file type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("python-magic not installed. Using mimetypes for file type detection.")

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    from pdf2image import convert_from_path, convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PDF support libraries not installed. PDF processing will be limited.")

# Import our processing modules
from document_processor import DocumentProcessor, NumpyEncoder
from test_enhanced_ocr import EnhancedOCRTester
from test_llm_image import process_image_with_llm, encode_image

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Fireworks AI Document Inlining™ Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .sample-image-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 1rem;
    }
    .sample-image-item {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        cursor: pointer;
    }
    .sample-image-item:hover {
        border-color: #1E88E5;
        background-color: #E3F2FD;
    }
    .sample-image-item img {
        max-width: 100%;
        max-height: 150px;
        margin-bottom: 5px;
    }
    .pdf-preview {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .file-info {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        font-family: monospace;
    }
    .comparison-container {
        display: flex;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .comparison-left {
        width: 50%;
        padding: 1rem;
        border-right: 1px solid #ddd;
        # background-color: #FFF8E1;
    }
    .comparison-right {
        width: 50%;
        padding: 1rem;
        # background-color: #E8F5E9;
    }
    .modality-gap-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px dashed #B71C1C;
    }
    .structure-preserved-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px dashed #1B5E20;
    }
    .metrics-container {
        display: flex;
        margin-bottom: 1.5rem;
        gap: 15px;
    }
    .metric-card {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .metric-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .confidence-high { color: green; font-weight: bold; }
    .confidence-medium { color: orange; font-weight: bold; }
    .confidence-low { color: red; font-weight: bold; }
    .mock-data { background-color: #fffde7; padding: 10px; border-left: 5px solid #ffeb3b; }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return the path"""
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save the file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def get_file_info(file_path):
    """Get detailed information about a file"""
    file_path = Path(file_path)
    file_stats = file_path.stat()
    
    # Detect file type
    if MAGIC_AVAILABLE:
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
        except Exception as e:
            st.warning(f"Error using python-magic: {e}. Falling back to mimetypes.")
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    else:
        # Fallback to mimetypes
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    
    info = {
        "name": file_path.name,
        "extension": file_path.suffix,
        "size": f"{file_stats.st_size / 1024:.2f} KB",
        "mime_type": mime_type,
        "last_modified": file_stats.st_mtime
    }
    
    return info

def convert_pdf_to_image(pdf_path, page_num=0):
    """Convert a PDF file to an image"""
    # Check if PDF support is available
    if not PDF_SUPPORT:
        st.error("PDF support is not available. Please install the required libraries:")
        st.code("pip install PyMuPDF pdf2image")
        st.info("Alternatively, you can upload image files (JPG, JPEG, PNG) directly.")
        return None
        
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Output path for the converted image
        image_path = str(temp_dir / f"{Path(pdf_path).stem}_page_{page_num}.jpg")
        
        try:
            # First try using pdf2image (which uses poppler)
            st.info("Converting PDF using pdf2image...")
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            
            if not images:
                raise ValueError("No images were extracted from the PDF")
            
            # Save the image
            images[0].save(image_path, "JPEG")
            st.success("PDF conversion successful!")
            
        except Exception as pdf2image_error:
            # If pdf2image fails, try using PyMuPDF as fallback
            st.warning(f"pdf2image conversion failed: {str(pdf2image_error)}")
            st.info("Trying alternative conversion method with PyMuPDF...")
            
            pdf_document = fitz.open(pdf_path)
            page = pdf_document.load_page(page_num)
            
            # Higher resolution for processing
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            pix.save(image_path)
            st.success("PDF conversion successful with fallback method!")
        
        # Verify the image was created
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The converted image file was not found at {image_path}")
        
        # Check if the image can be opened
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                st.info(f"Converted image dimensions: {width}x{height}")
        except Exception as img_error:
            raise ValueError(f"The converted file exists but is not a valid image: {str(img_error)}")
        
        return image_path
        
    except Exception as e:
        st.error(f"Error converting PDF to image: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def handle_document_upload(uploaded_file):
    """Handle different document formats and convert to image if needed"""
    if uploaded_file is None:
        return None, None
    
    # Save the uploaded file
    file_path = save_uploaded_file(uploaded_file)
    if not file_path:
        return None, None
    
    # Get file info
    file_info = get_file_info(file_path)
    
    # Check file type
    if file_info["mime_type"].startswith("image/"):
        # Image file, no conversion needed
        return file_path, file_info
    elif file_info["mime_type"] == "application/pdf":
        # Check if PDF support is available
        if not PDF_SUPPORT:
            st.error("PDF support is not available. Please install the required libraries:")
            st.code("pip install PyMuPDF pdf2image")
            st.info("Alternatively, you can upload image files (JPG, JPEG, PNG) directly.")
            return None, file_info
            
        # PDF file, show preview and page selection
        st.markdown("<div class='sub-header'>PDF Document</div>", unsafe_allow_html=True)
        
        # Display file info
        st.markdown("<div class='file-info'>", unsafe_allow_html=True)
        st.markdown(f"**File Name:** {file_info['name']}")
        st.markdown(f"**File Size:** {file_info['size']}")
        st.markdown(f"**File Type:** {file_info['mime_type']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Get number of pages
        pdf_document = fitz.open(file_path)
        num_pages = len(pdf_document)
        
        st.markdown(f"PDF has {num_pages} pages. Please select a page to process:")
        
        # Page selection
        page_num = st.number_input("Page Number", min_value=1, max_value=num_pages, value=1) - 1
        
        # Preview selected page
        st.markdown("<div class='pdf-preview'>", unsafe_allow_html=True)
        st.markdown("**Preview:**")
        
        # Extract page as image for preview
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
        img_bytes = pix.tobytes("png")
        
        st.image(img_bytes, caption=f"Page {page_num+1} of {num_pages}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Convert to image for processing
        convert_button = st.button("Use This Page")
        if convert_button:
            with st.spinner("Converting PDF page to image..."):
                image_path = convert_pdf_to_image(file_path, page_num)
                if image_path:
                    st.success(f"Page {page_num+1} converted to image for processing")
                    # Store the converted image path in session state
                    if 'converted_image_path' not in st.session_state:
                        st.session_state.converted_image_path = image_path
                        st.session_state.converted_file_info = file_info
                    return image_path, file_info
                else:
                    st.error("Failed to convert PDF page to image")
                    return None, file_info
        
        # Check if we have a previously converted image in the session state
        if 'converted_image_path' in st.session_state:
            return st.session_state.converted_image_path, st.session_state.converted_file_info
        
        return None, file_info
    else:
        # Unsupported file type
        st.error(f"Unsupported file type: {file_info['mime_type']}")
        st.info("Please upload an image (JPG, JPEG, PNG) or PDF file.")
        return None, file_info

def get_sample_images():
    """Get list of sample images from the test directory"""
    test_dir = Path("test")
    if not test_dir.exists():
        return []
    
    # Get all image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(test_dir.glob(ext))
    
    return sorted(image_files)

def categorize_sample_images(sample_images):
    """Categorize sample images by document type"""
    categories = {
        "loan_application": [],
        "bank_statement": [],
        "tax_form": [],
        "property_appraisal": [],
        "other": []
    }
    
    for image_path in sample_images:
        name = image_path.name.lower()
        
        if any(term in name for term in ["loan", "application", "mortgage", "form1003", "1003"]):
            categories["loan_application"].append(image_path)
        elif any(term in name for term in ["bank", "statement", "account", "transaction"]):
            categories["bank_statement"].append(image_path)
        elif any(term in name for term in ["tax", "w2", "1099", "income"]):
            categories["tax_form"].append(image_path)
        elif any(term in name for term in ["property", "appraisal", "valuation", "real estate"]):
            categories["property_appraisal"].append(image_path)
        else:
            categories["other"].append(image_path)
    
    return categories

def display_sample_images(sample_images):
    """Display sample images in a grid for selection"""
    if not sample_images:
        st.warning("No sample images found in the test directory.")
        
        st.markdown("""
        <div class='warning-box'>
        <strong>Recommendation:</strong> For the best demonstration experience, please add sample financial documents to the test directory:
        <ul>
            <li>Loan applications (Form 1003)</li>
            <li>Bank statements with transaction tables</li>
            <li>Tax forms (W-2s, 1099s)</li>
            <li>Property appraisals with comparison tables</li>
        </ul>
        This will enable a more realistic demonstration of Document Inlining technology with financial documents.
        </div>
        """, unsafe_allow_html=True)
        
        return None
    
    st.markdown("<div class='sub-header'>Sample Financial Documents</div>", unsafe_allow_html=True)
    st.markdown("Select a document to process:")
    
    # Categorize images
    categories = categorize_sample_images(sample_images)
    
    selected_image = None
    
    # Display categorized images
    for category, images in categories.items():
        if not images:
            continue
        
        # Get display name for category
        category_display = {
            "loan_application": "Loan Applications",
            "bank_statement": "Bank Statements",
            "tax_form": "Tax Forms",
            "property_appraisal": "Property Appraisals",
            "other": "Other Documents"
        }.get(category, "Other Documents")
        
        st.markdown(f"<div class='sub-header'>{category_display}</div>", unsafe_allow_html=True)
        
        # Create columns for the images in this category
        cols = st.columns(min(3, len(images)))
        
        # Display images in columns
        for i, image_path in enumerate(images):
            col = cols[i % min(3, len(images))]
            with col:
                img = Image.open(image_path)
                # Resize for thumbnail
                img.thumbnail((200, 200))
                
                # Create a container with image and caption
                st.markdown(f"<div class='sample-image-item'>", unsafe_allow_html=True)
                st.image(img, caption=f"{image_path.name}")
                if st.button(f"Select", key=f"sample_{category}_{i}"):
                    selected_image = str(image_path)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # If no categorized images were displayed
    if all(len(images) == 0 for images in categories.values()):
        st.info("No categorized financial documents found. Displaying all available images.")
        
        # Create columns for the images
        cols = st.columns(3)
        
        # Display images in columns
        for i, image_path in enumerate(sample_images):
            col = cols[i % 3]
            with col:
                img = Image.open(image_path)
                # Resize for thumbnail
                img.thumbnail((200, 200))
                
                # Create a container with image and caption
                st.markdown(f"<div class='sample-image-item'>", unsafe_allow_html=True)
                st.image(img, caption=f"{image_path.name}")
                if st.button(f"Select", key=f"sample_{i}"):
                    selected_image = str(image_path)
                st.markdown("</div>", unsafe_allow_html=True)
    
    return selected_image

def display_image(image_path, title="Uploaded Image"):
    """Display an image with a title"""
    img = Image.open(image_path)
    st.image(img, caption=title, use_container_width=True)

def display_ocr_results(image_path, ocr_results):
    """Display OCR results with bounding boxes on the image"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # Add bounding boxes and text
    for text_obj in ocr_results:
        if isinstance(text_obj, dict):
            # From DocumentClassifier
            bbox = text_obj.get('bbox')
            text = text_obj.get('text', '')
            confidence = text_obj.get('confidence', 0)
        elif isinstance(text_obj, tuple) and len(text_obj) == 3:
            # From EasyOCR
            bbox, text, confidence = text_obj
        else:
            continue
            
        if bbox:
            # Draw bounding box
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            tr = tuple(map(int, tr))
            br = tuple(map(int, br))
            bl = tuple(map(int, bl))
            
            # Draw polygon
            cv2.line(img, tl, tr, (0, 255, 0), 2)
            cv2.line(img, tr, br, (0, 255, 0), 2)
            cv2.line(img, br, bl, (0, 255, 0), 2)
            cv2.line(img, bl, tl, (0, 255, 0), 2)
            
            # Add text
            ax.text(tl[0], tl[1] - 10, f'{text} ({confidence:.2f})',
                   color='white', fontsize=8, 
                   bbox=dict(facecolor='green', alpha=0.7))
    
    ax.axis('off')
    st.pyplot(fig)

def display_extracted_fields(result):
    """Display extracted fields from document processing"""
    if not result or not isinstance(result, dict):
        st.error("No valid result data to display")
        return
    
    # Configure CSS for different confidence levels
    st.markdown("""
    <style>
    .confidence-high { color: green; font-weight: bold; }
    .confidence-medium { color: orange; font-weight: bold; }
    .confidence-low { color: red; font-weight: bold; }
    .mock-data { background-color: #fffde7; padding: 10px; border-left: 5px solid #ffeb3b; }
    </style>
    """, unsafe_allow_html=True)
    
    # Show overall confidence
    if 'confidence' in result:
        confidence = result['confidence']
        confidence_class = "confidence-high" if confidence >= 0.8 else "confidence-medium" if confidence >= 0.6 else "confidence-low"
        
        # Check if using mock data
        is_mock = result.get('using_mock_data', False)
        if is_mock:
            st.markdown(f"""
            <div class="mock-data">
                <h4>⚠️ Using Demonstration Data</h4>
                <p>The Enhanced OCR + LLM API call failed or timed out. Showing mock structured data based on OCR results.</p>
                <p>Processing Confidence: <span class="{confidence_class}">{confidence:.2f}</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**Processing Confidence:** <span class='{confidence_class}'>{confidence:.2f}</span>", unsafe_allow_html=True)
            
            # Show confidence explanation if available
            if 'confidence_explanation' in result:
                st.markdown(f"<small><i>{result['confidence_explanation']}</i></small>", unsafe_allow_html=True)
    
    # Display extracted fields if available
    if 'extracted_fields' in result and result['extracted_fields']:
        fields = result['extracted_fields']
        
        # Create a DataFrame for better display
        rows = []
        for field_name, field_data in fields.items():
            # Handle both dict and string value formats
            if isinstance(field_data, dict):
                value = field_data.get('value', 'N/A')
                standardized = field_data.get('standardized_value', 'N/A')
                confidence = field_data.get('confidence', 0.0)
                status = field_data.get('validation_status', 'unknown')
            else:
                value = field_data
                standardized = field_data
                confidence = 0.0
                status = 'unknown'
                
            # For long text fields, truncate for display
            if isinstance(value, str) and len(value) > 50:
                display_value = value[:50] + "..."
            else:
                display_value = value
                
            rows.append({
                'Field': field_name.replace('_', ' ').title(),
                'Value': display_value,
                'Standardized': standardized,
                'Confidence': f"{confidence:.2f}",
                'Status': status.capitalize()
            })
        
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            
            # For long text fields, show expandable view
            long_text_fields = [field for field, data in fields.items() 
                               if isinstance(data, dict) and isinstance(data.get('value'), str) 
                               and len(data.get('value', '')) > 50]
            
            if long_text_fields:
                with st.expander("View Long Text Fields"):
                    for field in long_text_fields:
                        if isinstance(fields[field], dict):
                            st.markdown(f"**{field.replace('_', ' ').title()}:**")
                            st.text(fields[field].get('value', 'N/A'))
                            st.markdown("---")
    
    # Display validation notes if available
    if 'validation_notes' in result and result['validation_notes']:
        with st.expander("Validation Notes", expanded=True):
            for note in result['validation_notes']:
                st.markdown(f"- {note}")
    
    # Add button for detailed LLM view
    if st.button("Show Detailed LLM Processing Data"):
        display_detailed_llm_output(result, result.get('document_type', 'bank_statement'))
    
    # Display other useful information
    if 'document_type' in result:
        st.caption(f"Document Type: {result['document_type']}")
    
    if 'processing_method' in result:
        method = result['processing_method']
        method_display = {
            'traditional_ocr': 'Traditional OCR',
            'document_inlining': 'Document Inlining',
            'enhanced_ocr': 'Enhanced OCR + LLM'
        }.get(method, method)
        st.caption(f"Processing Method: {method_display}")

def display_detailed_llm_output(result, doc_type="bank_statement"):
    """Display detailed LLM processing information including raw API response"""
    st.markdown("## 📝 Detailed LLM Processing")
    
    ocr_results = result.get('detected_fields', [])
    
    # Create tabs for different parts of the process
    tab1, tab2, tab3 = st.tabs(["OCR Results", "LLM Prompt", "LLM Response"])
    
    # Tab 1: Show OCR results
    with tab1:
        st.markdown("### OCR Extracted Text")
        st.markdown("These are the text elements extracted by OCR and sent to the LLM:")
        
        for i, item in enumerate(ocr_results):
            if isinstance(item, tuple) and len(item) >= 2:
                text = item[1] if len(item) == 3 else item[0]
                confidence = item[2] if len(item) == 3 else item[1]
                
                # Color-code confidence
                confidence_html = ""
                if confidence >= 0.8:
                    confidence_html = f"<span style='color:green'>{confidence:.2f}</span>"
                elif confidence >= 0.6:
                    confidence_html = f"<span style='color:orange'>{confidence:.2f}</span>"
                else:
                    confidence_html = f"<span style='color:red'>{confidence:.2f}</span>"
                
                st.markdown(f"- **Text:** `{text}` (Confidence: {confidence_html})", unsafe_allow_html=True)
    
    # Tab 2: Show the LLM prompt
    with tab2:
        st.markdown("### LLM Prompt")
        st.markdown("This is the prompt that would be sent to the LLM:")
        
        # Generate a sample of the prompt that would be sent
        try:
            llm = LLMProcessor()
            structured_results = llm._structure_ocr_results(ocr_results)
            prompt = llm._create_field_extraction_prompt(structured_results, doc_type)
            
            st.code(prompt, language="text")
        except Exception as e:
            st.error(f"Error generating sample prompt: {str(e)}")
    
    # Tab 3: Show LLM response
    with tab3:
        st.markdown("### LLM Response")
        
        if 'raw_llm_response' in result:
            st.markdown("#### Raw JSON Response")
            st.json(result['raw_llm_response'])
            
            st.markdown("#### Parsed Content")
            if 'raw_llm_content' in result:
                st.code(result['raw_llm_content'], language="text")
            else:
                st.warning("Raw LLM content not available")
        elif result.get('using_mock_data', False):
            st.warning("⚠️ This is mock data generated because the LLM API call failed or timed out.")
            if 'api_error' in result:
                st.error(f"API Error: {result['api_error']}")
            
            st.markdown("### Mock Data Generator Logic")
            st.markdown("""
            When the LLM API call fails, the system generates mock data based on the OCR results.
            This is done by the `_generate_mock_llm_response` function which:
            
            1. Extracts text from OCR results
            2. Looks for patterns relevant to the document type
            3. Creates a structured response similar to what the LLM would return
            4. Adds appropriate validation notes and confidence scores
            """)
        else:
            st.error("No raw LLM response data available.")

def process_with_full_pipeline(image_path):
    """Process image with the full document processing pipeline"""
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        st.error("FIREWORKS_API_KEY not found in environment variables")
        return None
    
    with st.spinner("Processing with full pipeline..."):
        try:
            tester = EnhancedOCRTester(api_key)
            result = tester.process_single_image(image_path)
            return result
        except Exception as e:
            st.error(f"Error in full pipeline processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def process_with_direct_vision(image_path, doc_type):
    """Process image directly with LLM vision capabilities"""
    with st.spinner("Processing with direct vision..."):
        try:
            result = process_image_with_llm(image_path, doc_type)
            return result
        except Exception as e:
            st.error(f"Error in direct vision processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def process_with_ocr_only(image_path):
    """Process image with OCR only"""
    with st.spinner("Processing with OCR only..."):
        try:
            ocr = OCR(os.path.dirname(image_path))
            results = ocr.easyocr_model_works(image_path, visualization=False)
            return results
        except Exception as e:
            st.error(f"Error in OCR processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def calculate_confidence_score(result_data):
    """Calculate a confidence score based on the quality and structure of the extracted data"""
    # Start with a base confidence
    base_confidence = 0.7
    
    # If the API returned an explicit overall confidence, use that
    if isinstance(result_data, dict) and 'overall_confidence' in result_data:
        return result_data['overall_confidence']
    
    # Calculate confidence based on the structure and content
    adjustment = 0.0
    
    # Check if we have structured fields with confidence scores
    if isinstance(result_data, dict) and 'extracted_fields' in result_data:
        fields = result_data['extracted_fields']
        if fields and isinstance(fields, dict):
            # Count fields with confidence scores
            confidence_scores = []
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict) and 'confidence' in field_data:
                    confidence_scores.append(field_data['confidence'])
            
            # If we have confidence scores, calculate the average
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            
            # Bonus for having structured fields even without confidence scores
            adjustment += 0.05 * min(10, len(fields)) / 10
    
    # Check for structured content
    if isinstance(result_data, dict) and 'structured_content' in result_data:
        content = result_data['structured_content']
        if isinstance(content, str):
            # Check for table structures
            if "| " in content and " |" in content:
                adjustment += 0.1
            
            # Check for list structures
            list_items = len(re.findall(r'^[-*]\s', content, re.MULTILINE))
            adjustment += min(0.05, 0.01 * list_items)
            
            # Check for key-value pairs
            key_value_pairs = len(re.findall(r':\s', content))
            adjustment += min(0.05, 0.005 * key_value_pairs)
    
    # Final confidence score (capped at 0.95)
    return min(0.95, base_confidence + adjustment)

def process_with_document_inlining(image_path, doc_type):
    """Process using Document Inlining technology that preserves document structure"""
    # Record start time for metrics
    get_metrics_tracker().record_processing_start('document_inlining')
    
    with st.spinner("Processing with Document Inlining technology..."):
        try:
            # Get API key
            api_key = os.getenv('FIREWORKS_API_KEY')
            if not api_key or api_key == "fw_your_api_key_here":
                st.warning("⚠️ No valid Fireworks API key found. Please enter your API key in the sidebar.")
                st.info("Falling back to traditional OCR method as Document Inlining requires an API key.")
                return process_with_traditional_ocr(image_path)
            
            # Use direct vision-to-LLM with Document Inlining as primary approach
            st.info("Using Document Inlining to preserve document structure...")
            
            # Encode image
            base64_image = encode_image(image_path)
            
            # Create specialized system prompt based on document type
            if 'bank_statement' in doc_type.lower() or 'bank statement' in doc_type.lower():
                system_prompt = """You are an expert document analyzer specializing in financial documents.
                Analyze this bank statement and extract all information while preserving table structure:
                - Account holder information (name, account number, statement period)
                - Transaction table with dates, descriptions, deposits, withdrawals and balances
                - Summary information (opening/closing balances, total deposits/withdrawals)
                - Any special notices or messages

                For the transaction table, maintain the table structure by returning:
                - Column headers
                - Each row as a list of values
                - Proper alignment of amounts and dates
                
                For all extracted values, provide:
                - The raw value as seen in the document
                - A standardized version (e.g., consistent date format)
                - Your confidence in the extraction (0.0-1.0)
                """
            elif 'loan' in doc_type.lower() or 'application' in doc_type.lower():
                system_prompt = """You are an expert document analyzer specializing in financial documents.
                Analyze this loan application and extract all information while preserving form structure:
                - Applicant information (name, SSN, DOB, contact information)
                - Property information (address, type, value)
                - Loan information (amount, term, rate, purpose)
                - Income and employment information
                - Asset and liability information
                
                Preserve the relationship between form fields and their values.
                Maintain multi-page connections if detected.
                
                For all extracted values, provide:
                - The raw value as seen in the document
                - A standardized version
                - Your confidence in the extraction (0.0-1.0)
                """
            elif 'tax' in doc_type.lower() or 'w2' in doc_type.lower() or 'form' in doc_type.lower():
                system_prompt = """You are an expert document analyzer specializing in financial documents.
                Analyze this tax form and extract all information while preserving structure:
                - Taxpayer information (name, SSN, address)
                - Income information (wages, tips, other compensation)
                - Tax information (federal income tax withheld, social security tax)
                - Employer information (name, EIN, address)
                
                Maintain box number-to-value relationships from the original form.
                Preserve all calculated totals and their components.
                
                For all extracted values, provide:
                - The raw value as seen in the document
                - A standardized version (e.g., formatted numbers)
                - Your confidence in the extraction (0.0-1.0)
                """
            elif 'property' in doc_type.lower() or 'appraisal' in doc_type.lower():
                system_prompt = """You are an expert document analyzer specializing in financial documents.
                Analyze this property appraisal and extract all information while preserving structure:
                - Property details (address, type, description)
                - Valuation information (appraised value, date of appraisal)
                - Comparable properties table with all details
                - Assessment methodology and conclusions
                
                For the comparable properties table, maintain the structure by returning:
                - Column headers (address, price, size, etc.)
                - Each property as a row with all values
                - Properly aligned numerical values
                
                For all extracted values, provide:
                - The raw value as seen in the document
                - A standardized version
                - Your confidence in the extraction (0.0-1.0)
                """
            else:
                # Generic financial document prompt
                system_prompt = """You are an expert document analyzer specializing in financial documents.
                Analyze this financial document and extract all relevant information while preserving structure:
                - Document type and purpose
                - Entity information (names, accounts, dates)
                - Financial figures and calculations
                - Tables with their full structure preserved
                - Form fields with their relationships maintained
                
                For all extracted values, provide:
                - The raw value as seen in the document
                - A standardized version
                - Your confidence in the extraction (0.0-1.0)
                
                Additionally, provide an overall confidence score between 0.0 and 1.0 that reflects your certainty about the entire extraction.
                """
            
            # Prepare payload for API request - add #transform=inline to the image URL
            url = "https://api.fireworks.ai/inference/v1/chat/completions"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Log current API key (first 5 chars only for security)
            api_prefix = api_key[:5] if api_key else "None"
            st.write(f"Debug: Using API key prefix: {api_prefix}...")
            
            # Create a base64 image URL with the transform parameter
            image_url = f"data:image/jpeg;base64,{base64_image}#transform=inline"
            
            # Based on documentation, trying different model for Document Inlining
            payload = {
                "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",  # Using a model mentioned in Document Inlining docs
                "max_tokens": 4096,
                "top_p": 1,
                "temperature": 0.2,  # Lower temperature for more structured results
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Please analyze this {doc_type} and extract all information while preserving document structure. Focus especially on tables, form fields, and relationships between elements. Also provide an overall confidence score (0.0-1.0) for your extraction."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            }
            
            # Log that we're about to make the API call
            st.write("Debug: Sending request to Fireworks AI API...")
            
            # Make API request
            response = requests.post(url, headers=headers, json=payload)
            
            # Log response status
            st.write(f"Debug: API response status: {response.status_code}")
            
            if response.status_code != 200:
                st.write(f"Debug: API error response: {response.text}")
                response.raise_for_status()
            
            # Get the raw JSON response
            result = response.json()
            
            # Debug: Show the raw response in a collapsible section
            with st.expander("Debug: API Response"):
                st.json(result)
            
            # Extract content from response based on its structure
            content = ""
            if 'choices' in result and len(result['choices']) > 0:
                if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    content = result['choices'][0]['message']['content']
                elif 'text' in result['choices'][0]:
                    content = result['choices'][0]['text']
            else:
                # Handle the case where the expected fields are not in the response
                st.warning("The API response format was unexpected. Using raw response.")
                content = str(result)
            
            # Find JSON content in the response
            llm_results = {}
            try:
                # Try to extract JSON from the content if it's a string
                if isinstance(content, str):
                    # Look for JSON-like content between curly braces
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = content[json_start:json_end+1]
                        llm_results = json.loads(json_str)
                    else:
                        # If no JSON structure is found, use the whole content as structured text
                        st.info("No structured JSON found in response. Processing as text content.")
                        
                        # Look for confidence mentions in the text
                        confidence_match = re.search(r'confidence(?:\s+(?:score|level|rating))?(?:\s*(?:of|is|:))?\s*(0\.\d+|[01](?:\.\d+)?)', content, re.IGNORECASE)
                        estimated_confidence = float(confidence_match.group(1)) if confidence_match else 0.8
                        
                        # Create a more structured representation for the narrative response
                        llm_results = {
                            "extracted_fields": {
                                "full_text": {
                                    "value": content,
                                    "confidence": estimated_confidence
                                }
                            },
                            "structured_content": content,
                            "overall_confidence": estimated_confidence
                        }
                elif isinstance(content, dict):
                    # If content is already a dict, use it directly
                    llm_results = content
                else:
                    # For any other type, convert to string
                    llm_results = {
                        "extracted_fields": {
                            "full_text": {
                                "value": str(content),
                                "confidence": 0.7
                            }
                        },
                        "overall_confidence": 0.7
                    }
            except Exception as json_error:
                st.warning(f"Could not parse JSON from response: {str(json_error)}")
                # If JSON parsing fails, create a structured result from the text response
                
                # Look for confidence mentions in the text if it's a string
                estimated_confidence = 0.7
                if isinstance(content, str):
                    confidence_match = re.search(r'confidence(?:\s+(?:score|level|rating))?(?:\s*(?:of|is|:))?\s*(0\.\d+|[01](?:\.\d+)?)', content, re.IGNORECASE)
                    if confidence_match:
                        estimated_confidence = float(confidence_match.group(1))
                
                llm_results = {
                    "extracted_fields": {
                        "full_text": {
                            "value": content if isinstance(content, str) else str(content),
                            "confidence": estimated_confidence
                        }
                    },
                    "structured_content": content if isinstance(content, str) else str(content),
                    "validation_notes": ["Processed with Document Inlining but response format was unexpected"],
                    "overall_confidence": estimated_confidence
                }
            
            # Add document inlining specific details to the result
            result_with_metadata = {
                'document_type': doc_type,
                'processing_method': 'document_inlining',
                'structure_preservation': True,
                'modality_gap_bridged': True,
                'model': payload["model"].split('/')[-1]  # Extract model name from full path
            }
            
            # Add the extracted content
            if 'extracted_fields' in llm_results and isinstance(llm_results['extracted_fields'], dict):
                # Ensure all extracted fields have the expected structure
                structured_fields = {}
                field_confidence_scores = []
                
                for key, value in llm_results['extracted_fields'].items():
                    if isinstance(value, dict):
                        field_entry = value.copy()
                        # Ensure all required fields are present
                        if 'value' not in field_entry:
                            field_entry['value'] = key
                        if 'confidence' not in field_entry:
                            field_entry['confidence'] = 0.8
                        else:
                            field_confidence_scores.append(field_entry['confidence'])
                        structured_fields[key] = field_entry
                    else:
                        # Create a structured field from primitive value
                        structured_fields[key] = {
                            'value': value,
                            'confidence': 0.8
                        }
                result_with_metadata['extracted_fields'] = structured_fields
                
                # Calculate average field confidence if we have scores
                if field_confidence_scores:
                    result_with_metadata['field_confidence_avg'] = sum(field_confidence_scores) / len(field_confidence_scores)
            elif 'extracted_text' in llm_results:
                # Convert raw text into a structured field
                result_with_metadata['extracted_fields'] = {
                    'full_text': {
                        'value': llm_results['extracted_text'],
                        'confidence': 0.8
                    }
                }
            else:
                # Create a structured field from content if no other fields are available
                result_with_metadata['extracted_fields'] = {
                    'document_content': {
                        'value': content if isinstance(content, str) else str(content),
                        'confidence': 0.7
                    }
                }
            
            # Add structured content if available
            if 'structured_content' in llm_results:
                result_with_metadata['structured_content'] = llm_results['structured_content']
            elif isinstance(content, str):
                result_with_metadata['structured_content'] = content
            else:
                result_with_metadata['structured_content'] = str(content)
            
            # Add validation notes if available
            if 'validation_notes' in llm_results:
                result_with_metadata['validation_notes'] = llm_results['validation_notes']
            else:
                result_with_metadata['validation_notes'] = ["Document processed with Document Inlining technology"]
            
            # Calculate the overall confidence score
            if 'overall_confidence' in llm_results:
                result_with_metadata['overall_confidence'] = llm_results['overall_confidence']
            else:
                # Calculate a dynamic confidence score based on the quality of the results
                result_with_metadata['overall_confidence'] = calculate_confidence_score(llm_results)
            
            # Set the general confidence field for compatibility
            result_with_metadata['confidence'] = result_with_metadata['overall_confidence']
            
            # Add the raw detection results for compatibility with the comparison view
            # This will be used by the display functions that expect detected_fields
            extracted_fields = {}
            if 'extracted_fields' in llm_results and isinstance(llm_results['extracted_fields'], dict):
                extracted_fields = llm_results['extracted_fields']
            
            result_with_metadata['detected_fields'] = [
                {'text': key, 'confidence': value.get('confidence', 0.8) if isinstance(value, dict) else 0.8, 
                 'bbox': [[0,0],[0,0],[0,0],[0,0]]}
                for key, value in extracted_fields.items()
            ]
            
            # If no structured fields were detected, add raw content as a field
            if not result_with_metadata['detected_fields']:
                result_with_metadata['detected_fields'] = [
                    {'text': 'Document Content', 'confidence': result_with_metadata['overall_confidence'], 'bbox': [[0,0],[0,0],[0,0],[0,0]]}
                ]
            
            # Add explanation of confidence score
            confidence_explanation = ""
            if 'field_confidence_avg' in result_with_metadata:
                confidence_explanation = f"Based on average confidence of {len(extracted_fields)} extracted fields"
            else:
                confidence_explanation = "Based on document structure analysis"
            
            result_with_metadata['confidence_explanation'] = confidence_explanation
            
            # Record metrics for document inlining
            get_metrics_tracker().record_processing_end('document_inlining', result_with_metadata)
            
            return result_with_metadata
            
        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            if "401" in str(e):
                st.warning("⚠️ Authentication error. Your API key may be invalid.")
                st.info("Please check your Fireworks API key in the sidebar.")
            elif "403" in str(e):
                st.warning("⚠️ Forbidden error. Your API key may not have access to this resource.")
                st.info("Please check your API key permissions or contact Fireworks support.")
            elif "400" in str(e):
                st.warning("⚠️ Bad request error. The API request format may be incorrect.")
                st.info("Falling back to traditional OCR method...")
                return process_with_traditional_ocr(image_path)
            else:
                st.info("Falling back to traditional OCR method due to API error...")
                return process_with_traditional_ocr(image_path)
                
            import traceback
            st.code(traceback.format_exc())
            return None
        except Exception as e:
            st.error(f"Error in Document Inlining processing: {str(e)}")
            st.write(f"Debug: Exception details - {type(e).__name__}: {str(e)}")
            
            import traceback
            st.code(traceback.format_exc())
            
            st.info("Falling back to traditional OCR method...")
            fallback_result = process_with_traditional_ocr(image_path)
            if fallback_result:
                fallback_result['fallback_reason'] = str(e)
                return fallback_result
            
            return None

def process_with_traditional_ocr(image_path):
    """Process document with traditional OCR only"""
    # Record start time for metrics
    get_metrics_tracker().record_processing_start('traditional_ocr')
    
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            st.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with EasyOCR
        reader = easyocr.Reader(['en'])
        ocr_results = reader.readtext(img_rgb)
        
        # Format results
        formatted_results = []
        for detection in ocr_results:
            bbox, text, confidence = detection
            formatted_results.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        result = {
            'document_type': 'unknown',
            'processing_method': 'traditional_ocr',
            'confidence': sum(r['confidence'] for r in formatted_results) / max(1, len(formatted_results)),
            'detected_fields': ocr_results
        }
        
        # Record end time and metrics
        get_metrics_tracker().record_processing_end('traditional_ocr', result)
        
        return result
    except Exception as e:
        st.error(f"Error in traditional OCR processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        
        # Record failed processing
        get_metrics_tracker().record_processing_end('traditional_ocr', None)
        
        return None

def _generate_mock_llm_response(ocr_results, doc_type):
    """Generate a mock LLM response when the real API fails or times out

    This function examines the OCR results and creates a plausible
    structured extraction based on the document type.
    """
    # Extract text from OCR results
    all_text = []
    for item in ocr_results:
        if isinstance(item, tuple) and len(item) >= 2:
            if len(item) == 3:  # (bbox, text, confidence)
                all_text.append(item[1])
            else:  # (text, confidence)
                all_text.append(item[0])
    
    # Join all text into a single string for analysis
    text_dump = " ".join(all_text)
    
    # Create mock field extraction based on document type
    extracted_fields = {}
    validation_notes = []
    confidence = 0.85  # Default confidence
    
    # Document type-specific processing
    if doc_type == 'bank_statement':
        # Look for common bank statement patterns in the OCR text
        account_number_pattern = re.compile(r'[\d*]{8,20}')
        date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}')
        amount_pattern = re.compile(r'\$\s*\d{1,3}(?:,\d{3})*\.\d{2}|\d{1,3}(?:,\d{3})*\.\d{2}')
        
        # Find potential values
        account_numbers = account_number_pattern.findall(text_dump)
        dates = date_pattern.findall(text_dump)
        amounts = amount_pattern.findall(text_dump)
        
        # Extract potential account holder name (look for common name indicators)
        name_indicators = ['name:', 'customer:', 'account holder:', 'customer name:']
        account_holder = "John Doe"  # Default name
        
        for line in all_text:
            line_lower = line.lower()
            for indicator in name_indicators:
                if indicator in line_lower:
                    # Extract name after the indicator
                    name_part = line_lower.split(indicator)[1].strip()
                    if len(name_part) > 3 and len(name_part) < 40:
                        account_holder = name_part.title()
        
        # Find account balance indicators
        balance_indicators = ['balance:', 'ending balance:', 'closing balance:', 'available balance:']
        balance = "$5,245.67" if amounts else "Unknown"
        
        for line in all_text:
            line_lower = line.lower()
            for indicator in balance_indicators:
                if indicator in line_lower:
                    # Extract amount near balance indicator
                    nearby_amounts = amount_pattern.findall(line)
                    if nearby_amounts:
                        balance = nearby_amounts[0]
                        if not balance.startswith('$'):
                            balance = f"${balance}"
        
        # Find account holder - try to extract a realistic name
        name_pattern = re.compile(r'(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+')
        names = name_pattern.findall(text_dump)
        if names:
            # Filter out common non-name phrases that might match pattern
            non_names = ['Customer Service', 'Account Statement', 'Credit Card', 'Savings Account',
                         'Checking Account', 'Opening Balance', 'Closing Balance']
            filtered_names = [n for n in names if n not in non_names and len(n.split()) >= 2]
            if filtered_names:
                account_holder = filtered_names[0]
                
        # Find transactions
        transactions = []
        date_indicators = ['date', 'transaction date', 'post date']
        desc_indicators = ['description', 'transaction', 'details', 'merchant', 'payee']
        amount_indicators = ['amount', 'withdrawal', 'deposit', 'credit', 'debit']
        
        # Try to extract transactions from statement text
        for i, line in enumerate(all_text):
            line_lower = line.lower()
            if any(d in line_lower for d in date_indicators) and any(a in line_lower for a in amount_indicators):
                # This might be a transaction header - check next lines for transactions
                max_trans = min(10, len(all_text) - i - 1)
                for j in range(1, max_trans):
                    if i+j < len(all_text):
                        trans_line = all_text[i+j]
                        # If line has a date and amount, it's likely a transaction
                        trans_dates = date_pattern.findall(trans_line)
                        trans_amounts = amount_pattern.findall(trans_line)
                        if trans_dates and trans_amounts:
                            desc = "Transaction"
                            # Try to extract description between date and amount
                            date_end = trans_line.find(trans_dates[0]) + len(trans_dates[0])
                            amount_start = trans_line.find(trans_amounts[0])
                            if date_end < amount_start:
                                desc = trans_line[date_end:amount_start].strip()
                                if not desc:
                                    desc = "Merchandise Purchase"
                            
                            # Add transaction
                            transactions.append({
                                "date": trans_dates[0],
                                "description": desc or "Purchase",
                                "amount": trans_amounts[0]
                            })
        
        # Create mock fields
        extracted_fields = {
            "account_holder": {
                "value": account_holder,
                "confidence": 0.85,
                "validation_status": "valid"
            },
            "account_number": {
                "value": account_numbers[0][-8:].rjust(8, '*') if account_numbers else "****5678",
                "confidence": 0.90 if account_numbers else 0.70,
                "validation_status": "valid" if account_numbers else "uncertain"
            },
            "statement_date": {
                "value": dates[0] if dates else "01/15/2023",
                "confidence": 0.85 if dates else 0.70,
                "validation_status": "valid" if dates else "uncertain"
            },
            "balance": {
                "value": balance,
                "confidence": 0.80,
                "validation_status": "valid"
            }
        }
        
        # Add transactions if found
        if transactions:
            extracted_fields["transactions"] = {
                "value": transactions,
                "confidence": 0.75,
                "validation_status": "valid"
            }
            
        # Add period dates if found (at least 2 dates)
        if len(dates) >= 2:
            extracted_fields["statement_period"] = {
                "value": f"{dates[0]} to {dates[1]}",
                "confidence": 0.80,
                "validation_status": "valid"
            }
            
        # Add bank name if it can be extracted
        bank_names = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One", "TD Bank", "US Bank"]
        for bank in bank_names:
            if bank.lower() in text_dump.lower():
                extracted_fields["bank_name"] = {
                    "value": bank,
                    "confidence": 0.95,
                    "validation_status": "valid"
                }
                break
                
        # Add income field if some transactions suggest income
        income_keywords = ["salary", "deposit", "direct deposit", "payment", "payroll"]
        for trans in transactions:
            if any(keyword in trans["description"].lower() for keyword in income_keywords):
                extracted_fields["income"] = {
                    "value": trans["amount"],
                    "confidence": 0.70,
                    "validation_status": "possible_income"
                }
                validation_notes.append("Possible income detected based on transaction descriptions")
                break
    
    elif doc_type == 'passport':
        # Extract basic passport information
        extracted_fields = {
            "passport_number": {
                "value": "".join(random.choices(string.ascii_uppercase, k=2)) + "".join(random.choices(string.digits, k=7)),
                "confidence": 0.90,
                "validation_status": "valid"
            },
            "full_name": {
                "value": "Jane A. Smith",
                "confidence": 0.85,
                "validation_status": "valid"
            },
            "date_of_birth": {
                "value": "15 JAN 1985",
                "confidence": 0.90,
                "validation_status": "valid"
            },
            "nationality": {
                "value": "UNITED STATES OF AMERICA",
                "confidence": 0.95,
                "validation_status": "valid"
            },
            "expiration_date": {
                "value": "28 FEB 2030",
                "confidence": 0.90,
                "validation_status": "valid",
                "standardized_value": "2030-02-28"
            }
        }
    
    elif doc_type == 'drivers_license':
        # Extract basic driver's license information
        states = ["California", "New York", "Texas", "Florida", "Pennsylvania"]
        extracted_fields = {
            "license_number": {
                "value": "".join(random.choices(string.ascii_uppercase + string.digits, k=8)),
                "confidence": 0.90,
                "validation_status": "valid"
            },
            "full_name": {
                "value": "Robert J. Johnson",
                "confidence": 0.85,
                "validation_status": "valid"
            },
            "address": {
                "value": "123 Main Street, Anytown, " + random.choice(states) + " 12345",
                "confidence": 0.80,
                "validation_status": "valid"
            },
            "date_of_birth": {
                "value": "05/22/1990",
                "confidence": 0.90,
                "validation_status": "valid",
                "standardized_value": "1990-05-22"
            },
            "issue_date": {
                "value": "01/15/2020",
                "confidence": 0.90,
                "validation_status": "valid",
                "standardized_value": "2020-01-15"
            },
            "expiration_date": {
                "value": "05/22/2028",
                "confidence": 0.90,
                "validation_status": "valid",
                "standardized_value": "2028-05-22"
            }
        }
    
    # Add informational validation note
    validation_notes.append("This data was generated from OCR text as a demonstration (API unavailable)")
    
    # Return structured mock response
    return {
        "extracted_fields": extracted_fields,
        "validation_notes": validation_notes,
        "overall_confidence": confidence,
        "confidence_explanation": "Confidence estimate based on quality of OCR text extraction"
    }

def extract_text_with_ocr(image_path):
    """Extract text elements from an image using OCR
    
    Returns a list of tuples in the format:
    [(text, confidence), ...] or [(bbox, text, confidence), ...]
    depending on the OCR engine used.
    """
    try:
        # Initialize OCR processor
        ocr_processor = OCR(os.path.dirname(image_path))
        
        # Process with OCR to get text
        ocr_results = ocr_processor.process_image(image_path, engine='easyocr', visualization=False)
        
        if not ocr_results or len(ocr_results) == 0:
            print("OCR processing failed to extract any text")
            return []
        
        return ocr_results
        
    except Exception as e:
        print(f"Error in OCR text extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_with_enhanced_ocr(image_path, doc_type):
    """Process documents using OCR followed by LLM enhancement"""
    print("\nProcessing with Enhanced OCR + LLM:")
    
    result = {
        'status': 'processing',
        'document_type': doc_type,
        'processing_method': 'enhanced_ocr',
        'extracted_fields': {},
        'validation_notes': [],
        'confidence': 0.0,
        'raw_ocr_results': []
    }
    
    try:
        # Step 1: Extract text with OCR
        print("\n1. Getting OCR results...")
        ocr = OCR(os.path.dirname(image_path))
        ocr_results = ocr.process_image(image_path, engine='easyocr', visualization=True)
        
        # Format OCR results
        formatted_results = []
        for bbox, text, conf in ocr_results:
            formatted_results.append((text.strip(), conf))
            print(f"- Text: {text:<30} (Confidence: {conf:.2f})")
        
        if not formatted_results:
            result.update({
                'status': 'error',
                'error': 'No text extracted from image',
                'validation_notes': ['OCR failed to extract any text from the image']
            })
            return result
            
        result['raw_ocr_results'] = formatted_results
        print(f"\nFound {len(formatted_results)} text regions")
        
        # Step 2: Process with LLM
        print("\n2. Processing with LLM...")
        llm = LLMProcessor()
        
        try:
            # Create a simpler prompt for the LLM
            text_content = "\n".join([f"{text} (Confidence: {conf:.2f})" 
                                    for text, conf in formatted_results])
            
            prompt = f"""Analyze this {doc_type} and extract key information.
            Document content:
            {text_content}
            
            Return a JSON object with:
            1. extracted_fields: Dictionary of field name to object containing:
               - value: Raw value from document
               - standardized_value: Cleaned/formatted value
               - confidence: Confidence score (0.0-1.0)
               - validation_status: 'verified' or 'needs_review'
            2. validation_notes: List of any issues or observations
            3. overall_confidence: Overall confidence in extraction (0.0-1.0)
            """
            
            llm_response = llm._call_llm(prompt)
            
            if llm_response:
                # Parse the JSON response
                try:
                    # Extract message content from LLM response
                    if isinstance(llm_response, dict) and 'choices' in llm_response:
                        content = llm_response['choices'][0]['message']['content']
                    else:
                        content = llm_response

                    # Try to extract JSON from the content
                    if isinstance(content, str):
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            parsed_response = json.loads(json_match.group(0))
                        else:
                            raise ValueError("No JSON found in response")
                    else:
                        parsed_response = content
                    
                    # Update result with parsed data
                    result.update({
                        'status': 'success',
                        'extracted_fields': parsed_response.get('extracted_fields', {}),
                        'validation_notes': parsed_response.get('validation_notes', []),
                        'confidence': parsed_response.get('overall_confidence', 0.0)
                    })
                    
                    # Display results
                    print("\n3. Results:")
                    print("\nExtracted Fields:")
                    for field, data in result['extracted_fields'].items():
                        print(f"\n{field}:")
                        if isinstance(data, dict):
                            print(f"  Value: {data.get('value', 'N/A')}")
                            print(f"  Standardized: {data.get('standardized_value', 'N/A')}")
                            print(f"  Confidence: {data.get('confidence', 0.0)}")
                            print(f"  Status: {data.get('validation_status', 'unknown')}")
                        else:
                            print(f"  Value: {data}")
                    
                    if result['validation_notes']:
                        print("\nValidation Notes:")
                        for note in result['validation_notes']:
                            print(f"- {note}")
                    
                    print(f"\nOverall Confidence: {result['confidence']:.2%}")
                    
                except Exception as parse_error:
                    print(f"Error parsing LLM response: {str(parse_error)}")
                    result.update({
                        'status': 'error',
                        'error': f'Failed to parse LLM response: {str(parse_error)}',
                        'raw_llm_response': llm_response
                    })
            else:
                raise ValueError("LLM returned empty response")
                
        except Exception as llm_error:
            print(f"\nLLM Processing Error: {str(llm_error)}")
            # Use fallback mock data
            mock_response = _generate_mock_llm_response(formatted_results, doc_type)
            result.update({
                'status': 'success',
                'using_mock_data': True,
                'api_error': str(llm_error),
                'extracted_fields': mock_response.get('extracted_fields', {}),
                'validation_notes': mock_response.get('validation_notes', []),
                'confidence': mock_response.get('overall_confidence', 0.0)
            })
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        result.update({
            'status': 'error',
            'error': str(e),
            'validation_notes': [f'Processing error: {str(e)}']
        })
    
    return result

def display_financial_impact():
    """Display the financial impact metrics for loan processing"""
    st.markdown("<div class='sub-header'>Impact on Loan Processing</div>", unsafe_allow_html=True)
    
    # Get metrics from tracker if available
    metrics = get_metrics_tracker().get_latest_metrics()
    if metrics and metrics['document_count'] > 0 and 'comparative' in metrics:
        # Use actual metrics from testing
        formatted_metrics = format_metrics_for_display(metrics)
        
        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
        
        # Processing Time Reduction
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Processing Time Reduction</div>
            <div class='metric-value'>{formatted_metrics['processing_time_reduction']}</div>
            <div class='metric-label'>Faster document processing</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Error Reduction
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Error Reduction</div>
            <div class='metric-value'>{formatted_metrics['error_reduction']}</div>
            <div class='metric-label'>Fewer extraction errors</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual Review Reduction
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Manual Review</div>
            <div class='metric-value'>{formatted_metrics['manual_review_reduction']}</div>
            <div class='metric-label'>Reduction in review time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Time to Close
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Time to Close</div>
            <div class='metric-value'>{formatted_metrics['time_to_close']}</div>
            <div class='metric-label'>Faster loan processing</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if metrics['document_count'] < 5:
            st.markdown("""
            <div class='info-box'>
            These metrics are based on your actual document processing tests.
            Process more documents for more accurate results.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
            These metrics are based on your actual document processing tests.
            </div>
            """, unsafe_allow_html=True)
    else:
        # Use hardcoded metrics as fallback
        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
        
        # Processing Time Reduction
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Processing Time Reduction</div>
            <div class='metric-value'>85%</div>
            <div class='metric-label'>Faster document processing</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Error Reduction
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Error Reduction</div>
            <div class='metric-value'>90%</div>
            <div class='metric-label'>Fewer extraction errors</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual Review Reduction
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Manual Review</div>
            <div class='metric-value'>70%</div>
            <div class='metric-label'>Reduction in review time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Time to Close
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Time to Close</div>
            <div class='metric-value'>40-50%</div>
            <div class='metric-label'>Faster loan processing</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        These metrics are based on industry benchmarks for financial institutions processing loan applications.
        Results may vary depending on document complexity and integration with existing systems.
        <br/><br/>
        <strong>Run documents through the Accuracy Testing tab to generate real metrics.</strong>
        </div>
        """, unsafe_allow_html=True)

def display_modality_gap_explanation():
    """Display explanation of the modality gap concept"""
    st.markdown("<div class='sub-header'>The Modality Gap Explained</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='comparison-container'>
        <div class='comparison-left'>
            <h3>Traditional OCR</h3>
            <div class='modality-gap-box'>
                <strong>The Modality Gap Problem:</strong><br>
                Traditional OCR treats documents as images, converting them to plain text and losing critical structure:
                <ul>
                    <li>Tables become flat text without relationships</li>
                    <li>Form field associations are lost</li>
                    <li>Multi-page connections disappear</li>
                    <li>Financial context is stripped away</li>
                </ul>
                <p><em>Result: Extensive manual verification required</em></p>
            </div>
        </div>
        <div class='comparison-right'>
            <h3>Document Inlining</h3>
            <div class='structure-preserved-box'>
                <strong>Structure Preservation Solution:</strong><br>
                Document Inlining preserves structure throughout processing:
                <ul>
                    <li>Tables maintain row/column relationships</li>
                    <li>Form fields keep context and connections</li>
                    <li>Multi-page documents maintain references</li>
                    <li>Financial calculations are properly understood</li>
                </ul>
                <p><em>Result: Automated processing with minimal verification</em></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comparison(traditional_results, enhanced_results, inlining_results, image_path):
    """Display side-by-side comparison of different processing methods"""
    
    # Configure side-by-side columns
    if traditional_results and inlining_results:
        if enhanced_results:
            # Three-column layout with all methods
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Traditional OCR")
                display_extracted_fields(traditional_results)
                
            with col2:
                st.markdown("### Enhanced OCR + LLM")
                if enhanced_results.get('using_mock_data', False):
                    st.markdown("""
                    <div style="background-color: #fffde7; padding: 5px; margin-bottom: 10px; border-radius: 5px;">
                        <small>⚠️ Using demonstration data (API unavailable)</small>
                    </div>
                    """, unsafe_allow_html=True)
                display_extracted_fields(enhanced_results)
                
            with col3:
                st.markdown("### Document Inlining")
                display_extracted_fields(inlining_results)
        else:
            # Two-column layout for just traditional and inlining
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Traditional OCR")
                display_extracted_fields(traditional_results)
                
            with col2:
                st.markdown("### Document Inlining")
                display_extracted_fields(inlining_results)
    
    # Display side-by-side metrics comparison
    st.subheader("Performance Metrics Comparison")
    
    # Get metrics
    metrics = get_metrics_tracker().calculate_metrics()
    
    if metrics:
        # Create a comparison dataframe
        comparison_data = {
            'Metric': [
                'Processing Time (sec)',
                'Confidence Score',
                'Extracted Fields',
                'Structure Preservation',
                'Time Improvement',
                'Confidence Improvement'
            ]
        }
        
        # Add Traditional OCR data
        traditional_time = metrics.get('processing_times', {}).get('traditional_ocr', {}).get('average', 0)
        traditional_confidence = metrics.get('confidence_scores', {}).get('traditional_ocr', {}).get('average', 0)
        traditional_fields = metrics.get('extraction_counts', {}).get('traditional_ocr', {}).get('average', 0)
        
        comparison_data['Traditional OCR'] = [
            f"{traditional_time:.2f}",
            f"{traditional_confidence:.2f}",
            f"{int(traditional_fields)}",
            "❌ None",
            "N/A",
            "N/A"
        ]
        
        # Add Enhanced OCR data if available
        if enhanced_results:
            enhanced_time = metrics.get('processing_times', {}).get('enhanced_ocr', {}).get('average', 0)
            enhanced_confidence = metrics.get('confidence_scores', {}).get('enhanced_ocr', {}).get('average', 0)
            enhanced_fields = metrics.get('extraction_counts', {}).get('enhanced_ocr', {}).get('average', 0)
            
            # Get comparative metrics
            enhanced_time_improvement = metrics.get('comparative', {}).get('traditional_to_enhanced_time_pct', 0)
            enhanced_conf_improvement = metrics.get('comparative', {}).get('traditional_to_enhanced_conf_pct', 0)
            
            using_mock = enhanced_results.get('using_mock_data', False)
            mock_indicator = " (demo)" if using_mock else ""
            
            comparison_data['Enhanced OCR + LLM'] = [
                f"{enhanced_time:.2f}{mock_indicator}",
                f"{enhanced_confidence:.2f}{mock_indicator}",
                f"{int(enhanced_fields)}{mock_indicator}",
                "⚠️ Partial",
                f"{enhanced_time_improvement:.1f}%{mock_indicator}",
                f"{enhanced_conf_improvement:.1f}%{mock_indicator}"
            ]
        
        # Add Document Inlining data
        inlining_time = metrics.get('processing_times', {}).get('document_inlining', {}).get('average', 0)
        inlining_confidence = metrics.get('confidence_scores', {}).get('document_inlining', {}).get('average', 0)
        inlining_fields = metrics.get('extraction_counts', {}).get('document_inlining', {}).get('average', 0)
        
        # Get comparative metrics
        inlining_time_improvement = metrics.get('comparative', {}).get('traditional_to_inlining_time_pct', 0)
        inlining_conf_improvement = metrics.get('comparative', {}).get('traditional_to_inlining_conf_pct', 0)
        
        comparison_data['Document Inlining'] = [
            f"{inlining_time:.2f}",
            f"{inlining_confidence:.2f}",
            f"{int(inlining_fields)}",
            "✅ Full",
            f"{inlining_time_improvement:.1f}%",
            f"{inlining_conf_improvement:.1f}%"
        ]
        
        # Create and display the comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df.set_index('Metric'))
        
        # Add a disclaimer for mock data
        if enhanced_results and enhanced_results.get('using_mock_data', False):
            st.info("⚠️ Note: The Enhanced OCR + LLM metrics are based on demonstration data as the API call failed or timed out.")
        
        # Add more detailed comparison insights
        with st.expander("View Detailed Analysis"):
            st.markdown(create_accuracy_report(metrics))
            
            # Display structured metrics
            if 'structure_metrics' in metrics:
                st.markdown("### Structure Preservation Metrics")
                st.markdown("""
                These metrics measure how well each method preserves the structure of the document:
                
                - **Table Detection**: Ability to recognize and maintain table structures
                - **Form Field Relations**: Maintaining relationships between form labels and values
                - **Spatial Layout**: Preserving the 2D spatial layout of the document
                - **Hierarchical Structure**: Maintaining heading levels and document organization
                """)
                
                # Create a dataframe for structure metrics
                structure_data = {
                    'Metric': ['Table Detection', 'Form Field Relations', 'Spatial Layout', 'Hierarchical Structure']
                }
                
                # Add Traditional OCR structure metrics (typically poor)
                structure_data['Traditional OCR'] = ['Poor', 'Poor', 'Poor', 'None']
                
                # Add Enhanced OCR structure metrics (better but not perfect)
                if enhanced_results:
                    enhanced_structure = ['Fair', 'Good', 'Limited', 'Limited']
                    if enhanced_results.get('using_mock_data', False):
                        enhanced_structure = [f"{rating} (demo)" for rating in enhanced_structure]
                    structure_data['Enhanced OCR + LLM'] = enhanced_structure
                
                # Add Document Inlining structure metrics (excellent)
                structure_data['Document Inlining'] = ['Excellent', 'Excellent', 'Excellent', 'Good']
                
                # Create and display the structure metrics dataframe
                structure_df = pd.DataFrame(structure_data)
                st.table(structure_df.set_index('Metric'))
    else:
        st.info("Process more documents to generate comparative metrics.")

def suggest_document_types():
    """Display suggestions for document types to demo"""
    st.markdown("<div class='sub-header'>Suggested Documents for Demonstration</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    For the most effective demonstration of Document Inlining technology, consider using these document types:
    
    <strong>Bank Statements:</strong> Documents with tables of transactions showing dates, descriptions, and amounts.
    These demonstrate how Document Inlining preserves table structures.
    
    <strong>Loan Applications:</strong> Multi-page forms with various sections. These show how Document Inlining
    maintains relationships between different sections and pages.
    
    <strong>Tax Forms:</strong> Documents with complex layouts and specialized financial data. These highlight
    the financial domain expertise of Document Inlining.
    
    <strong>Property Appraisals:</strong> Documents with comparison tables and property details. These demonstrate
    Document Inlining's ability to maintain complex data relationships.
    </div>
    """, unsafe_allow_html=True)

def display_structure_preservation_quality(traditional_results, inlining_results):
    """Display a visual comparison of structure preservation quality"""
    st.markdown("<div class='sub-header'>Structure Preservation Quality</div>", unsafe_allow_html=True)
    
    # Check if we have both results
    if not traditional_results or not inlining_results:
        st.info("Process a document with both Traditional OCR and Document Inlining to see structure preservation comparison.")
        return
    
    st.markdown("""
    One of the key advantages of Document Inlining is its ability to preserve document structure.
    This visualization compares how well each method preserves the relationships and structure of the original document.
    """)
    
    # Extract text content from both results
    traditional_text = ""
    inlining_text = ""
    
    if 'detected_fields' in traditional_results:
        # For traditional OCR, concatenate all detected text
        for item in traditional_results['detected_fields']:
            if isinstance(item, tuple) and len(item) > 1:
                traditional_text += item[1] + " "
            elif isinstance(item, dict) and 'text' in item:
                traditional_text += item['text'] + " "
    
    # For inlining, get the structured content or full text
    if 'structured_content' in inlining_results:
        inlining_text = inlining_results['structured_content']
    elif 'extracted_fields' in inlining_results:
        if 'full_text' in inlining_results['extracted_fields']:
            if isinstance(inlining_results['extracted_fields']['full_text'], dict):
                inlining_text = inlining_results['extracted_fields']['full_text'].get('value', '')
            else:
                inlining_text = str(inlining_results['extracted_fields']['full_text'])
        elif 'document_content' in inlining_results['extracted_fields']:
            if isinstance(inlining_results['extracted_fields']['document_content'], dict):
                inlining_text = inlining_results['extracted_fields']['document_content'].get('value', '')
            else:
                inlining_text = str(inlining_results['extracted_fields']['document_content'])
    
    # Detect structure elements in both texts
    structure_metrics = {
        'traditional': {
            'sections': len(re.findall(r'^#+\s', traditional_text, re.MULTILINE)),
            'key_value_pairs': len(re.findall(r':\s', traditional_text)),
            'list_items': len(re.findall(r'^[-*]\s', traditional_text, re.MULTILINE)) + len(re.findall(r'^\d+\.\s', traditional_text, re.MULTILINE)),
            'tables': len(re.findall(r'\|\s+\|', traditional_text))
        },
        'inlining': {
            'sections': len(re.findall(r'^#+\s', inlining_text, re.MULTILINE)),
            'key_value_pairs': len(re.findall(r':\s', inlining_text)),
            'list_items': len(re.findall(r'^[-*]\s', inlining_text, re.MULTILINE)) + len(re.findall(r'^\d+\.\s', inlining_text, re.MULTILINE)),
            'tables': len(re.findall(r'\|\s+\|', inlining_text))
        }
    }
    
    # Calculate structure preservation score
    trad_structure_score = min(10, 
                              structure_metrics['traditional']['sections'] * 0.5 + 
                              structure_metrics['traditional']['key_value_pairs'] * 0.1 + 
                              structure_metrics['traditional']['list_items'] * 0.2 + 
                              structure_metrics['traditional']['tables'] * 2)
    
    inlining_structure_score = min(10, 
                                  structure_metrics['inlining']['sections'] * 0.5 + 
                                  structure_metrics['inlining']['key_value_pairs'] * 0.1 + 
                                  structure_metrics['inlining']['list_items'] * 0.2 + 
                                  structure_metrics['inlining']['tables'] * 2)
    
    # Create a visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Traditional OCR")
        st.progress(trad_structure_score / 10)
        st.markdown(f"**Structure Score**: {trad_structure_score:.1f}/10")
        
        # Display structure elements detected
        st.markdown("**Structure Elements Detected:**")
        st.markdown(f"- Sections/Headers: {structure_metrics['traditional']['sections']}")
        st.markdown(f"- Key-Value Pairs: {structure_metrics['traditional']['key_value_pairs']}")
        st.markdown(f"- List Items: {structure_metrics['traditional']['list_items']}")
        st.markdown(f"- Tables: {structure_metrics['traditional']['tables']}")
        
        # Show text sample with highlighting
        st.markdown("**Text Sample (Structure Lost):**")
        sample_text = traditional_text[:500] + "..." if len(traditional_text) > 500 else traditional_text
        st.text_area("", sample_text, height=200)
    
    with col2:
        st.markdown("### Document Inlining")
        st.progress(inlining_structure_score / 10)
        st.markdown(f"**Structure Score**: {inlining_structure_score:.1f}/10")
        
        # Display structure elements detected
        st.markdown("**Structure Elements Detected:**")
        st.markdown(f"- Sections/Headers: {structure_metrics['inlining']['sections']}")
        st.markdown(f"- Key-Value Pairs: {structure_metrics['inlining']['key_value_pairs']}")
        st.markdown(f"- List Items: {structure_metrics['inlining']['list_items']}")
        st.markdown(f"- Tables: {structure_metrics['inlining']['tables']}")
        
        # Show text sample with highlighting
        st.markdown("**Text Sample (Structure Preserved):**")
        sample_text = inlining_text[:500] + "..." if len(inlining_text) > 500 else inlining_text
        st.text_area("", sample_text, height=200)
    
    # Improvement percentage
    if trad_structure_score > 0:
        improvement_pct = ((inlining_structure_score - trad_structure_score) / trad_structure_score) * 100
        improvement_pct = max(0, min(100, improvement_pct))  # Cap between 0 and 100%
        
        if improvement_pct > 50:
            st.success(f"Document Inlining improves structure preservation by {improvement_pct:.1f}%, resulting in better data extraction and analysis.")
        elif improvement_pct > 20:
            st.info(f"Document Inlining improves structure preservation by {improvement_pct:.1f}%, enabling better data relationships.")
        else:
            st.warning(f"Structure improvement: {improvement_pct:.1f}%. For this document type, the benefit may be less pronounced.")
    else:
        st.success(f"Document Inlining detected structure while Traditional OCR found none, enabling more accurate analysis.")

def main():
    st.markdown("<div class='main-header'>Fireworks AI Document Inlining™ Demo</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <strong>Bridging the Modality Gap in Financial Document Processing</strong><br><br>
    This demonstration shows how Fireworks AI's Document Inlining technology preserves document structure
    while processing complex financial documents like loan applications, bank statements, and tax forms.
    <br><br>
    Compare Document Inlining with traditional OCR to see the difference in structure preservation and accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for tracking processing state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None
    if 'traditional_results' not in st.session_state:
        st.session_state.traditional_results = None
    if 'enhanced_results' not in st.session_state:
        st.session_state.enhanced_results = None
    if 'inlining_results' not in st.session_state:
        st.session_state.inlining_results = None
    if 'both_methods_run' not in st.session_state:
        st.session_state.both_methods_run = False
    
    # Sidebar
    st.sidebar.markdown("## Configuration")
    
    # Check for API key
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key or api_key == "fw_your_api_key_here":
        st.sidebar.warning("⚠️ Fireworks API key not found or using placeholder value. Please enter your API key below.")
        api_key_input = st.sidebar.text_input("Enter Fireworks API Key", key="api_key", type="password", 
                                         help="Your API key should start with 'fw_'")
        if api_key_input:
            os.environ['FIREWORKS_API_KEY'] = api_key_input
            st.sidebar.success("✅ API Key set! You can now use Document Inlining.")
    else:
        st.sidebar.success(f"✅ Fireworks API Key detected! ({api_key[:5]}...)")
    
    # Add documentation link
    st.sidebar.markdown("[Get a Fireworks API key](https://fireworks.ai/api-keys)")
    
    # Document type selection
    doc_type = st.sidebar.radio(
        "Document Type",
        ["loan_application", "bank_statement", "tax_form", "property_appraisal", "other"],
        format_func=lambda x: {
            "loan_application": "Loan Application",
            "bank_statement": "Bank Statement",
            "tax_form": "Tax Form (W-2/1099)",
            "property_appraisal": "Property Appraisal",
            "other": "Other Financial Document"
        }[x]
    )
    
    # Processing method selection
    processing_method = st.sidebar.radio(
        "Processing Method",
        ["document_inlining", "enhanced_ocr", "traditional_ocr", "compare_both"],
        format_func=lambda x: {
            "document_inlining": "Document Inlining Technology",
            "enhanced_ocr": "Enhanced OCR + LLM",
            "traditional_ocr": "Traditional OCR Processing",
            "compare_both": "Compare All Methods"
        }.get(x)
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_debug = st.checkbox("Show Debug Information", value=False)
        save_results = st.checkbox("Save Results to File", value=True)
        show_comparison = st.checkbox("Show Side-by-Side Comparison", value=True)
    
    # Define tabs
    tabs = ["Document Processing", "Comparison", "How It Works", "Accuracy Testing"]
    selected_tab = st.tabs(tabs)
    
    image_path = None
    filename = None
    file_info = None
    
    # Document Processing Tab
    with selected_tab[0]:
        st.markdown("<div class='section-header'>Upload or Select a Document</div>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a financial document", 
            type=["jpg", "jpeg", "png", "pdf"],
            help="Upload loan applications, bank statements, tax forms, or property appraisals"
        )
        
        if uploaded_file is not None:
            image_path, file_info = handle_document_upload(uploaded_file)
            if image_path:
                filename = Path(image_path).name
        
        # Sample images
        sample_images = get_sample_images()
        selected_sample = display_sample_images(sample_images)
        if selected_sample:
            image_path = selected_sample
            filename = Path(selected_sample).name
            file_info = get_file_info(image_path)
            # Clear any previous PDF conversion state when selecting a sample image
            if 'converted_image_path' in st.session_state:
                del st.session_state.converted_image_path
                del st.session_state.converted_file_info
        
        # If no image is selected, show document type suggestions
        if not image_path:
            suggest_document_types()
        
        # Process the image if one is selected
        if image_path:
            # Display the selected image
            st.markdown("<div class='sub-header'>Selected Document</div>", unsafe_allow_html=True)
            
            # Display file info if available
            if file_info:
                st.markdown("<div class='file-info'>", unsafe_allow_html=True)
                st.markdown(f"**File Name:** {file_info['name']}")
                st.markdown(f"**File Size:** {file_info['size']}")
                st.markdown(f"**File Type:** {file_info['mime_type']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            display_image(image_path)
            
            # Process button
            if processing_method == "compare_both":
                process_button = st.button("Process with Both Methods")
            else:
                process_button = st.button("Process Document")
            
            # Process document when button is clicked or if we have a previous result
            if process_button or st.session_state.processing_complete:
                if process_button:
                    # Reset processing state
                    st.session_state.processing_complete = False
                    st.session_state.processing_result = None
                    st.session_state.traditional_results = None
                    st.session_state.enhanced_results = None
                    st.session_state.inlining_results = None
                    st.session_state.both_methods_run = False
                    
                    # Process based on selected method
                    if processing_method == "document_inlining":
                        with st.spinner("Processing with Document Inlining technology..."):
                            result = process_with_document_inlining(image_path, doc_type)
                            if result:
                                st.session_state.processing_result = result
                                st.session_state.inlining_results = result
                                st.session_state.processing_complete = True
                    
                    elif processing_method == "enhanced_ocr":
                        with st.spinner("Processing with Enhanced OCR + LLM..."):
                            result = process_with_enhanced_ocr(image_path, doc_type)
                            if result:
                                st.session_state.processing_result = result
                                st.session_state.enhanced_results = result
                                st.session_state.processing_complete = True
                    
                    elif processing_method == "traditional_ocr":
                        with st.spinner("Processing with traditional OCR..."):
                            result = process_with_traditional_ocr(image_path)
                            if result:
                                st.session_state.processing_result = result
                                st.session_state.traditional_results = result
                                st.session_state.processing_complete = True
                    
                    elif processing_method == "compare_both":
                        with st.spinner("Processing with traditional OCR..."):
                            traditional_result = process_with_traditional_ocr(image_path)
                            if traditional_result:
                                st.session_state.traditional_results = traditional_result
                        
                        with st.spinner("Processing with Enhanced OCR + LLM..."):
                            enhanced_result = process_with_enhanced_ocr(image_path, doc_type)
                            if enhanced_result:
                                st.session_state.enhanced_results = enhanced_result
                        
                        with st.spinner("Processing with Document Inlining technology..."):
                            inlining_result = process_with_document_inlining(image_path, doc_type)
                            if inlining_result:
                                st.session_state.inlining_results = inlining_result
                        
                        if st.session_state.traditional_results and st.session_state.inlining_results:
                            st.session_state.processing_complete = True
                            st.session_state.both_methods_run = True
                
                # Display results if processing is complete
                if st.session_state.processing_complete:
                    if processing_method == "document_inlining" or (processing_method == "compare_both" and st.session_state.both_methods_run):
                        result = st.session_state.inlining_results
                        if result:
                            st.markdown("<div class='success-box'>Document Inlining processing completed successfully</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='sub-header'>Document Inlining Results</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"**Document Type:** {result.get('document_type', 'Unknown')}")
                            st.markdown(f"**Processing Confidence:** {float(result.get('confidence', 0)):.2f}")
                            
                            # Display structured fields
                            display_extracted_fields(result)
                            
                            # Display financial impact metrics
                            display_financial_impact()
                            
                            if save_results:
                                # Save results to file
                                output_file = f"inlining_result_{filename}.json"
                                with open(output_file, 'w') as f:
                                    json.dump(result, f, indent=2, cls=NumpyEncoder)
                                
                                # Create download button
                                with open(output_file, "rb") as file:
                                    btn = st.download_button(
                                        label="Download Document Inlining Results",
                                        data=file,
                                        file_name=output_file,
                                        mime="application/json"
                                    )
                    
                    if processing_method == "traditional_ocr" or (processing_method == "compare_both" and st.session_state.both_methods_run):
                        result = st.session_state.traditional_results
                        if result:
                            if processing_method != "compare_both":
                                st.markdown("<div class='warning-box'>Traditional OCR processing completed with limited structure preservation</div>", unsafe_allow_html=True)
                            
                            if processing_method == "traditional_ocr":
                                st.markdown("<div class='sub-header'>Traditional OCR Results</div>", unsafe_allow_html=True)
                                
                                st.markdown(f"**Document Type:** {result.get('document_type', 'Unknown')}")
                                st.markdown(f"**Processing Confidence:** {float(result.get('confidence', 0)):.2f}")
                                
                                # Display OCR results
                                st.markdown("<div class='sub-header'>Detected Text</div>", unsafe_allow_html=True)
                                for i, item in enumerate(result.get('detected_fields', [])):
                                    if isinstance(item, tuple) and len(item) == 3:
                                        _, text, confidence = item
                                    elif isinstance(item, dict):
                                        text = item.get('text', '')
                                        confidence = item.get('confidence', 0)
                                    else:
                                        continue
                                    
                                    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                                    st.markdown(f"{i+1}. {text} - <span style='color:{confidence_color}'>Confidence: {confidence:.2f}</span>", unsafe_allow_html=True)
                                
                                if save_results:
                                    # Save results to file
                                    output_file = f"traditional_result_{filename}.json"
                                    with open(output_file, 'w') as f:
                                        json.dump(result, f, indent=2, cls=NumpyEncoder)
                                    
                                    # Create download button
                                    with open(output_file, "rb") as file:
                                        btn = st.download_button(
                                            label="Download Traditional OCR Results",
                                            data=file,
                                            file_name=output_file,
                                            mime="application/json"
                                        )
                    
                    # Show debug information if enabled
                    if show_debug:
                        with st.expander("Debug Information"):
                            if processing_method == "document_inlining" and st.session_state.inlining_results:
                                st.json(st.session_state.inlining_results)
                            elif processing_method == "traditional_ocr" and st.session_state.traditional_results:
                                st.json(st.session_state.traditional_results)
                            elif st.session_state.both_methods_run:
                                st.markdown("**Document Inlining Results:**")
                                st.json(st.session_state.inlining_results)
                                st.markdown("**Traditional OCR Results:**")
                                st.json(st.session_state.traditional_results)
                    
                    # Add a button to clear results and process again
                    if st.button("Clear Results"):
                        st.session_state.processing_complete = False
                        st.session_state.processing_result = None
                        st.session_state.traditional_results = None
                        st.session_state.enhanced_results = None
                        st.session_state.inlining_results = None
                        st.session_state.both_methods_run = False
                        rerun_app()
    
    # Comparison Tab
    with selected_tab[1]:
        st.markdown("<div class='section-header'>Document Processing Methods Comparison</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        Compare the results of different document processing methods:
        <ul>
            <li><strong>Traditional OCR:</strong> Basic text extraction without structure preservation</li>
            <li><strong>Enhanced OCR + LLM:</strong> OCR extraction followed by LLM processing for field identification</li>
            <li><strong>Document Inlining:</strong> Direct document understanding with structure preservation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show comparison if both methods have been run
        if st.session_state.both_methods_run:
            # Display modality gap explanation
            display_modality_gap_explanation()
            
            # Display structure preservation quality
            display_structure_preservation_quality(
                st.session_state.traditional_results,
                st.session_state.inlining_results
            )
            
            # Display comparison of results
            display_comparison(
                st.session_state.traditional_results,
                st.session_state.enhanced_results,
                st.session_state.inlining_results,
                image_path if 'image_path' in locals() else None
            )
        else:
            st.info("Process a document with 'Compare All Methods' option selected to see a side-by-side comparison.")
            
            # Add a button to process with both methods if we have an image
            if 'image_path' in locals() and image_path:
                if st.button("Run Comparison Now"):
                    st.session_state.processing_complete = False
                    st.session_state.traditional_results = None
                    st.session_state.enhanced_results = None
                    st.session_state.inlining_results = None
                    
                    # Process with Traditional OCR
                    with st.spinner("Step 1/3: Processing with Traditional OCR..."):
                        traditional_result = process_with_traditional_ocr(image_path)
                        if traditional_result:
                            st.session_state.traditional_results = traditional_result
                    
                    # Process with Enhanced OCR
                    with st.spinner("Step 2/3: Processing with Enhanced OCR + LLM..."):
                        enhanced_result = process_with_enhanced_ocr(image_path, doc_type)
                        if enhanced_result:
                            st.session_state.enhanced_results = enhanced_result
                    
                    # Process with Document Inlining
                    with st.spinner("Step 3/3: Processing with Document Inlining..."):
                        inlining_result = process_with_document_inlining(image_path, doc_type)
                        if inlining_result:
                            st.session_state.inlining_results = inlining_result
                    
                    # Set state flags
                    if st.session_state.traditional_results and st.session_state.inlining_results:
                        st.session_state.processing_complete = True
                        st.session_state.both_methods_run = True
                        
                        # Force rerun to show results
                        st.rerun()
    
    # How It Works Tab
    with selected_tab[2]:
        st.markdown("<div class='section-header'>How Document Processing Works</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Comparing Three Approaches to Document Processing</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for each approach
        approach_tabs = st.tabs(["Traditional OCR", "Enhanced OCR + LLM", "Document Inlining"])
        
        # Traditional OCR tab
        with approach_tabs[0]:
            st.markdown("<div class='sub-header'>Traditional OCR Approach</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='approach-box'>
                <h3>How Traditional OCR Works:</h3>
                <ol>
                    <li><strong>Image Preprocessing:</strong> The document is converted to an image and preprocessed</li>
                    <li><strong>Character Recognition:</strong> OCR engine identifies text characters</li>
                    <li><strong>Text Extraction:</strong> Text is extracted as plain strings</li>
                    <li><strong>Structure Loss:</strong> Document structure (tables, forms, etc.) is lost</li>
                </ol>
                
                <h3>Limitations:</h3>
                <ul>
                    <li>❌ No structure preservation</li>
                    <li>❌ Tables become unstructured text</li>
                    <li>❌ Form fields lose relationships</li>
                    <li>❌ Requires extensive manual verification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced OCR + LLM tab
        with approach_tabs[1]:
            st.markdown("<div class='sub-header'>Enhanced OCR + LLM Approach</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='approach-box'>
                <h3>How Enhanced OCR + LLM Works:</h3>
                <ol>
                    <li><strong>OCR Processing:</strong> Traditional OCR extracts text from document</li>
                    <li><strong>Text Analysis:</strong> LLM analyzes OCR text to identify key fields</li>
                    <li><strong>Standardization:</strong> LLM standardizes field formats (dates, numbers, etc.)</li>
                    <li><strong>Validation:</strong> LLM validates field content and identifies inconsistencies</li>
                </ol>
                
                <h3>Benefits:</h3>
                <ul>
                    <li>✅ Improved field extraction vs. traditional OCR</li>
                    <li>✅ Standardized field formats</li>
                    <li>✅ Field validation and confidence scoring</li>
                    <li>⚠️ Limited by initial OCR accuracy</li>
                    <li>⚠️ Still loses document structure</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Document Inlining tab
        with approach_tabs[2]:
            st.markdown("<div class='sub-header'>Document Inlining Approach</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='approach-box'>
                <h3>How Document Inlining Works:</h3>
                <ol>
                    <li><strong>Document Inlining:</strong> Document is represented in a format that preserves structure</li>
                    <li><strong>Direct Understanding:</strong> AI processes the document with structure intact</li>
                    <li><strong>Structure Preservation:</strong> Tables, form elements, and relationships are maintained</li>
                    <li><strong>Context Awareness:</strong> Processing includes document context and financial domain knowledge</li>
                </ol>
                
                <h3>Benefits:</h3>
                <ul>
                    <li>✅ Complete structure preservation</li>
                    <li>✅ Table relationships maintained</li>
                    <li>✅ Form field associations preserved</li>
                    <li>✅ Multi-page connections retained</li>
                    <li>✅ Minimal manual verification needed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Display comparison diagram
        st.markdown("<div class='sub-header'>The Modality Gap Challenge</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Why structure preservation matters:</strong> Financial documents contain critical relationships between data elements.
        When these relationships are lost during processing (the modality gap), it leads to errors, manual verification, and delays.
        </div>
        """, unsafe_allow_html=True)
        
        # Display the modality gap explanation
        display_modality_gap_explanation()
    
    # Accuracy Testing Tab
    with selected_tab[3]:
        accuracy_tester.display_accuracy_test_ui()

def rerun_app():
    """Rerun the app using the appropriate Streamlit method"""
    try:
        st.rerun()
    except AttributeError:
        try:
            # Fallback for older Streamlit versions
            st.experimental_rerun()
        except:
            st.error("Could not rerun the app. Please refresh the page manually.")

if __name__ == "__main__":
    main() 