# Document Processing Streamlit Application

This Streamlit application provides a user-friendly interface for the document processing pipeline. It allows you to upload document images (passports, driver's licenses) and process them using different methods.

## Features

- **Multiple Processing Methods**:
  - Full Pipeline (OCR + LLM + Validation)
  - Direct Vision (LLM Only)
  - OCR Only

- **Interactive UI**:
  - Upload images directly through the browser
  - Use sample images from the test directory
  - View processing results with visualizations
  - Download structured results as JSON

- **Document Types**:
  - Passports
  - Driver's Licenses
  - Other document types

- **Supported File Formats**:
  - Images (JPG, JPEG, PNG)
  - PDF documents (with page selection)

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection (for API calls to Fireworks AI)
- System dependencies:
  - **libmagic**: For file type detection
    - macOS: `brew install libmagic`
    - Ubuntu/Debian: `sudo apt-get install libmagic1`
    - Windows: See [python-magic documentation](https://github.com/ahupp/python-magic#dependencies)
  - **Poppler**: For PDF processing
    - macOS: `brew install poppler`
    - Ubuntu/Debian: `sudo apt-get install poppler-utils`
    - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies:
   - For PDF support:
     ```bash
     # macOS
     brew install poppler libmagic
     
     # Ubuntu/Debian
     sudo apt-get install poppler-utils libmagic1
     ```

4. Set up your Fireworks AI API key:
   
   Create a `.env` file in the root directory with the following content:
   ```
   FIREWORKS_API_KEY=your_api_key_here
   ```
   
   Alternatively, you can enter your API key directly in the Streamlit app.

5. Download sample images (optional):
   ```bash
   python download_sample_images.py
   ```
   This will download sample passport and driver's license images to the `test` directory.

## Running the App

Run the Streamlit app with:

```bash
streamlit run app.py
```

Or use the provided launcher scripts:

```bash
# On macOS/Linux
./run_app.sh

# On Windows
run_app.bat
```

This will start the application and open it in your default web browser (typically at http://localhost:8501).

## Usage Instructions

1. **Select Configuration**:
   - Choose the document type (Passport, Driver's License, or Other)
   - Select the processing method
   - Configure advanced options if needed

2. **Select an Image or Document**:
   - **Upload Your Document**: Click "Browse files" to upload an image or PDF
   - **Use Sample Images**: Select one of the sample images from the test directory
   - Supported formats: JPG, JPEG, PNG, PDF

3. **For PDF Documents**:
   - Select the page you want to process
   - Preview the page
   - Click "Use This Page" to convert it to an image for processing

4. **Process Document**:
   - Click the "Process Document" button
   - View the results displayed on the page

5. **Download Results**:
   - Click "Download Results" to save the structured data as a JSON file

## Processing Methods Explained

### Full Pipeline
- Uses the complete document processing pipeline
- Includes document classification, OCR, LLM processing, and validation
- For passports, includes MRZ processing and cross-validation
- Provides the most comprehensive analysis

### Direct Vision
- Sends the image directly to the LLM (DeepSeek V3)
- Bypasses OCR preprocessing
- Faster but may be less accurate for complex documents

### OCR Only
- Uses EasyOCR to extract text from the document
- No LLM enhancement or validation
- Useful for basic text extraction

## Sample Images

The application comes with sample images that can be downloaded using the `download_sample_images.py` script. These images include:

- **Passports**:
  - Sample US Passport
  - Sample US Passport Card

- **Driver's Licenses**:
  - Sample Enhanced Driver's License
  - Sample Standard Driver's License

These images are sourced from public domain or creative commons sources and are intended for testing purposes only.

## Working with Different Document Types

### PDF Documents
The application can process PDF documents by:
1. Extracting individual pages as images
2. Allowing you to select which page to process
3. Converting the selected page to an image for OCR and analysis

This is particularly useful for multi-page documents like passport scans, ID collections, or application forms.

### Other Document Types
For document types other than passports and driver's licenses:
1. Select "Other Document" as the document type
2. The system will attempt to extract text and identify fields based on general patterns
3. Results may vary depending on the document structure

## Troubleshooting

- **API Key Issues**: Ensure your Fireworks AI API key is correctly set in the `.env` file or entered in the app
- **Image Processing Errors**: Try using a clearer image or a different processing method
- **Missing Dependencies**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Sample Images Not Showing**: Run `python download_sample_images.py` to download sample images
- **PDF Processing Issues**: Ensure Poppler is installed on your system
- **File Type Detection Issues**: Ensure libmagic is installed on your system

## License

[Include your license information here] 